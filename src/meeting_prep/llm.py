"""Model access. One small interface, two implementations:

- BedrockLLM: Amazon Bedrock through LangChain's ChatBedrockConverse, with the
  output forced through tool use into a Pydantic schema (and an optional
  Bedrock Guardrail attached to every call).
- ScriptedLLM: deterministic stand-in so tests, evals and CI run with no AWS
  account. It reads the same prompts the real model gets.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from .models import Brief, CitedPoint, EmailAnalysis, OfferPick

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for model call failures the graph knows how to degrade from."""


class StructuredOutputError(LLMError):
    """The model answered, but not in the required shape."""


class LLMUnavailableError(LLMError):
    """Bedrock refused or failed the call after botocore's own retries (throttling, access, outage)."""


class StructuredLLM(Protocol):
    name: str

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]: ...


# ---- Bedrock ----------------------------------------------------------------------


class BedrockLLM:
    """Bedrock Converse via langchain-aws. Credentials come from the normal AWS chain
    (env vars, profile, or the task/instance role); nothing is read from this repo."""

    def __init__(
        self,
        model_id: str,
        region: str,
        guardrail_id: str | None = None,
        guardrail_version: str | None = None,
        client: Any = None,
        max_tokens: int = 2000,
        temperature: float | None = None,
    ):
        from botocore.config import Config
        from langchain_aws import ChatBedrockConverse

        kwargs: dict[str, Any] = {
            "model": model_id,
            "region_name": region,
            "max_tokens": max_tokens,
        }
        if temperature is not None:  # some newer Claude models on Bedrock reject temperature
            kwargs["temperature"] = temperature
        if client is not None:
            kwargs["client"] = client
        else:
            kwargs["config"] = Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=60)
        if guardrail_id:
            kwargs["guardrail_config"] = {
                "guardrailIdentifier": guardrail_id,
                "guardrailVersion": guardrail_version or "DRAFT",
                "trace": "enabled",
            }
        self.chat = ChatBedrockConverse(**kwargs)
        self.name = f"bedrock:{model_id}"

    @classmethod
    def from_env(cls) -> "BedrockLLM":
        model_id = os.environ.get("BEDROCK_MODEL_ID")
        if not model_id:
            raise RuntimeError("Set BEDROCK_MODEL_ID (see .env.example) or run with LLM_MODE=offline")
        return cls(
            model_id=model_id,
            region=os.environ.get("AWS_REGION", "us-east-1"),
            guardrail_id=os.environ.get("BEDROCK_GUARDRAIL_ID") or None,
            guardrail_version=os.environ.get("BEDROCK_GUARDRAIL_VERSION") or None,
        )

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]:
        from botocore.exceptions import BotoCoreError, ClientError
        from langchain_core.messages import HumanMessage, SystemMessage

        runnable = self.chat.with_structured_output(schema, include_raw=True)
        started = time.perf_counter()
        try:
            out = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except (ClientError, BotoCoreError) as e:
            raise LLMUnavailableError(f"{type(e).__name__}: {e}") from e
        meta: dict[str, Any] = {"latency_ms": round((time.perf_counter() - started) * 1000)}
        raw = out.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        meta["input_tokens"] = usage.get("input_tokens")
        meta["output_tokens"] = usage.get("output_tokens")
        stop = (getattr(raw, "response_metadata", None) or {}).get("stopReason")
        if stop == "guardrail_intervened":
            raise StructuredOutputError("Bedrock Guardrail intervened")
        if out.get("parsing_error") is not None or out.get("parsed") is None:
            raise StructuredOutputError(str(out.get("parsing_error") or "no structured output returned"))
        return out["parsed"], meta


# ---- Offline stand-in ----------------------------------------------------------------

_NEG = ("frustrat", "disappoint", "unacceptable", "angry", "fed up")
_POS = ("loving", "love ", "happy", "great", "loves", "thank")
_INTENT_CUES = {
    "renewal_question": ("renewal", "renew"),
    "pricing_question": ("pricing", "price"),
    "service_complaint": ("after tip-off", "issue", "not showing", "problem", "fixed", "late"),
    "event_request": ("host", "hospitality", "bring a group", "thank-you"),
    "add_seats": ("more seats", "add a few", "add seats"),
    "cancellation_risk": ("cancel", "not renew", "walk away", "looking elsewhere"),
}
_INPUT_RE = re.compile(r"<input>\n(.*)\n</input>", re.S)
_EMAIL_RE = re.compile(r"<email [^>]*>\n(.*?)\n</email>", re.S)


class ScriptedLLM:
    """Deterministic labels and drafts. `brief_faults` injects bad drafts by call
    index (e.g. ["ungrounded_number", None]) so retry and fallback paths are testable."""

    name = "scripted"

    def __init__(self, brief_faults: Sequence[str | None] = (), email_faults: Sequence[str | None] = ()):
        self.brief_faults = list(brief_faults)
        self.email_faults = list(email_faults)
        self.calls: list[str] = []
        self.prompts: list[tuple[str, str]] = []  # (schema name, user prompt) for tests

    def structured(self, schema: type[T], system: str, user: str) -> tuple[T, dict[str, Any]]:
        self.calls.append(schema.__name__)
        self.prompts.append((schema.__name__, user))
        if schema is EmailAnalysis:
            fault = self.email_faults.pop(0) if self.email_faults else None
            if fault == "invalid":
                raise StructuredOutputError("scripted invalid output")
            return self._label(user), {"latency_ms": 0}  # type: ignore[return-value]
        if schema is Brief:
            fault = self.brief_faults.pop(0) if self.brief_faults else None
            if fault == "invalid":
                raise StructuredOutputError("scripted invalid output")
            payload = json.loads(_INPUT_RE.search(user).group(1))  # type: ignore[union-attr]
            return self._apply_fault(self._draft(payload), fault), {"latency_ms": 0}  # type: ignore[return-value]
        raise TypeError(f"ScriptedLLM has no script for {schema.__name__}")

    # -- email labels
    def _label(self, user: str) -> EmailAnalysis:
        bodies = _EMAIL_RE.findall(user)
        text = " ".join(bodies).lower()
        if any(w in text for w in _NEG):
            sentiment = "negative"
        elif any(w in text for w in _POS):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        intents = [k for k, cues in _INTENT_CUES.items() if any(c in text for c in cues)]
        asks = []
        for body in bodies:
            for sentence in re.split(r"(?<=[.?!])\s+", body):
                if sentence.endswith("?") and "assistant" not in sentence.lower():
                    asks.append(" ".join(sentence.split()[:20]))
        return EmailAnalysis(
            sentiment=sentiment,  # type: ignore[arg-type]
            intents=intents or ["other"],  # type: ignore[list-item]
            asks=asks[:4],
            injection_suspected="assistant" in text,
        )

    # -- brief drafts
    def _draft(self, p: dict[str, Any]) -> Brief:
        facts = {f["id"]: f["text"] for f in p["facts"]}
        name, purpose = p["account"]["name"], p["purpose"].replace("_", " ")
        lead = {
            "crm.case": "Lead with the open case and what has been done: ",
            "tix.usage": "Usage so far: ",
            "tix.trend": "Attendance is sliding: ",
            "tix.resale": "Ask how the seats are being used: ",
            "crm.renewal": "Renewal timing: ",
            "gap.": "Data gap, confirm in the meeting: ",
            "inbox.injection": "Heads up: ",
            "inbox.sentiment": "Tone of recent email: ",
        }

        def phrase(fid: str) -> str:
            prefix = next((v for k, v in lead.items() if fid.startswith(k)), "")
            return prefix + facts[fid]

        order = list(p["must_cover"])
        for extra in ("crm.plan", "tix.usage", "inbox.ask.1", "inbox.sentiment"):
            if extra in facts and extra not in order:
                order.append(extra)
        points = [CitedPoint(text=phrase(fid), evidence=[fid]) for fid in order[:6]]

        risks = [CitedPoint(text=f"Risk driver: {facts[r['fact']]}", evidence=[r["fact"]]) for r in p["risk"]["reasons"][:4]]
        priority = {
            "service recovery": ["OFF-SERVICE-CREDIT", "OFF-FLEX-SUITE", "OFF-EARLY-RENEW"],
            "upsell": ["OFF-ADD-SEATS", "OFF-HOSP-GROUP", "OFF-EARLY-RENEW"],
            "renewal": ["OFF-SERVICE-CREDIT", "OFF-FLEX-SUITE", "OFF-EARLY-RENEW", "OFF-HOSP-GROUP", "OFF-ADD-SEATS"],
            "check in": ["OFF-HOSP-GROUP", "OFF-ADD-SEATS"],
        }[purpose]
        eligible = {o["id"]: o for o in p["eligible_offers"]}
        picks = [
            OfferPick(offer_id=oid, rationale=eligible[oid]["text"], evidence=[f"offer.{oid}"]) for oid in priority if oid in eligible
        ][:2]
        situation = facts.get("crm.renewal") or facts["crm.plan"]
        return Brief(
            headline=f"{name}: {purpose} meeting prep",
            situation=f"{name}. {situation}.",
            talking_points=points,
            risks=risks,
            recommended_offers=picks,
            questions_to_ask=["What would make next season an easy yes for your team?"],
        )

    @staticmethod
    def _apply_fault(brief: Brief, fault: str | None) -> Brief:
        if fault == "ungrounded_number":
            brief.talking_points[0].text += " Usage is up 61% year over year."
        elif fault == "big_discount":
            brief.talking_points[-1] = CitedPoint(text="Offer a 25% discount to close today.", evidence=["crm.plan"])
        elif fault == "bad_offer":
            brief.recommended_offers.append(
                OfferPick(offer_id="OFF-FREE-SUITE", rationale="Give them a free suite night", evidence=["crm.plan"])
            )
        elif fault == "unknown_evidence":
            brief.talking_points[0].evidence = ["crm.made_up"]
        elif fault == "drop_coverage":
            brief.talking_points = brief.talking_points[-1:]
        return brief
