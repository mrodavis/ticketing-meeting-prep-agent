"""The LangGraph workflow.

    START ─┬─ fetch_crm ───────────────────────┐
           ├─ fetch_ticketing ─────────────────┤
           ├─ fetch_events ────────────────────┼─ assess ─ draft_brief ─ check_brief ─┬─ human_review ─ publish ─ END
           └─ fetch_inbox ─ analyze_emails ────┘     │          ▲                     │      (interrupt)
                                                     │          └── one retry ────────┤
                                                     └─ END (no CRM account)          └─ fallback ─ human_review

Dependencies (sources, model, clock) arrive through LangGraph's runtime context,
so the same compiled graph runs against fixtures in CI and real clients in prod.
State holds plain JSON so any checkpointer can persist it between the draft and
the rep's decision.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt

from .assess import Assessment, assess, event_window
from .fallback import fallback_brief
from .guardrails import check_brief
from .llm import LLMError, StructuredLLM
from .models import Brief, EmailAnalysis, Fact, MeetingRequest, ReviewDecision
from .prompts import BRIEF_SYSTEM, EMAIL_SYSTEM, brief_user, email_user
from .security import injection_heuristic, redact
from .sources import (
    CrmSource,
    CrmWriter,
    EventsSource,
    InboxSource,
    NotFoundError,
    TicketingSource,
    TransientSourceError,
)

MAX_DRAFTS = 2  # first draft plus one corrective retry


@dataclass
class Deps:
    crm: CrmSource
    ticketing: TicketingSource
    events: EventsSource
    inbox: InboxSource
    writer: CrmWriter
    llm: StructuredLLM
    offers: dict[str, Any]
    today: Callable[[], date] = date.today
    max_source_attempts: int = 3
    backoff_s: float = 0.2
    sleep: Callable[[float], None] = time.sleep


class State(TypedDict, total=False):
    request: dict[str, Any]
    account: dict[str, Any] | None
    ticketing: dict[str, Any] | None
    events: list[dict[str, Any]] | None
    emails: list[dict[str, Any]]
    email_analysis: dict[str, Any] | None
    injection: bool
    redactions: list[str]
    source_errors: Annotated[list[dict[str, str]], operator.add]
    assessment: dict[str, Any]
    draft: dict[str, Any] | None
    draft_attempts: int
    violations: list[str]
    model_violations: list[str]
    brief_origin: str
    review_note: str | None
    review: dict[str, Any] | None
    crm_write: dict[str, Any] | None
    status: str
    trace: Annotated[list[dict[str, Any]], operator.add]


def _req(state: State) -> MeetingRequest:
    return MeetingRequest.model_validate(state["request"])


def _ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def _call_source(name: str, fn: Callable[[], Any], deps: Deps) -> tuple[Any, list[dict[str, Any]], dict[str, str] | None]:
    """Bounded retry with backoff for transient failures, one trace entry per attempt.
    A source that never answers becomes a data gap instead of failing the run."""
    entries: list[dict[str, Any]] = []
    for attempt in range(1, deps.max_source_attempts + 1):
        t0 = time.perf_counter()
        try:
            out = fn()
            entries.append({"node": name, "kind": "tool", "attempt": attempt, "ok": True, "ms": _ms(t0)})
            return out, entries, None
        except NotFoundError:
            entries.append({"node": name, "kind": "tool", "attempt": attempt, "ok": False, "error": "not_found", "ms": _ms(t0)})
            return None, entries, {"source": name, "error": "record not found"}
        except TransientSourceError:
            entries.append({"node": name, "kind": "tool", "attempt": attempt, "ok": False, "error": "transient", "ms": _ms(t0)})
            if attempt < deps.max_source_attempts:
                deps.sleep(deps.backoff_s * 2 ** (attempt - 1))
    return None, entries, {"source": name, "error": f"no response after {deps.max_source_attempts} attempts"}


# ---- fetch -------------------------------------------------------------------------


def fetch_crm(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    out, trace, err = _call_source("crm", lambda: deps.crm.get_account(_req(state).account_id), deps)
    return {"account": out, "trace": trace, "source_errors": [err] if err else []}


def fetch_ticketing(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    out, trace, err = _call_source("ticketing", lambda: deps.ticketing.get_activity(_req(state).account_id), deps)
    return {"ticketing": out, "trace": trace, "source_errors": [err] if err else []}


def fetch_events(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    start, end = event_window(deps.today(), _req(state).meeting_date)
    out, trace, err = _call_source("events", lambda: deps.events.upcoming(start, end), deps)
    return {"events": out, "trace": trace, "source_errors": [err] if err else []}


def fetch_inbox(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    out, trace, err = _call_source("inbox", lambda: deps.inbox.recent(_req(state).account_id), deps)
    return {"emails": out or [], "trace": trace, "source_errors": [err] if err else []}


# ---- model call 1: label email --------------------------------------------------------


def analyze_emails(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    emails = state.get("emails") or []
    if not emails:
        return {"email_analysis": None, "injection": False, "redactions": []}

    heuristic = any(injection_heuristic(m["body"]) for m in emails)
    redactions: list[str] = []
    safe = []
    for m in emails:
        body, found = redact(m["body"])
        redactions += found
        safe.append({**m, "body": body})

    trace: list[dict[str, Any]] = []
    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            analysis, meta = deps.llm.structured(EmailAnalysis, EMAIL_SYSTEM, email_user(safe))
            trace.append({"node": "analyze_emails", "kind": "llm", "attempt": attempt, "ok": True, "ms": _ms(t0), **meta})
            return {
                "email_analysis": analysis.model_dump(mode="json"),
                "injection": heuristic or analysis.injection_suspected,
                "redactions": sorted(set(redactions)),
                "trace": trace,
            }
        except LLMError as e:
            trace.append({"node": "analyze_emails", "kind": "llm", "attempt": attempt, "ok": False, "error": str(e)[:200], "ms": _ms(t0)})
    return {
        "email_analysis": None,
        "injection": heuristic,
        "redactions": sorted(set(redactions)),
        "trace": trace,
        "source_errors": [{"source": "email_analysis", "error": "model call failed twice"}],
    }


# ---- rules ---------------------------------------------------------------------------


def assess_node(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    t0 = time.perf_counter()
    if not state.get("account"):
        return {"status": "failed", "trace": [{"node": "assess", "kind": "rule", "ok": False, "error": "no CRM account", "ms": _ms(t0)}]}
    email = state.get("email_analysis")
    a = assess(
        request=_req(state),
        today=deps.today(),
        crm=state["account"],  # type: ignore[arg-type]
        ticketing=state.get("ticketing"),
        events=state.get("events"),
        email=EmailAnalysis.model_validate(email) if email else None,
        email_dates=[m["date"] for m in state.get("emails") or []],
        injection=bool(state.get("injection")),
        offers_catalog=deps.offers,
        source_errors=[e for e in state.get("source_errors", []) if e["source"] != "crm"],
    )
    return {
        "assessment": _dump_assessment(a),
        "draft_attempts": 0,
        "trace": [{"node": "assess", "kind": "rule", "ok": True, "ms": _ms(t0), "risk_tier": a.risk["tier"]}],
    }


def _dump_assessment(a: Assessment) -> dict[str, Any]:
    return {
        "facts": [f.model_dump() for f in a.facts.values()],
        "signals": a.signals,
        "risk": a.risk,
        "must_cover": a.must_cover,
        "eligible_offers": a.eligible_offers,
        "data_gaps": a.data_gaps,
    }


def _load_assessment(d: dict[str, Any]) -> Assessment:
    facts = {f["id"]: Fact.model_validate(f) for f in d["facts"]}
    return Assessment(facts, d["signals"], d["risk"], d["must_cover"], d["eligible_offers"], d["data_gaps"])


# ---- model call 2: draft the brief ------------------------------------------------------


def draft_brief(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    req, a, account = _req(state), state["assessment"], state["account"]
    attempt = state.get("draft_attempts", 0) + 1
    payload = {
        "account": {"id": req.account_id, "name": account["name"], "segment": account["segment"]},  # type: ignore[index]
        "purpose": req.purpose.value,
        "meeting_date": req.meeting_date.isoformat(),
        "risk": {"tier": a["risk"]["tier"], "reasons": a["risk"]["reasons"]},
        "facts": [{"id": f["id"], "text": f["text"]} for f in a["facts"]],
        "must_cover": a["must_cover"],
        "eligible_offers": [{"id": o["id"], "name": o["name"], "text": o["text"]} for o in a["eligible_offers"]],
        "discount_authority_pct": deps.offers["rep_discount_authority_pct"],
    }
    previous = state.get("violations") if attempt > 1 else None
    t0 = time.perf_counter()
    try:
        brief, meta = deps.llm.structured(Brief, BRIEF_SYSTEM, brief_user(payload, previous))
    except LLMError as e:
        return {
            "draft": None,
            "draft_attempts": attempt,
            "violations": [f"model_error:{type(e).__name__}"],
            "trace": [{"node": "draft_brief", "kind": "llm", "attempt": attempt, "ok": False, "error": str(e)[:200], "ms": _ms(t0)}],
        }
    return {
        "draft": brief.model_dump(mode="json"),
        "draft_attempts": attempt,
        "brief_origin": "model",
        "trace": [{"node": "draft_brief", "kind": "llm", "attempt": attempt, "ok": True, "ms": _ms(t0), **meta}],
    }


def check_node(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    deps = runtime.context
    t0 = time.perf_counter()
    if state.get("draft") is None:
        violations = state.get("violations") or ["no_draft"]
    else:
        a = _load_assessment(state["assessment"])
        violations = check_brief(
            Brief.model_validate(state["draft"]),
            a.facts,
            [o["id"] for o in a.eligible_offers],
            a.must_cover,
            deps.offers["rep_discount_authority_pct"],
        )
    update: dict[str, Any] = {
        "violations": violations,
        "trace": [{"node": "check_brief", "kind": "guard", "ok": not violations, "violations": violations, "ms": _ms(t0)}],
    }
    if violations:
        update["model_violations"] = violations
    else:
        update["status"] = "awaiting_review"
    return update


def route_after_check(state: State) -> str:
    if not state.get("violations"):
        return "human_review"
    if state.get("draft_attempts", 0) < MAX_DRAFTS:
        return "draft_brief"
    return "fallback"


def fallback_node(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    a = _load_assessment(state["assessment"])
    brief = fallback_brief(_req(state), state["account"]["name"], a)  # type: ignore[index]
    return {
        "draft": brief.model_dump(mode="json"),
        "brief_origin": "fallback",
        "violations": [],
        "status": "awaiting_review",
        "trace": [{"node": "fallback", "kind": "rule", "ok": True, "ms": 0}],
    }


# ---- human in the loop -------------------------------------------------------------------


def review_flags(state: State) -> list[str]:
    flags = []
    if state.get("brief_origin") == "fallback":
        flags.append("fallback_brief")
    flags += [f"data_gap:{g}" for g in state["assessment"]["data_gaps"]]
    if state.get("injection"):
        flags.append("injection_suspected")
    if state.get("violations"):
        flags.append("edit_has_violations")
    return flags


def human_review(state: State, runtime: Runtime[Deps]) -> Command:
    deps = runtime.context
    a = state["assessment"]
    decision_raw = interrupt(
        {
            "account": {"id": state["account"]["id"], "name": state["account"]["name"]},  # type: ignore[index]
            "risk": a["risk"],
            "brief": state["draft"],
            "brief_origin": state.get("brief_origin"),
            "violations": state.get("violations") or [],
            "flags": review_flags(state),
            "note": state.get("review_note"),
        }
    )
    d = ReviewDecision.model_validate(decision_raw)
    record = d.model_dump(mode="json", exclude={"edited_brief"})

    if d.decision == "reject":
        return Command(goto=END, update={"review": record, "status": "rejected"})

    if d.decision == "edit":
        if d.edited_brief is None:
            return Command(goto="human_review", update={"review_note": "An edit needs edited_brief."})
        assessment = _load_assessment(a)
        violations = check_brief(
            d.edited_brief,
            assessment.facts,
            [o["id"] for o in assessment.eligible_offers],
            assessment.must_cover,
            deps.offers["rep_discount_authority_pct"],
        )
        return Command(
            goto="human_review",
            update={
                "draft": d.edited_brief.model_dump(mode="json"),
                "brief_origin": "human_edit",
                "violations": violations,
                "review_note": "Edit saved. Fix the listed checks before approving." if violations else "Edit saved.",
            },
        )

    if state.get("violations"):
        return Command(goto="human_review", update={"review_note": "Approval blocked: the current brief fails checks."})
    return Command(goto="publish", update={"review": record, "status": "approved"})


def publish(state: State, runtime: Runtime[Deps]) -> dict[str, Any]:
    """Writes the approved brief to CRM. Safe under RetryPolicy because the write is
    idempotent on the meeting id."""
    deps = runtime.context
    req = _req(state)
    t0 = time.perf_counter()
    result = deps.writer.upsert_activity(
        req.meeting_id,
        req.account_id,
        {
            "type": "meeting_prep_brief",
            "meeting_date": req.meeting_date.isoformat(),
            "brief": state["draft"],
            "risk_tier": state["assessment"]["risk"]["tier"],
            "origin": state.get("brief_origin"),
            "approved_by": (state.get("review") or {}).get("reviewer"),
        },
    )
    return {
        "crm_write": {k: v for k, v in result.items() if k != "body"},
        "status": "published",
        "trace": [{"node": "publish", "kind": "tool", "ok": True, "ms": _ms(t0), "created": result["created"]}],
    }


# ---- assembly ----------------------------------------------------------------------------


def build_graph(checkpointer: Any = None, publish_retry_interval: float = 0.5):
    g = StateGraph(State, context_schema=Deps)
    g.add_node("fetch_crm", fetch_crm)
    g.add_node("fetch_ticketing", fetch_ticketing)
    g.add_node("fetch_events", fetch_events)
    g.add_node("fetch_inbox", fetch_inbox)
    g.add_node("analyze_emails", analyze_emails)
    g.add_node("assess", assess_node)
    g.add_node("draft_brief", draft_brief)
    g.add_node("check_brief", check_node)
    g.add_node("fallback", fallback_node)
    g.add_node("human_review", human_review, destinations=("publish", "human_review", END))
    g.add_node(
        "publish",
        publish,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=publish_retry_interval,
            jitter=False,
            retry_on=TransientSourceError,
        ),
    )

    for node in ("fetch_crm", "fetch_ticketing", "fetch_events", "fetch_inbox"):
        g.add_edge(START, node)
    g.add_edge("fetch_inbox", "analyze_emails")
    g.add_edge(["fetch_crm", "fetch_ticketing", "fetch_events", "analyze_emails"], "assess")
    g.add_conditional_edges("assess", lambda s: END if s.get("status") == "failed" else "draft_brief", ["draft_brief", END])
    g.add_edge("draft_brief", "check_brief")
    g.add_conditional_edges("check_brief", route_after_check, ["human_review", "draft_brief", "fallback"])
    g.add_edge("fallback", "human_review")
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer)
