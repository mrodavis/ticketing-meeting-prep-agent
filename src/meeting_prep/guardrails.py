"""Checks every drafted brief must pass before a person sees it as model output.

A failed check sends the draft back to the model once with the violations
listed; a second failure swaps in a deterministic brief built from the facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Brief, Fact

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DISCOUNT_RES = [
    re.compile(r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount|reduction|rebate|loyalty credit|credit)", re.I),
    re.compile(r"(?:discount|reduction|rebate|credit)\s+of\s+(?:up to\s+)?(\d+(?:\.\d+)?)\s*%", re.I),
]
_PROMISE_RE = re.compile(r"\b(guarantee[ds]?|we will waive|has agreed to|have agreed to)\b", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?:\(?\b\d{3}\)?[-.\s])?\b\d{3}[-.]\d{4}\b")
_LONG_DIGITS_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in _NUM_RE.findall(text):
        cleaned = raw.rstrip(",").replace(",", "")
        try:
            out.add(repr(float(cleaned)))
        except ValueError:
            continue
    return out


def brief_texts(brief: Brief) -> list[str]:
    texts = [brief.headline, brief.situation, *brief.questions_to_ask]
    texts += [p.text for p in brief.talking_points]
    texts += [r.text for r in brief.risks]
    texts += [o.rationale for o in brief.recommended_offers]
    return texts


def check_brief(
    brief: Brief,
    facts: dict[str, Fact],
    eligible_offer_ids: Iterable[str],
    must_cover: Iterable[str],
    discount_authority_pct: float,
) -> list[str]:
    violations: list[str] = []
    eligible = set(eligible_offer_ids)

    cited: set[str] = set()
    for item in [*brief.talking_points, *brief.risks]:
        for fid in item.evidence:
            if fid not in facts:
                violations.append(f"unknown_evidence:{fid}")
            cited.add(fid)
    for pick in brief.recommended_offers:
        if pick.offer_id not in eligible:
            violations.append(f"ineligible_offer:{pick.offer_id}")
        for fid in pick.evidence:
            if fid not in facts:
                violations.append(f"unknown_evidence:{fid}")

    for fid in must_cover:
        if fid not in cited:
            violations.append(f"missing_coverage:{fid}")

    grounded: set[str] = set()
    for fact in facts.values():
        grounded |= _numbers(fact.text)

    for text in brief_texts(brief):
        for n in sorted(_numbers(text) - grounded):
            violations.append(f"ungrounded_number:{float(n):g}")
        for rx in _DISCOUNT_RES:
            for m in rx.finditer(text):
                if float(m.group(1)) > discount_authority_pct:
                    violations.append(f"discount_over_authority:{m.group(1)}%")
        if m := _PROMISE_RE.search(text):
            violations.append(f"promise_language:{m.group(1).lower()}")
        if _EMAIL_RE.search(text):
            violations.append("pii:email_address")
        if _PHONE_RE.search(text):
            violations.append("pii:phone_number")
        if _LONG_DIGITS_RE.search(text):
            violations.append("pii:long_digit_sequence")

    return list(dict.fromkeys(violations))
