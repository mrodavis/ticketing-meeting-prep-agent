"""Deterministic brief used when the model's draft fails checks twice (or the
model is unavailable). Plain, but every line is a fact the rep can trust."""

from __future__ import annotations

from .assess import Assessment
from .models import Brief, CitedPoint, MeetingRequest, OfferPick


def fallback_brief(request: MeetingRequest, account_name: str, a: Assessment) -> Brief:
    points = [CitedPoint(text=a.facts[fid].text, evidence=[fid]) for fid in a.must_cover[:6]]
    if not points:
        base = [fid for fid in ("crm.plan", "tix.usage", "crm.renewal") if fid in a.facts]
        points = [CitedPoint(text=a.facts[fid].text, evidence=[fid]) for fid in base] or [
            CitedPoint(text=a.facts["crm.plan"].text, evidence=["crm.plan"])
        ]
    risks = [CitedPoint(text=r["reason"], evidence=[r["fact"]]) for r in a.risk["reasons"][:4]]
    offers = [OfferPick(offer_id=o["id"], rationale=o["text"], evidence=[f"offer.{o['id']}"]) for o in a.eligible_offers[:3]]
    return Brief(
        headline=f"{account_name}: {request.purpose.value.replace('_', ' ')} prep (built from facts only)",
        situation=(
            f"Risk tier {a.risk['tier']} from code rules. The model draft did not pass checks, "
            "so this brief lists the facts that must be covered. Review before the meeting."
        ),
        talking_points=points,
        risks=risks,
        recommended_offers=offers,
        questions_to_ask=[],
    )
