"""Deterministic assessment: turn raw source data into citable facts, signals, a
risk score with reasons, the facts a brief must cover, and the eligible offers.

No model is involved here. The model's email labels come in as input, but what
they do to the score is decided by the rules below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from .models import EmailAnalysis, Fact, MeetingRequest, Purpose

SEVERITY_POINTS = {"high": 3, "medium": 1, "low": 0}
SOURCE_LABELS = {
    "crm": "CRM",
    "ticketing": "Ticketing data",
    "events": "The events calendar",
    "inbox": "The inbox",
    "email_analysis": "Email analysis",
}
EVENT_WINDOW_DAYS = 45


@dataclass
class Assessment:
    facts: dict[str, Fact]
    signals: dict[str, Any]
    risk: dict[str, Any]
    must_cover: list[str]
    eligible_offers: list[dict[str, Any]]
    data_gaps: list[str] = field(default_factory=list)


def _pct(part: int, whole: int) -> int:
    return round(100 * part / whole) if whole else 0


def assess(
    request: MeetingRequest,
    today: date,
    crm: dict[str, Any],
    ticketing: dict[str, Any] | None,
    events: list[dict[str, Any]] | None,
    email: EmailAnalysis | None,
    email_dates: list[str],
    injection: bool,
    offers_catalog: dict[str, Any],
    source_errors: list[dict[str, str]],
) -> Assessment:
    facts: dict[str, Fact] = {}
    signals: dict[str, Any] = {}
    reasons: list[dict[str, Any]] = []
    must_cover: list[str] = []
    gaps: list[str] = []

    def add(fid: str, text: str, source: str) -> str:
        facts[fid] = Fact(id=fid, text=text, source=source)  # type: ignore[arg-type]
        return fid

    def score(points: int, why: str, fid: str) -> None:
        if points:
            reasons.append({"points": points, "reason": why, "fact": fid})

    # ---- CRM -------------------------------------------------------------------------
    plan = crm["plan"]
    add(
        "crm.plan",
        f"Current plan: {plan['product']} ({plan['team']}), {plan['seats']} seats, ${plan['contract_value']:,} contract value",
        "crm",
    )
    renewal = crm.get("renewal")
    if renewal:
        close = date.fromisoformat(renewal["close_date"])
        days = (close - today).days
        signals["days_to_renewal"] = days
        fid = add(
            "crm.renewal",
            f"Renewal {renewal['opportunity_id']} is at stage {renewal['stage']}: ${renewal['amount']:,}, closing {renewal['close_date']} ({days} days away)",
            "crm",
        )
        if 0 <= days <= 45:
            score(1, "Renewal closes within 45 days", fid)
        if request.purpose == Purpose.renewal or 0 <= days <= 60:
            must_cover.append(fid)

    open_cases = []
    recent_high = False
    for case in crm.get("cases", []):
        opened = date.fromisoformat(case["opened"])
        age = (today - opened).days
        fid = add(
            f"crm.case.{case['id']}",
            f"{case['id']} ({case['severity']} severity, {case['status']}): {case['subject']}; opened {case['opened']}, {age} days ago",
            "crm",
        )
        if case["severity"] == "high" and age <= 90:
            recent_high = True
        if case["status"] == "open":
            open_cases.append(case)
            score(SEVERITY_POINTS[case["severity"]], f"Open {case['severity']} severity case", fid)
            if case["severity"] in ("high", "medium"):
                must_cover.append(fid)
    signals["open_cases"] = len(open_cases)

    # ---- Ticketing ---------------------------------------------------------------------
    if ticketing:
        issued, scanned = ticketing["issued"], ticketing["scanned"]
        usage = _pct(scanned, issued)
        signals["usage_pct"] = usage
        fid = add(
            "tix.usage",
            f"Season to date: {scanned} of {issued} issued tickets scanned ({usage}%) across {ticketing['home_games']} home games",
            "ticketing",
        )
        if usage < 50:
            score(3, "Usage below 50%", fid)
        elif usage < 75:
            score(1, "Usage between 50% and 75%", fid)
        if usage < 60:
            must_cover.append(fid)

        last4 = ticketing.get("last_4_scanned") or []
        if len(last4) == 4 and ticketing["home_games"]:
            season_avg = scanned / ticketing["home_games"]
            declining = sum(last4) / 4 < 0.75 * season_avg
            signals["usage_declining"] = declining
            fid = add(
                "tix.trend",
                f"Last 4 home games: {', '.join(str(n) for n in last4)} of {plan['seats']} seats scanned",
                "ticketing",
            )
            if declining:
                score(1, "Attendance falling over the last 4 games", fid)
                must_cover.append(fid)

        resale = _pct(ticketing["resold"], issued)
        transfer = _pct(ticketing["transferred"], issued)
        signals["resale_pct"] = resale
        fid = add(
            "tix.resale",
            f"{ticketing['transferred']} tickets transferred ({transfer}%) and {ticketing['resold']} resold on the secondary market ({resale}%)",
            "ticketing",
        )
        if resale >= 30:
            score(3, "30% or more of tickets resold", fid)
            must_cover.append(fid)
    else:
        signals["usage_pct"] = None

    # ---- Events ------------------------------------------------------------------------
    hospitality_events = []
    for ev in events or []:
        if ev["hospitality_available"] or (crm["segment"] == "suite" and ev["suite_nights_available"] > 0):
            parts = []
            if ev["hospitality_available"]:
                parts.append("hospitality space available")
            if ev["suite_nights_available"]:
                n = ev["suite_nights_available"]
                parts.append(f"{n} suite night{'s' if n != 1 else ''} open")
            add(f"evt.{ev['id']}", f"{ev['name']} on {ev['date']}: {', '.join(parts)}", "events")
            if ev["hospitality_available"]:
                hospitality_events.append(ev)

    # ---- Email (model labels, code decides what they mean) ------------------------------
    if injection:
        fid = add(
            "inbox.injection",
            "A recent client email contained instructions aimed at an AI system. It was treated as untrusted text and not acted on; confirm any terms directly with the client.",
            "system",
        )
        must_cover.append(fid)
    if email:
        latest = max(email_dates) if email_dates else "recent"
        intents = ", ".join(i.replace("_", " ") for i in email.intents) or "general"
        fid = add("inbox.sentiment", f"Latest client email ({latest}) reads {email.sentiment}; topics: {intents}", "inbox")
        if email.sentiment == "negative":
            score(2, "Negative tone in recent email", fid)
        if "cancellation_risk" in email.intents:
            score(2, "Email signals possible cancellation", fid)
            must_cover.append(fid)
        if not injection:  # asks are model-paraphrased untrusted text; drop them when the email looks hostile
            for i, ask in enumerate(email.asks, 1):
                add(f"inbox.ask.{i}", f"Client asked: {ask}", "inbox")

    # ---- Gaps --------------------------------------------------------------------------
    for err in source_errors:
        label = SOURCE_LABELS.get(err["source"], err["source"])
        fid = add(
            f"gap.{err['source']}",
            f"{label} was unavailable when this brief was prepared ({err['error']}); anything from it is missing",
            "system",
        )
        gaps.append(err["source"])
        must_cover.append(fid)

    # ---- Offers (eligibility is a rule, not a model judgment) --------------------------
    eligible = []
    usage = signals.get("usage_pct")
    for offer in offers_catalog["offers"]:
        if crm["segment"] not in offer["eligible_segments"]:
            continue
        need = offer.get("requires")
        ok = (
            need is None
            or (need == "hospitality_event" and hospitality_events)
            or (need == "usage_at_least_80" and usage is not None and usage >= 80)
            or (need == "renewal_open" and renewal and signals["days_to_renewal"] >= 0)
            or (need == "high_severity_case" and recent_high)
        )
        if ok:
            eligible.append(offer)
            add(f"offer.{offer['id']}", f"{offer['name']}: {offer['text']}", "system")

    total = sum(r["points"] for r in reasons)
    tier = "high" if total >= 6 else "medium" if total >= 3 else "low"
    drivers = "; ".join(f"{r['reason']} (+{r['points']})" for r in reasons) or "no risk rules triggered"
    add("risk.summary", f"Risk tier from code rules: {tier}, score {total}. Drivers: {drivers}", "system")
    risk = {
        "score": total,
        "tier": tier,
        "reasons": reasons,
        "complete": not gaps,
    }
    return Assessment(
        facts=facts,
        signals=signals,
        risk=risk,
        must_cover=list(dict.fromkeys(must_cover)),
        eligible_offers=eligible,
        data_gaps=gaps,
    )


def event_window(today: date, meeting: date) -> tuple[date, date]:
    return today, max(today, meeting) + timedelta(days=EVENT_WINDOW_DAYS)
