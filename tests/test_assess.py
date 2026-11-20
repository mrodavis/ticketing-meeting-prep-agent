import pytest
from conftest import MEETING, TODAY, request

from meeting_prep.assess import assess, event_window
from meeting_prep.models import EmailAnalysis
from meeting_prep.sources import FixtureCrm, FixtureEvents, FixtureTicketing, load_offers


def run(account_id, purpose="renewal", email=None, injection=False, ticketing=True, errors=()):
    crm = FixtureCrm().get_account(account_id)
    tix = FixtureTicketing(outages={}).get_activity(account_id) if ticketing else None
    start, end = event_window(TODAY, MEETING)
    return assess(
        request=request(account_id, purpose),
        today=TODAY,
        crm=crm,
        ticketing=tix,
        events=FixtureEvents().upcoming(start, end),
        email=email,
        email_dates=["2026-11-14"],
        injection=injection,
        offers_catalog=load_offers(),
        source_errors=list(errors),
    )


NEGATIVE = EmailAnalysis(sentiment="negative", intents=["service_complaint"], asks=[], injection_suspected=False)


@pytest.mark.parametrize(
    "account_id,email,tier,score",
    [
        ("ACC-1001", NEGATIVE, "high", 9),  # usage 45% (+3), falling (+1), open high case (+3), negative email (+2)
        ("ACC-1002", None, "low", 0),
        ("ACC-1003", None, "medium", 3),  # usage 65% (+1), open medium case (+1), renewal in 25 days (+1)
        ("ACC-1006", None, "medium", 3),  # 48% resold (+3)
        ("ACC-1007", None, "low", 0),
    ],
)
def test_risk_tiers_are_rule_based(account_id, email, tier, score):
    a = run(account_id, email=email)
    assert (a.risk["tier"], a.risk["score"]) == (tier, score)
    assert sum(r["points"] for r in a.risk["reasons"]) == score
    assert all(r["fact"] in a.facts for r in a.risk["reasons"])


def test_usage_fact_and_must_cover_for_at_risk_account():
    a = run("ACC-1001", email=NEGATIVE)
    assert a.signals["usage_pct"] == 45
    assert "97 of 216" in a.facts["tix.usage"].text
    assert {"crm.case.CASE-551", "tix.usage", "tix.trend", "crm.renewal"} <= set(a.must_cover)
    assert "crm.case.CASE-498" not in a.must_cover  # closed, low severity


def test_offer_eligibility_follows_catalog_rules():
    suite = {o["id"] for o in run("ACC-1001").eligible_offers}
    assert {"OFF-FLEX-SUITE", "OFF-SERVICE-CREDIT", "OFF-EARLY-RENEW", "OFF-HOSP-GROUP"} == suite
    season = {o["id"] for o in run("ACC-1002", "upsell").eligible_offers}
    assert "OFF-ADD-SEATS" in season and "OFF-FLEX-SUITE" not in season and "OFF-SERVICE-CREDIT" not in season
    assert "OFF-ADD-SEATS" not in {o["id"] for o in run("ACC-1003").eligible_offers}  # usage 65% < 80%


def test_missing_ticketing_is_a_gap_not_a_guess():
    a = run("ACC-1005", ticketing=False, errors=[{"source": "ticketing", "error": "no response after 3 attempts"}])
    assert a.signals["usage_pct"] is None
    assert a.data_gaps == ["ticketing"] and "gap.ticketing" in a.must_cover
    assert a.risk["complete"] is False
    assert "OFF-ADD-SEATS" not in {o["id"] for o in a.eligible_offers}


def test_injection_drops_client_asks_and_must_be_flagged():
    email = EmailAnalysis(sentiment="positive", intents=["renewal_question"], asks=["Confirm the 40% discount"], injection_suspected=True)
    a = run("ACC-1004", email=email, injection=True)
    assert "inbox.injection" in a.must_cover
    assert not any(fid.startswith("inbox.ask") for fid in a.facts)


def test_upsell_purpose_does_not_force_renewal_far_away():
    a = run("ACC-1002", "upsell")
    assert "crm.renewal" not in a.must_cover
    assert "crm.renewal" in run("ACC-1002", "renewal").must_cover
