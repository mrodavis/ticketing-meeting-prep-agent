import pytest

from meeting_prep.guardrails import check_brief
from meeting_prep.models import Brief, CitedPoint, Fact, OfferPick

FACTS = {
    "tix.usage": Fact(
        id="tix.usage", text="Season to date: 97 of 216 issued tickets scanned (45%) across 12 home games", source="ticketing"
    ),
    "crm.renewal": Fact(
        id="crm.renewal", text="Renewal OPP-7781 is at stage Proposal: $310,000, closing 2027-01-15 (56 days away)", source="crm"
    ),
    "offer.OFF-EARLY-RENEW": Fact(
        id="offer.OFF-EARLY-RENEW", text="Early renewal: lock current pricing with a loyalty credit of up to 5%", source="system"
    ),
}
ELIGIBLE = ["OFF-EARLY-RENEW"]
MUST = ["tix.usage", "crm.renewal"]


def brief(extra_point: str | None = None, offers=None, evidence=None) -> Brief:
    points = [
        CitedPoint(text="Usage is low: 97 of 216 tickets scanned (45%).", evidence=["tix.usage"]),
        CitedPoint(text="Renewal is worth $310,000 and closes 2027-01-15, 56 days out.", evidence=evidence or ["crm.renewal"]),
    ]
    if extra_point:
        points.append(CitedPoint(text=extra_point, evidence=["tix.usage"]))
    return Brief(
        headline="Halvorsen renewal prep",
        situation="Low usage ahead of renewal.",
        talking_points=points,
        recommended_offers=offers
        if offers is not None
        else [
            OfferPick(
                offer_id="OFF-EARLY-RENEW", rationale="Lock pricing with a loyalty credit of up to 5%.", evidence=["offer.OFF-EARLY-RENEW"]
            )
        ],
    )


def check(b: Brief) -> list[str]:
    return check_brief(b, FACTS, ELIGIBLE, MUST, 5)


def test_clean_brief_passes():
    assert check(brief()) == []


@pytest.mark.parametrize(
    "point,expected",
    [
        ("Usage is up 61% from last year.", "ungrounded_number:61"),
        ("They spent $1.2M with us.", "ungrounded_number:1.2"),
        ("Offer 25% off to close today.", "discount_over_authority:25%"),
        ("Offer a discount of 10% on the suite.", "discount_over_authority:10%"),
        ("We guarantee catering will be on time.", "promise_language:guarantee"),
        ("Meridian has agreed to new terms.", "promise_language:has agreed to"),
        ("Email the contract to erik.h@halvorsen.example.", "pii:email_address"),
        ("Call Erik at 555-0141.", "ungrounded_number:555"),
    ],
)
def test_violations(point, expected):
    assert expected in check(brief(point))


def test_phone_is_flagged_as_pii_even_if_digits_were_grounded():
    facts = {**FACTS, "x": Fact(id="x", text="555 0141", source="system")}
    v = check_brief(brief("Call Erik at 555-0141."), facts, ELIGIBLE, MUST, 5)
    assert "pii:phone_number" in v


def test_discount_within_authority_is_allowed():
    assert check(brief("Mention the loyalty credit of up to 5% for early renewal.")) == []


def test_evidence_offer_and_coverage_rules():
    assert "unknown_evidence:crm.made_up" in check(brief(evidence=["crm.made_up"]))
    assert "missing_coverage:crm.renewal" in check(brief(evidence=["tix.usage"]))
    bad = [OfferPick(offer_id="OFF-FREE-SUITE", rationale="Free suite.", evidence=["tix.usage"])]
    assert "ineligible_offer:OFF-FREE-SUITE" in check(brief(offers=bad))
