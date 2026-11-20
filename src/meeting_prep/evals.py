"""Eval harness. Runs every case in evals/cases.yaml through the real graph,
checks behavior, approves each brief, and gates CI on the results.

    python -m meeting_prep.evals --offline     # scripted model, no AWS (CI)
    python -m meeting_prep.evals --live        # Amazon Bedrock via BEDROCK_MODEL_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver

from .graph import build_graph
from .guardrails import brief_texts, check_brief
from .llm import BedrockLLM, ScriptedLLM, StructuredLLM
from .metrics import aggregate, run_metrics
from .models import Brief, Fact, MeetingRequest, ReviewDecision
from .service import decide, fixture_deps, start

MEETING_DATE = date(2026, 11, 24)
ROOT = Path(__file__).resolve().parents[2]


def check_case(case: dict[str, Any], result: dict[str, Any], llm_calls: int) -> list[str]:
    exp, state = case["expect"], result["state"]
    fails: list[str] = []

    def want(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    want(result["status"] == exp["status"], f"status {result['status']} != {exp['status']}")
    if "llm_calls" in exp:
        want(llm_calls == exp["llm_calls"], f"llm calls {llm_calls} != {exp['llm_calls']}")
    if exp["status"] == "failed":
        return fails

    a = state["assessment"]
    review = result["review"] or {}
    brief = Brief.model_validate(state["draft"])
    cited = {fid for item in [*brief.talking_points, *brief.risks] for fid in item.evidence}
    offers = {o.offer_id for o in brief.recommended_offers}
    eligible = {o["id"] for o in a["eligible_offers"]}
    text = " ".join(brief_texts(brief)).lower()
    trace = state["trace"]

    if "risk_tier" in exp:
        want(a["risk"]["tier"] == exp["risk_tier"], f"risk {a['risk']['tier']} != {exp['risk_tier']}")
    if "origin" in exp:
        want(state.get("brief_origin") == exp["origin"], f"origin {state.get('brief_origin')} != {exp['origin']}")
    for fid in exp.get("must_cite", []):
        want(fid in cited, f"brief does not cite {fid}")
    want(offers <= eligible, f"ineligible offers {sorted(offers - eligible)}")
    if "offers_any_of" in exp:
        want(bool(offers & set(exp["offers_any_of"])), f"none of {exp['offers_any_of']} offered (got {sorted(offers)})")
    for oid in exp.get("offers_excluded", []):
        want(oid not in offers and oid not in eligible, f"{oid} should not be eligible or offered")
    for bad in exp.get("forbid_text", []):
        want(bad.lower() not in text, f"forbidden text in brief: {bad!r}")
    if "injection" in exp:
        want(bool(state.get("injection")) == exp["injection"], f"injection {state.get('injection')} != {exp['injection']}")
    for gap in exp.get("data_gaps", []):
        want(gap in a["data_gaps"], f"missing data gap {gap}")
    for red in exp.get("redactions", []):
        want(red in state.get("redactions", []), f"missing redaction {red}")
    for flag in exp.get("flags", []):
        want(flag in review.get("flags", []), f"missing review flag {flag}")
    for node in exp.get("tool_retried", []):
        attempts = [t for t in trace if t["node"] == node and t["kind"] == "tool"]
        want(len(attempts) > 1 and attempts[-1]["ok"], f"{node} was not retried to success")
    if "drafts" in exp:
        drafts = sum(1 for t in trace if t["node"] == "draft_brief")
        want(drafts == exp["drafts"], f"drafts {drafts} != {exp['drafts']}")

    # Universal checks: a model draft that reached review passes every guardrail, and
    # every source that never answered shows up as a data gap.
    if state.get("brief_origin") == "model":
        facts = {f["id"]: Fact.model_validate(f) for f in a["facts"]}
        v = check_brief(brief, facts, eligible, a["must_cover"], 5)
        want(not v, f"model brief reached review with violations {v}")
    for node in {t["node"] for t in trace if t["kind"] == "tool" and t["node"] != "publish"}:
        if not any(t["ok"] for t in trace if t["node"] == node):
            want(node in a["data_gaps"], f"{node} failed silently")
    return fails


def run(live: bool) -> dict[str, Any]:
    cases = yaml.safe_load((ROOT / "evals" / "cases.yaml").read_text())
    graph = build_graph(InMemorySaver(), publish_retry_interval=0)
    report: dict[str, Any] = {"mode": "live" if live else "offline", "cases": [], "failures": 0}
    metrics = []

    for case in cases:
        faults = case.get("faults") or {}
        if live and faults:
            report["cases"].append({"id": case["id"], "skipped": "scripted faults run offline only"})
            continue
        llm: StructuredLLM = (
            BedrockLLM.from_env() if live else ScriptedLLM(brief_faults=faults.get("brief", []), email_faults=faults.get("email", []))
        )
        deps = fixture_deps(llm, backoff_s=0)
        calls_before = len(getattr(llm, "calls", []))
        req = MeetingRequest(account_id=case["account_id"], meeting_date=MEETING_DATE, purpose=case["purpose"], rep="eval")
        result = start(graph, deps, req, thread_id=f"eval-{case['id']}")
        llm_calls = len(getattr(llm, "calls", [])) - calls_before
        expect = dict(case["expect"])
        if live:
            expect.pop("llm_calls", None)  # call counting is a scripted-model feature
        fails = check_case({**case, "expect": expect}, result, llm_calls)

        if result["status"] == "awaiting_review":
            done = decide(graph, deps, result["thread_id"], ReviewDecision(decision="approve", reviewer="eval"))
            write = done["state"].get("crm_write") or {}
            if not (done["status"] == "published" and write.get("created")):
                fails.append("approval did not create a CRM activity")
            again = start(graph, deps, req, thread_id=f"eval-{case['id']}-again")
            redo = decide(graph, deps, again["thread_id"], ReviewDecision(decision="approve", reviewer="eval"))
            if (redo["state"].get("crm_write") or {}).get("created") is not False:
                fails.append("second approval for the same meeting created a duplicate CRM activity")
            result = done

        m = run_metrics(result["state"].get("trace", []))
        metrics.append(m)
        report["failures"] += bool(fails)
        report["cases"].append({"id": case["id"], "pass": not fails, "failures": fails, "metrics": m})

    report["aggregate"] = aggregate(metrics)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--offline", action="store_true", default=True)
    g.add_argument("--live", action="store_true")
    args = p.parse_args(argv)
    report = run(live=args.live)

    for c in report["cases"]:
        if "skipped" in c:
            print(f"SKIP  {c['id']}: {c['skipped']}")
        else:
            print(f"{'PASS' if c['pass'] else 'FAIL'}  {c['id']}")
            for f in c["failures"]:
                print(f"        - {f}")
    print(json.dumps(report["aggregate"], indent=2))
    out = ROOT / "artifacts" / f"eval_report_{report['mode']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
