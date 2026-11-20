import sqlite3

import pytest
from conftest import request
from langgraph.checkpoint.sqlite import SqliteSaver

from meeting_prep.graph import build_graph
from meeting_prep.llm import ScriptedLLM
from meeting_prep.models import Brief, ReviewDecision
from meeting_prep.service import decide, snapshot, start


def approve(who="rep"):
    return ReviewDecision(decision="approve", reviewer=who)


def test_happy_path_stops_for_review_then_publishes(graph, make_deps):
    deps = make_deps()
    r = start(graph, deps, request("ACC-1001"), thread_id="t-happy")
    assert r["status"] == "awaiting_review"
    assert r["review"]["risk"]["tier"] == "high"
    assert r["state"]["brief_origin"] == "model"
    assert deps.writer.activities == {}  # nothing written before a person approves

    done = decide(graph, deps, "t-happy", approve())
    assert done["status"] == "published"
    assert done["state"]["crm_write"]["created"] is True
    assert list(deps.writer.activities) == ["ACC-1001:2026-11-24"]


def test_sources_fetch_in_parallel_and_transient_failures_retry(graph, make_deps):
    r = start(graph, make_deps(), request("ACC-1003"), thread_id="t-retry")
    tix = [t for t in r["state"]["trace"] if t["node"] == "ticketing"]
    assert [t["ok"] for t in tix] == [False, True]
    assert r["state"]["assessment"]["data_gaps"] == []


def test_source_that_never_answers_becomes_a_flagged_gap(graph, make_deps):
    r = start(graph, make_deps(), request("ACC-1005"), thread_id="t-gap")
    tix = [t for t in r["state"]["trace"] if t["node"] == "ticketing"]
    assert len(tix) == 3 and not any(t["ok"] for t in tix)
    assert "data_gap:ticketing" in r["review"]["flags"]
    cited = {fid for p in r["review"]["brief"]["talking_points"] for fid in p["evidence"]}
    assert "gap.ticketing" in cited


def test_unknown_account_ends_without_a_draft(graph, make_deps):
    llm = ScriptedLLM()
    r = start(graph, make_deps(llm), request("ACC-1999"), thread_id="t-404")
    assert r["status"] == "failed" and r["review"] is None
    assert "Brief" not in llm.calls


def test_raw_email_never_reaches_the_drafting_model_and_card_is_redacted(graph, make_deps):
    llm = ScriptedLLM()
    r = start(graph, make_deps(llm), request("ACC-1004"), thread_id="t-inject")
    email_prompts = [u for name, u in llm.prompts if name == "EmailAnalysis"]
    brief_prompts = [u for name, u in llm.prompts if name == "Brief"]
    assert email_prompts and "4111" not in email_prompts[0] and "[REDACTED CARD]" in email_prompts[0]
    assert all("SYSTEM NOTE" not in u and "castellan-billing" not in u for u in brief_prompts)
    assert r["state"]["injection"] is True and "injection_suspected" in r["review"]["flags"]


def test_bad_draft_gets_one_corrective_retry(graph, make_deps):
    llm = ScriptedLLM(brief_faults=["ungrounded_number"])
    r = start(graph, make_deps(llm), request("ACC-1002", "upsell"), thread_id="t-fix")
    assert llm.calls.count("Brief") == 2
    assert "ungrounded_number:61" in llm.prompts[-1][1]  # violations fed back to the model
    assert r["state"]["brief_origin"] == "model" and r["review"]["violations"] == []


def test_two_bad_drafts_fall_back_to_facts_only_brief(graph, make_deps):
    llm = ScriptedLLM(brief_faults=["big_discount", "invalid"])
    r = start(graph, make_deps(llm), request("ACC-1001"), thread_id="t-fallback")
    assert r["state"]["brief_origin"] == "fallback"
    assert "fallback_brief" in r["review"]["flags"]
    assert "25%" not in str(r["review"]["brief"])


def test_edit_is_rechecked_and_blocks_approval_until_fixed(graph, make_deps):
    deps = make_deps()
    r = start(graph, deps, request("ACC-1002", "upsell"), thread_id="t-edit")
    edited = Brief.model_validate(r["review"]["brief"])
    edited.situation = "They want 30% off next year."
    r = decide(graph, deps, "t-edit", ReviewDecision(decision="edit", reviewer="rep", edited_brief=edited))
    assert r["status"] == "awaiting_review"
    assert {"ungrounded_number:30", "discount_over_authority:30%"} <= set(r["review"]["violations"])

    r = decide(graph, deps, "t-edit", approve())
    assert r["status"] == "awaiting_review" and "blocked" in r["review"]["note"]
    assert deps.writer.activities == {}

    edited.situation = "Strong usage and a client event request."
    r = decide(graph, deps, "t-edit", ReviewDecision(decision="edit", reviewer="rep", edited_brief=edited))
    assert r["review"]["violations"] == [] and r["state"]["brief_origin"] == "human_edit"
    assert decide(graph, deps, "t-edit", approve())["status"] == "published"


def test_reject_writes_nothing(graph, make_deps):
    deps = make_deps()
    start(graph, deps, request("ACC-1007", "upsell"), thread_id="t-reject")
    r = decide(graph, deps, "t-reject", ReviewDecision(decision="reject", reviewer="rep", note="meeting cancelled"))
    assert r["status"] == "rejected" and deps.writer.activities == {}
    with pytest.raises(ValueError):
        decide(graph, deps, "t-reject", approve())


def test_publish_retries_transient_crm_errors_without_duplicates(graph, make_deps):
    deps = make_deps()
    deps.writer.fail_next = 2
    start(graph, deps, request("ACC-1002", "upsell"), thread_id="t-pub")
    r = decide(graph, deps, "t-pub", approve())
    assert r["status"] == "published" and deps.writer.calls == 3 and len(deps.writer.activities) == 1

    start(graph, deps, request("ACC-1002", "upsell"), thread_id="t-pub-2")
    r2 = decide(graph, deps, "t-pub-2", approve())
    assert r2["state"]["crm_write"]["created"] is False
    assert len(deps.writer.activities) == 1


def test_review_survives_a_process_restart(tmp_path, make_deps):
    db = tmp_path / "checkpoints.sqlite"
    deps = make_deps()
    g1 = build_graph(SqliteSaver(sqlite3.connect(db, check_same_thread=False)), publish_retry_interval=0)
    start(g1, deps, request("ACC-1001"), thread_id="t-durable")

    g2 = build_graph(SqliteSaver(sqlite3.connect(db, check_same_thread=False)), publish_retry_interval=0)
    assert snapshot(g2, "t-durable")["status"] == "awaiting_review"
    assert decide(g2, deps, "t-durable", approve())["status"] == "published"
