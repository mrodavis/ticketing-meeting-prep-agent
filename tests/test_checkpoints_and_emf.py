"""Postgres checkpointer and CloudWatch EMF output.

The Postgres tests run when PGHOST is set (CI starts a postgres service container);
otherwise they're skipped.
"""

import io
import json
import os
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from meeting_prep import checkpoints, emf
from meeting_prep.api import create_app
from meeting_prep.graph import build_graph

AUTH = {"Authorization": "Bearer test-token"}
needs_postgres = pytest.mark.skipif(not os.environ.get("PGHOST"), reason="set PGHOST (and PGUSER, PGDATABASE...) to run")


@needs_postgres
def test_a_different_task_can_resume_the_run(make_deps):
    """Task A drafts; task B, with its own connection pool, takes the approval."""
    deps = make_deps()
    saver_a, saver_b = checkpoints.postgres_saver(), checkpoints.postgres_saver()
    task_a = TestClient(create_app(graph=build_graph(saver_a), deps=deps, token="test-token"))
    task_b = TestClient(create_app(graph=build_graph(saver_b), deps=deps, token="test-token"))
    account = "ACC-1002"
    r = task_a.post(
        "/briefs",
        json={"account_id": account, "meeting_date": "2026-11-24", "purpose": "upsell", "rep": f"ci-{uuid.uuid4().hex[:6]}"},
        headers=AUTH,
    )
    assert r.status_code == 201 and r.json()["status"] == "awaiting_review"
    tid = r.json()["thread_id"]
    assert task_b.get(f"/briefs/{tid}", headers=AUTH).json()["status"] == "awaiting_review"
    done = task_b.post(f"/briefs/{tid}/decision", json={"decision": "approve", "reviewer": "ci"}, headers=AUTH)
    assert done.status_code == 200 and done.json()["status"] == "published"
    saver_a.conn.close()
    saver_b.conn.close()


@needs_postgres
def test_concurrent_startup_runs_migrations_once():
    """Four tasks boot at once against an empty schema, as on a first deploy."""
    import psycopg

    schema = f"cp_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect("", autocommit=True) as admin:
        admin.execute(f"CREATE SCHEMA {schema}")
    errors, savers = [], []

    def boot():
        try:
            savers.append(checkpoints.postgres_saver(conninfo=f"options=-csearch_path={schema}", max_size=2))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=boot) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for s in savers:
        s.conn.close()
    with psycopg.connect("", autocommit=True) as admin:
        tables = admin.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema = %s", (schema,)).fetchone()[0]
        admin.execute(f"DROP SCHEMA {schema} CASCADE")
    assert not errors and len(savers) == 4
    assert tables == 4


def test_backend_selection(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "sqlite")
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cp.sqlite"))
    assert type(checkpoints.from_env()).__name__ == "SqliteSaver"
    monkeypatch.setenv("CHECKPOINT_BACKEND", "dynamo")
    with pytest.raises(RuntimeError, match="unknown CHECKPOINT_BACKEND"):
        checkpoints.from_env()


def test_emf_is_off_unless_enabled(monkeypatch):
    monkeypatch.delenv("EMF_ENABLED", raising=False)
    assert emf.emit({"BriefRuns": 1}) is None


def test_emf_document_shape(monkeypatch):
    monkeypatch.setenv("EMF_ENABLED", "1")
    out = io.StringIO()
    metrics = {
        "fallback": False,
        "first_draft_passed": True,
        "tool_attempts": 6,
        "tool_calls": 4,
        "llm_attempts": 3,
        "llm_attempts_ok": 2,
        "input_tokens": 1200,
        "output_tokens": 300,
    }
    emf.emit(emf.run_values(metrics, data_gaps=1, latency_ms=4210.6), stream=out, units=emf.RUN_UNITS)
    doc = json.loads(out.getvalue())
    directive = doc["_aws"]["CloudWatchMetrics"][0]
    assert directive["Namespace"] == "MeetingPrep" and directive["Dimensions"] == [["Service"]]
    assert {m["Name"]: m["Unit"] for m in directive["Metrics"]}["RunLatency"] == "Milliseconds"
    assert (doc["ToolRetries"], doc["LlmCallFailures"], doc["RunLatency"], doc["Service"]) == (2, 1, 4211, "meeting-prep")


def test_api_emits_one_line_per_run(monkeypatch, capsys, graph, make_deps):
    monkeypatch.setenv("EMF_ENABLED", "1")
    client = TestClient(create_app(graph=graph, deps=make_deps(), token="test-token"))
    r = client.post(
        "/briefs", json={"account_id": "ACC-1001", "meeting_date": "2026-11-24", "purpose": "renewal", "rep": "J. Ortiz"}, headers=AUTH
    )
    client.post(f"/briefs/{r.json()['thread_id']}/decision", json={"decision": "approve", "reviewer": "J. Ortiz"}, headers=AUTH)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{"_aws"')]
    assert [("BriefRuns" in d, "DecisionApprove" in d) for d in lines] == [(True, False), (False, True)]
