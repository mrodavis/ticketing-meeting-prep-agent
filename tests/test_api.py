import pytest
from fastapi.testclient import TestClient

from meeting_prep.api import create_app

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(graph, make_deps):
    return TestClient(create_app(graph=graph, deps=make_deps(), token="test-token"))


def body(account_id="ACC-1001", purpose="renewal"):
    return {"account_id": account_id, "meeting_date": "2026-11-24", "purpose": purpose, "rep": "J. Ortiz"}


def test_requires_staff_token(client):
    assert client.get("/healthz").status_code == 200
    assert client.post("/briefs", json=body()).status_code == 401
    assert client.post("/briefs", json=body(), headers={"Authorization": "Bearer nope"}).status_code == 401


def test_create_review_and_approve(client):
    r = client.post("/briefs", json=body(), headers=AUTH)
    assert r.status_code == 201
    data = r.json()
    assert data["status"] == "awaiting_review" and data["risk"]["tier"] == "high"
    assert "## " in data["markdown"]
    tid = data["thread_id"]

    assert client.get(f"/briefs/{tid}", headers=AUTH).json()["status"] == "awaiting_review"
    trace = client.get(f"/briefs/{tid}/trace", headers=AUTH).json()
    assert trace["metrics"]["tool_calls"] == 4

    done = client.post(f"/briefs/{tid}/decision", json={"decision": "approve", "reviewer": "J. Ortiz"}, headers=AUTH)
    assert done.status_code == 200 and done.json()["status"] == "published"
    again = client.post(f"/briefs/{tid}/decision", json={"decision": "approve", "reviewer": "J. Ortiz"}, headers=AUTH)
    assert again.status_code == 409


def test_unknown_account_and_thread(client):
    assert client.post("/briefs", json=body("ACC-1999"), headers=AUTH).status_code == 404
    assert client.get("/briefs/nope", headers=AUTH).status_code == 404
    assert client.post("/briefs", json={**body(), "account_id": "bad"}, headers=AUTH).status_code == 422


def test_raw_email_is_not_returned(client):
    data = client.post("/briefs", json=body("ACC-1004"), headers=AUTH).json()
    text = str(data)
    assert "SYSTEM NOTE" not in text and "4111" not in text and "castellan-billing" not in text
