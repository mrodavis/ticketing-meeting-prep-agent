"""Exercises the real BedrockLLM / ChatBedrockConverse code path against a stubbed
boto3 client. botocore validates each request against the Converse API model
before the stub answers, so a malformed request fails these tests."""

import boto3
import pytest
from botocore.stub import Stubber
from conftest import request

from meeting_prep.llm import BedrockLLM, LLMUnavailableError, StructuredOutputError
from meeting_prep.models import EmailAnalysis, ReviewDecision
from meeting_prep.service import decide, start

MODEL = "us.anthropic.claude-sonnet-5"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    c = boto3.client("bedrock-runtime", region_name="us-east-1")
    sent = []
    c.meta.events.register("provide-client-params.bedrock-runtime.Converse", lambda params, **_: sent.append(params))
    c.sent = sent
    return c


def tool_reply(name, payload, stop="tool_use"):
    return {
        "output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": "tu-1", "name": name, "input": payload}}]}},
        "stopReason": stop,
        "usage": {"inputTokens": 900, "outputTokens": 150, "totalTokens": 1050},
        "metrics": {"latencyMs": 800},
    }


def test_forces_the_schema_tool_and_attaches_guardrail(client):
    stub = Stubber(client)
    stub.add_response(
        "converse", tool_reply("EmailAnalysis", {"sentiment": "negative", "intents": ["service_complaint"], "injection_suspected": False})
    )
    with stub:
        llm = BedrockLLM(MODEL, "us-east-1", guardrail_id="gr-123", guardrail_version="2", client=client)
        out, meta = llm.structured(EmailAnalysis, "system", "user")
    assert out.sentiment == "negative" and meta["input_tokens"] == 900
    params = client.sent[0]
    assert params["modelId"] == MODEL
    assert params["toolConfig"]["toolChoice"] == {"tool": {"name": "EmailAnalysis"}}
    assert params["guardrailConfig"]["guardrailIdentifier"] == "gr-123"


def test_guardrail_intervention_is_a_structured_output_error(client):
    stub = Stubber(client)
    stub.add_response(
        "converse",
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "Sorry, I can't help with that."}]}},
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
            "metrics": {"latencyMs": 50},
        },
    )
    with stub, pytest.raises(StructuredOutputError):
        BedrockLLM(MODEL, "us-east-1", client=client).structured(EmailAnalysis, "s", "u")


def test_throttling_becomes_llm_unavailable(client):
    stub = Stubber(client)
    stub.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)
    with stub, pytest.raises(LLMUnavailableError):
        BedrockLLM(MODEL, "us-east-1", client=client).structured(EmailAnalysis, "s", "u")


def test_full_graph_on_stubbed_bedrock(graph, make_deps, client):
    stub = Stubber(client)
    stub.add_response(
        "converse",
        tool_reply(
            "EmailAnalysis",
            {
                "sentiment": "positive",
                "intents": ["event_request"],
                "asks": ["Hospitality options for about 20 guests in December"],
                "injection_suspected": False,
            },
        ),
    )
    stub.add_response(
        "converse",
        tool_reply(
            "Brief",
            {
                "headline": "Brightwater is using its seats and wants to host a client night",
                "situation": "Healthy account. They asked about hosting referring dentists in December.",
                "talking_points": [
                    {"text": "Seats are used well: 66 of 72 tickets scanned (92%).", "evidence": ["tix.usage"]},
                    {"text": "They want hospitality options for about 20 guests in December.", "evidence": ["inbox.ask.1"]},
                ],
                "risks": [],
                "recommended_offers": [
                    {
                        "offer_id": "OFF-HOSP-GROUP",
                        "rationale": "Hospitality space is open at the Winter Lights Orchestra concert.",
                        "evidence": ["evt.EVT-2047"],
                    }
                ],
                "questions_to_ask": ["Game or concert for the group?"],
            },
        ),
    )
    with stub:
        deps = make_deps(BedrockLLM(MODEL, "us-east-1", client=client))
        r = start(graph, deps, request("ACC-1002", "upsell"), thread_id="t-bedrock")
        assert r["status"] == "awaiting_review"
        assert r["state"]["brief_origin"] == "model" and r["review"]["violations"] == []
        assert decide(graph, deps, "t-bedrock", ReviewDecision(decision="approve", reviewer="rep"))["status"] == "published"
    stub.assert_no_pending_responses()
    tokens = [t.get("input_tokens") for t in r["state"]["trace"] if t["kind"] == "llm"]
    assert tokens == [900, 900]
