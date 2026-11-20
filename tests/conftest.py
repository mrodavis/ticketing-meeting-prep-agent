from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from meeting_prep.graph import build_graph
from meeting_prep.llm import ScriptedLLM
from meeting_prep.models import MeetingRequest
from meeting_prep.service import fixture_deps

TODAY = date(2026, 11, 20)
MEETING = date(2026, 11, 24)


@pytest.fixture
def graph():
    return build_graph(InMemorySaver(), publish_retry_interval=0)


@pytest.fixture
def make_deps():
    def _make(llm=None, **overrides):
        return fixture_deps(llm or ScriptedLLM(), today=TODAY, backoff_s=0, **overrides)

    return _make


def request(account_id: str, purpose: str = "renewal") -> MeetingRequest:
    return MeetingRequest(account_id=account_id, meeting_date=MEETING, purpose=purpose, rep="test rep")
