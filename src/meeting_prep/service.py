"""Thin service layer shared by the API, the CLI and the evals: start a prep run
(which stops at the human review interrupt) and resume it with a decision."""

from __future__ import annotations

import os
import uuid
from datetime import date
from typing import Any

from langgraph.types import Command

from .graph import Deps
from .llm import BedrockLLM, ScriptedLLM, StructuredLLM
from .models import MeetingRequest, ReviewDecision
from .sources import FixtureCrm, FixtureEvents, FixtureInbox, FixtureTicketing, InMemoryCrmWriter, load_offers


def llm_from_env() -> StructuredLLM:
    mode = os.environ.get("LLM_MODE", "offline")
    if mode == "bedrock":
        return BedrockLLM.from_env()
    if mode == "offline":
        return ScriptedLLM()
    raise RuntimeError(f"unknown LLM_MODE {mode!r} (use offline or bedrock)")


def fixture_deps(llm: StructuredLLM | None = None, today: date | None = None, **overrides: Any) -> Deps:
    as_of = today or date.fromisoformat(os.environ.get("AS_OF_DATE", "2026-11-20"))
    deps = Deps(
        crm=FixtureCrm(),
        ticketing=FixtureTicketing(),
        events=FixtureEvents(),
        inbox=FixtureInbox(),
        writer=InMemoryCrmWriter(),
        llm=llm or llm_from_env(),
        offers=load_offers(),
        today=lambda: as_of,
    )
    for k, v in overrides.items():
        setattr(deps, k, v)
    return deps


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def snapshot(graph: Any, thread_id: str) -> dict[str, Any]:
    snap = graph.get_state(_config(thread_id))
    if not snap.values:
        raise KeyError(thread_id)
    values = dict(snap.values)
    pending = [i.value for i in snap.interrupts] if snap.interrupts else []
    return {
        "thread_id": thread_id,
        "status": "awaiting_review" if pending else values.get("status"),
        "review": pending[0] if pending else None,
        "state": values,
    }


def start(graph: Any, deps: Deps, request: MeetingRequest, thread_id: str | None = None) -> dict[str, Any]:
    thread_id = thread_id or f"prep-{uuid.uuid4().hex[:12]}"
    graph.invoke({"request": request.model_dump(mode="json")}, _config(thread_id), context=deps)
    return snapshot(graph, thread_id)


def decide(graph: Any, deps: Deps, thread_id: str, decision: ReviewDecision) -> dict[str, Any]:
    current = snapshot(graph, thread_id)
    if current["status"] != "awaiting_review":
        raise ValueError(f"thread {thread_id} is {current['status']}, not awaiting review")
    graph.invoke(Command(resume=decision.model_dump(mode="json")), _config(thread_id), context=deps)
    return snapshot(graph, thread_id)
