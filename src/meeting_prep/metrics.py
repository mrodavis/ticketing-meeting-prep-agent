"""Per-run and aggregate metrics from the trace: tool-call success, model-call
success, guardrail pass rate on first draft, fallbacks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def run_metrics(trace: list[dict[str, Any]]) -> dict[str, Any]:
    tool = [t for t in trace if t["kind"] == "tool"]
    llm = [t for t in trace if t["kind"] == "llm"]
    guard = [t for t in trace if t["kind"] == "guard"]
    by_node: dict[str, list[dict[str, Any]]] = {}
    for t in tool:
        by_node.setdefault(t["node"], []).append(t)
    return {
        "tool_attempts": len(tool),
        "tool_attempts_ok": sum(t["ok"] for t in tool),
        "tool_calls": len(by_node),
        "tool_calls_first_try": sum(1 for calls in by_node.values() if calls[0]["ok"]),
        "tool_calls_eventual": sum(1 for calls in by_node.values() if any(c["ok"] for c in calls)),
        "llm_attempts": len(llm),
        "llm_attempts_ok": sum(t["ok"] for t in llm),
        "first_draft_passed": bool(guard) and guard[0]["ok"],
        "drafts": sum(1 for t in llm if t["node"] == "draft_brief"),
        "fallback": any(t["node"] == "fallback" for t in trace),
        "input_tokens": sum(t.get("input_tokens") or 0 for t in llm),
        "output_tokens": sum(t.get("output_tokens") or 0 for t in llm),
    }


def aggregate(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs = list(runs)
    total = lambda k: sum(r[k] for r in runs)  # noqa: E731
    calls = total("tool_calls") or 1
    llm = total("llm_attempts") or 1
    return {
        "runs": len(runs),
        "tool_success_first_try": round(total("tool_calls_first_try") / calls, 3),
        "tool_success_eventual": round(total("tool_calls_eventual") / calls, 3),
        "llm_call_success": round(total("llm_attempts_ok") / llm, 3),
        "first_draft_pass_rate": round(sum(r["first_draft_passed"] for r in runs) / max(len(runs), 1), 3),
        "fallback_rate": round(sum(r["fallback"] for r in runs) / max(len(runs), 1), 3),
    }
