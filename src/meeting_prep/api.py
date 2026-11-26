"""Internal HTTP API: start a prep run, read it, and post the rep's decision.

Staff bearer token on every route except /healthz. Raw email bodies never leave
the service; responses carry the brief, the facts it may cite, risk and trace.
"""

from __future__ import annotations

import hmac
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException

from . import checkpoints, emf
from .graph import Deps, build_graph
from .metrics import run_metrics
from .models import MeetingRequest, ReviewDecision
from .render import render_markdown
from .service import decide, fixture_deps, snapshot, start


def public_view(result: dict[str, Any]) -> dict[str, Any]:
    state = result["state"]
    a = state.get("assessment") or {}
    return {
        "thread_id": result["thread_id"],
        "status": result["status"],
        "account_id": state["request"]["account_id"],
        "risk": a.get("risk"),
        "data_gaps": a.get("data_gaps", []),
        "brief_origin": state.get("brief_origin"),
        "brief": state.get("draft"),
        "review": result["review"],
        "crm_write": state.get("crm_write"),
        "facts": a.get("facts", []),
        "markdown": render_markdown(state, result["review"]),
    }


def create_app(graph: Any = None, deps: Deps | None = None, token: str | None = None) -> FastAPI:
    token = token or os.environ.get("STAFF_TOKEN")
    if not token:
        raise RuntimeError("Set STAFF_TOKEN")
    if graph is None:
        graph = build_graph(checkpoints.from_env())
    deps = deps or fixture_deps()
    app = FastAPI(title="Meeting prep agent")

    def staff(authorization: str = Header(default="")) -> None:
        supplied = authorization.removeprefix("Bearer ").strip()
        if not supplied or not hmac.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="missing or invalid staff token")

    def load(thread_id: str) -> dict[str, Any]:
        try:
            return snapshot(graph, thread_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown thread") from None

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "model": deps.llm.name}

    @app.post("/briefs", status_code=201, dependencies=[Depends(staff)])
    def create_brief(req: MeetingRequest) -> dict[str, Any]:
        started = time.perf_counter()
        result = start(graph, deps, req)
        state = result["state"]
        gaps = len((state.get("assessment") or {}).get("data_gaps", []))
        emf.emit(emf.run_values(run_metrics(state.get("trace", [])), gaps, (time.perf_counter() - started) * 1000), units=emf.RUN_UNITS)
        if result["status"] == "failed":
            errors = result["state"].get("source_errors", [])
            if any(e["source"] == "crm" and e["error"] == "record not found" for e in errors):
                raise HTTPException(status_code=404, detail=f"account {req.account_id} not found")
            raise HTTPException(status_code=503, detail="CRM unavailable", headers={"Retry-After": "30"})
        return public_view(result)

    @app.get("/briefs/{thread_id}", dependencies=[Depends(staff)])
    def get_brief(thread_id: str) -> dict[str, Any]:
        return public_view(load(thread_id))

    @app.get("/briefs/{thread_id}/trace", dependencies=[Depends(staff)])
    def get_trace(thread_id: str) -> dict[str, Any]:
        trace = load(thread_id)["state"].get("trace", [])
        return {"trace": trace, "metrics": run_metrics(trace)}

    @app.post("/briefs/{thread_id}/decision", dependencies=[Depends(staff)])
    def post_decision(thread_id: str, decision: ReviewDecision) -> dict[str, Any]:
        load(thread_id)
        try:
            result = decide(graph, deps, thread_id, decision)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        emf.emit({f"Decision{decision.decision.capitalize()}": 1})
        return public_view(result)

    return app
