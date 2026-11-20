"""Command line: prep a meeting, then approve, edit or reject it in a later call.
Runs are checkpointed to SQLite, so the decision can come hours after the draft.

    meeting-prep prep ACC-1001 --date 2026-11-24 --purpose renewal --rep "J. Ortiz"
    meeting-prep decide <thread_id> approve --reviewer "J. Ortiz"
    meeting-prep show <thread_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .graph import build_graph
from .models import MeetingRequest, ReviewDecision
from .render import render_markdown
from .service import decide, fixture_deps, snapshot, start


def _graph():
    path = Path(os.environ.get("CHECKPOINT_DB", ".data/checkpoints.sqlite"))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


def _print(result: dict) -> None:
    print(render_markdown(result["state"], result["review"]))
    print(f"thread: {result['thread_id']}  status: {result['status']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="meeting-prep")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prep")
    a.add_argument("account_id")
    a.add_argument("--date", required=True, type=date.fromisoformat)
    a.add_argument("--purpose", default="renewal", choices=["renewal", "upsell", "service_recovery", "check_in"])
    a.add_argument("--rep", default="demo rep")

    d = sub.add_parser("decide")
    d.add_argument("thread_id")
    d.add_argument("decision", choices=["approve", "reject"])
    d.add_argument("--reviewer", required=True)

    s = sub.add_parser("show")
    s.add_argument("thread_id")
    s.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    graph, deps = _graph(), fixture_deps()

    if args.cmd == "prep":
        req = MeetingRequest(account_id=args.account_id, meeting_date=args.date, purpose=args.purpose, rep=args.rep)
        _print(start(graph, deps, req))
    elif args.cmd == "decide":
        _print(decide(graph, deps, args.thread_id, ReviewDecision(decision=args.decision, reviewer=args.reviewer)))
    else:
        result = snapshot(graph, args.thread_id)
        if args.json:
            json.dump(result, sys.stdout, indent=2, default=str)
            print()
        else:
            _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
