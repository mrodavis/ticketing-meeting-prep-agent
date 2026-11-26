"""CloudWatch Embedded Metric Format. With EMF_ENABLED=1 each run prints one JSON line
to stdout; the awslogs driver ships it to CloudWatch Logs, which turns it into metrics
with no agent and no PutMetricData calls. The only dimension is the service name, so
the metric count stays fixed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, TextIO

NAMESPACE = "MeetingPrep"
SERVICE = "meeting-prep"


def emit(values: dict[str, float], stream: TextIO | None = None, units: dict[str, str] | None = None) -> dict[str, Any] | None:
    if os.environ.get("EMF_ENABLED") != "1" or not values:
        return None
    units = units or {}
    doc: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["Service"]],
                    "Metrics": [{"Name": k, "Unit": units.get(k, "Count")} for k in values],
                }
            ],
        },
        "Service": SERVICE,
        **values,
    }
    print(json.dumps(doc), file=stream or sys.stdout, flush=True)
    return doc


def run_values(metrics: dict[str, Any], data_gaps: int, latency_ms: float) -> dict[str, float]:
    """Maps metrics.run_metrics() output to the numbers the alarms and dashboard use."""
    return {
        "BriefRuns": 1,
        "Fallback": int(bool(metrics["fallback"])),
        "FirstDraftPassed": int(bool(metrics["first_draft_passed"])),
        "ToolRetries": metrics["tool_attempts"] - metrics["tool_calls"],
        "LlmCallFailures": metrics["llm_attempts"] - metrics["llm_attempts_ok"],
        "DataGaps": data_gaps,
        "InputTokens": metrics["input_tokens"],
        "OutputTokens": metrics["output_tokens"],
        "RunLatency": round(latency_ms),
    }


RUN_UNITS = {"RunLatency": "Milliseconds"}
