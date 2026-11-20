"""Data sources the graph reads from, and the CRM it writes the approved brief to.

Each source is a small Protocol so the fixture implementations here can be
swapped for real clients (Salesforce REST/SOQL for CRM, the ticketing vendor's
API, the events calendar, a mailbox) without touching the graph.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date
from importlib import resources
from typing import Any, Protocol


class SourceError(Exception):
    """Base class for source failures."""


class TransientSourceError(SourceError):
    """Timeouts, throttling, 5xx. Safe to retry."""


class NotFoundError(SourceError):
    """The record does not exist. Retrying will not help."""


class CrmSource(Protocol):
    def get_account(self, account_id: str) -> dict[str, Any]: ...


class TicketingSource(Protocol):
    def get_activity(self, account_id: str) -> dict[str, Any]: ...


class EventsSource(Protocol):
    def upcoming(self, start: date, end: date) -> list[dict[str, Any]]: ...


class InboxSource(Protocol):
    def recent(self, account_id: str) -> list[dict[str, Any]]: ...


class CrmWriter(Protocol):
    def upsert_activity(self, idempotency_key: str, account_id: str, body: dict[str, Any]) -> dict[str, Any]: ...


def _load(name: str) -> dict[str, Any]:
    return json.loads(resources.files("meeting_prep").joinpath("data", name).read_text())


class FixtureCrm:
    def __init__(self, data: dict[str, Any] | None = None):
        self._accounts = (data or _load("crm.json"))["accounts"]

    def get_account(self, account_id: str) -> dict[str, Any]:
        if account_id not in self._accounts:
            raise NotFoundError(f"account {account_id} not found")
        return {"id": account_id, **self._accounts[account_id]}

    def exists(self, account_id: str) -> bool:
        return account_id in self._accounts


class FixtureTicketing:
    """Ticket activity with simulated outages so retries and degradation get exercised."""

    def __init__(self, data: dict[str, Any] | None = None, outages: dict[str, int] | None = None):
        data = data or _load("ticketing.json")
        self._accounts = data["accounts"]
        raw = data.get("outages", {}) if outages is None else outages
        self._failures_left = {k: v for k, v in raw.items() if not k.startswith("_")}
        self._lock = threading.Lock()

    def get_activity(self, account_id: str) -> dict[str, Any]:
        with self._lock:
            left = self._failures_left.get(account_id, 0)
            if left > 0:
                self._failures_left[account_id] = left - 1
                raise TransientSourceError("ticketing API timed out")
        if account_id not in self._accounts:
            raise NotFoundError(f"no ticket activity for {account_id}")
        return dict(self._accounts[account_id])


class FixtureEvents:
    def __init__(self, data: dict[str, Any] | None = None):
        self._events = (data or _load("events.json"))["events"]

    def upcoming(self, start: date, end: date) -> list[dict[str, Any]]:
        return [e for e in self._events if start <= date.fromisoformat(e["date"]) <= end]


class FixtureInbox:
    def __init__(self, data: dict[str, Any] | None = None):
        self._accounts = (data or _load("inbox.json"))["accounts"]

    def recent(self, account_id: str) -> list[dict[str, Any]]:
        return [dict(m) for m in self._accounts.get(account_id, [])]


def load_offers() -> dict[str, Any]:
    return _load("offers.json")


@dataclass
class InMemoryCrmWriter:
    """Stands in for creating a Salesforce Task/Note. Idempotent on the key."""

    activities: dict[str, dict[str, Any]] = field(default_factory=dict)
    fail_next: int = 0
    calls: int = 0

    def upsert_activity(self, idempotency_key: str, account_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        if self.fail_next > 0:
            self.fail_next -= 1
            raise TransientSourceError("CRM write timed out")
        if idempotency_key in self.activities:
            return {**self.activities[idempotency_key], "created": False}
        record = {"activity_id": f"ACT-{len(self.activities) + 1:04d}", "account_id": account_id, "body": body}
        self.activities[idempotency_key] = record
        return {**record, "created": True}
