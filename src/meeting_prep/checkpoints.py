"""Where paused runs are kept between the draft and the rep's decision.

CHECKPOINT_BACKEND=sqlite (default): a local file, fine for one process.
CHECKPOINT_BACKEND=postgres: shared by every task behind the load balancer, so a rep
can approve on a different task than the one that drafted. The connection comes from
the standard libpq variables (PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
PGSSLMODE), which ECS fills from Secrets Manager, or from CHECKPOINT_URL if set.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# Any constant works; it only has to be the same in every task.
SETUP_LOCK_ID = 7_310_442


def postgres_saver(conninfo: str | None = None, max_size: int | None = None) -> Any:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=conninfo if conninfo is not None else os.environ.get("CHECKPOINT_URL", ""),
        min_size=1,
        max_size=max_size or int(os.environ.get("CHECKPOINT_POOL_MAX", "5")),
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool)
    try:
        with pool.connection() as conn:
            _locked(conn, saver.setup)
    except Exception:
        pool.close()
        raise
    return saver


def _locked(conn: Any, fn: Any, timeout_s: float = 120.0, poll_s: float = 0.25) -> None:
    """Run fn while holding a Postgres advisory lock, so tasks that start together run
    the schema migrations one at a time. Without it they collide (DuplicateTable,
    UniqueViolation).

    The lock is polled with pg_try_advisory_lock instead of waiting in
    pg_advisory_lock. The migrations include CREATE INDEX CONCURRENTLY, which waits
    for every transaction holding an older snapshot to finish, and a session blocked
    inside pg_advisory_lock holds one. The holder would wait on the waiters while
    they wait on the holder, and the setup connection and the lock connection are
    different sessions, so Postgres can't detect the deadlock.
    """
    deadline = time.monotonic() + timeout_s
    while not conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (SETUP_LOCK_ID,)).fetchone()["ok"]:
        if time.monotonic() > deadline:
            raise TimeoutError("another task held the checkpoint migration lock too long")
        time.sleep(poll_s)
    try:
        fn()
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (SETUP_LOCK_ID,))


def sqlite_saver(path: str | None = None) -> Any:
    from langgraph.checkpoint.sqlite import SqliteSaver

    p = Path(path or os.environ.get("CHECKPOINT_DB", ".data/checkpoints.sqlite"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(p, check_same_thread=False))


def from_env() -> Any:
    backend = os.environ.get("CHECKPOINT_BACKEND", "sqlite")
    if backend == "postgres":
        return postgres_saver()
    if backend == "sqlite":
        return sqlite_saver()
    raise RuntimeError(f"unknown CHECKPOINT_BACKEND {backend!r} (sqlite or postgres)")
