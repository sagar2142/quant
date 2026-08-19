"""Database connections — MASTER_PLAN §13.7.

One place that knows how to reach Postgres, so no module builds a connection
string of its own and none of them disagree about the port.

**A missing database is not a crash.** Research runs on a laptop with the
container stopped are normal, and a validation run that dies because Postgres
is down teaches nothing. `optional_connection` returns None and says why; the
caller decides whether that is fatal. The one thing it must never do is fail
silently, because a run that quietly skipped recording is a trial that never
got counted (§5.2).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from core.config import settings

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["connection_url", "optional_connection"]

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_SECONDS = 5


def connection_url() -> str:
    """libpq URL. psycopg does not want SQLAlchemy's driver suffix."""
    return settings.db_url.replace("postgresql+psycopg://", "postgresql://")


@contextmanager
def optional_connection() -> Iterator[Any | None]:
    """Yield a connection, or None when Postgres cannot be reached.

    Yields:
        An open psycopg connection, closed on exit, or None. The reason is
        printed rather than logged quietly: a caller that silently skipped
        recording would leave the trial counter understated, which is the exact
        failure the counter exists to prevent.
    """
    try:
        import psycopg  # noqa: PLC0415 - optional at runtime, not at import
    except ImportError:
        print("  psycopg is not installed; this run will not be recorded")
        yield None
        return

    try:
        conn = psycopg.connect(connection_url(), connect_timeout=CONNECT_TIMEOUT_SECONDS)
    except psycopg.Error as exc:
        print(f"  Postgres unreachable ({exc.__class__.__name__}); this run will not be recorded")
        print("  start it with: docker compose up -d postgres")
        yield None
        return

    try:
        yield conn
    finally:
        conn.close()
