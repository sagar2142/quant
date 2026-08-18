"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

# 127.0.0.1, never "localhost". On Windows, localhost resolves to ::1 first and
# Docker publishes the port on IPv4 only, so "localhost" hangs until timeout.
DEFAULT_DB_URL = "postgresql://neutron:neutron@127.0.0.1:5433/neutron"

CONNECT_TIMEOUT_S = 5

MIGRATIONS = Path(__file__).resolve().parent.parent / "db" / "migrations"

#: Presence of this table means the schema is already applied. Checking one
#: sentinel is enough: migrations run as a set, so a partial schema is a
#: broken database rather than a state to be repaired here.
SENTINEL_TABLE = "hypotheses"


def _db_url() -> str:
    url = os.environ.get("NEUTRON_DB_URL", DEFAULT_DB_URL)
    # psycopg speaks plain libpq URLs; strip any SQLAlchemy driver suffix.
    return url.replace("postgresql+psycopg://", "postgresql://")


def _schema_exists(conn: object) -> bool:
    with conn.cursor() as cur:  # type: ignore[attr-defined]
        cur.execute("SELECT to_regclass(%s)", [SENTINEL_TABLE])
        return cur.fetchone()[0] is not None


def _apply_migrations(conn: object) -> None:
    """Run db/migrations/*.sql in filename order.

    Applied by the tests themselves rather than by a CI step, so a fresh clone
    and a fresh CI runner behave identically — the schema is part of what the
    integration tests are testing, and a suite that silently skips because
    nobody ran a setup command is worse than one that fails.
    """
    for path in sorted(MIGRATIONS.glob("*.sql")):
        with conn.cursor() as cur:  # type: ignore[attr-defined]
            cur.execute(path.read_text(encoding="utf-8"))
    conn.commit()  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        import psycopg
    except ImportError:
        return False
    try:
        conn = psycopg.connect(_db_url(), connect_timeout=CONNECT_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — any connection failure means "skip"
        return False
    try:
        if not _schema_exists(conn):
            _apply_migrations(conn)
        return True
    finally:
        conn.close()


@pytest.fixture
def db(db_available: bool) -> Iterator[object]:
    """A connection that rolls back everything, so tests never leave residue."""
    if not db_available:
        pytest.skip("postgres not reachable — run: docker compose up -d postgres")
    import psycopg

    conn = psycopg.connect(_db_url(), connect_timeout=CONNECT_TIMEOUT_S)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()
