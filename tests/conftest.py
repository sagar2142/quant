"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# 127.0.0.1, never "localhost". On Windows, localhost resolves to ::1 first and
# Docker publishes the port on IPv4 only, so "localhost" hangs until timeout.
DEFAULT_DB_URL = "postgresql://neutron:neutron@127.0.0.1:5433/neutron"

CONNECT_TIMEOUT_S = 5


def _db_url() -> str:
    url = os.environ.get("NEUTRON_DB_URL", DEFAULT_DB_URL)
    # psycopg speaks plain libpq URLs; strip any SQLAlchemy driver suffix.
    return url.replace("postgresql+psycopg://", "postgresql://")


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(_db_url(), connect_timeout=CONNECT_TIMEOUT_S):
            return True
    except Exception:  # noqa: BLE001 — any connection failure means "skip"
        return False


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
