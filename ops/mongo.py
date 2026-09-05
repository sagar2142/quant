"""MongoDB connection — MASTER_PLAN §13.7.

**Accounts live here; the research ledger does not.** Hypotheses, experiments
and the trial counter stay in Postgres because what makes them trustworthy is
exactly what Mongo does not offer: a CHECK constraint on the economic
mechanism, a UNIQUE on the statement, a trigger that increments the trial count
on every experiment insert. Those are not conveniences — they are the reason a
verdict cannot be quietly un-made (§5.1). Moving them would turn enforced rules
into remembered ones.

Accounts are a different shape: a handful of documents, no cross-row
invariants, read once per request. Mongo is a reasonable home for them, and
using it there costs the research protocol nothing.

**The URI is never in the repository.** It carries a password, so it is read
from `MONGODB_URI` in the environment and `.env` is gitignored. Nothing in this
module logs it, and the connection is described by host and database only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from core.envfile import load_env_file

__all__ = [
    "DATABASE_NAME",
    "database_name",
    "describe_uri",
    "mongo_available",
    "mongo_client",
    "mongo_database",
]


load_env_file()


def database_name() -> str:
    """The database this system uses. Read late, so `.env` has been applied."""
    return os.environ.get("MONGODB_DATABASE", "neutron").strip() or "neutron"


#: The database this system creates and uses.
DATABASE_NAME = database_name()

#: How long to wait before deciding Atlas is unreachable. Short: a console that
#: hangs for thirty seconds on a dead cluster is worse than one that says so.
CONNECT_TIMEOUT_MS = 5_000
SERVER_SELECTION_TIMEOUT_MS = 5_000


def mongo_uri() -> str:
    """The configured connection string, or an empty string."""
    load_env_file()
    return os.environ.get("MONGODB_URI", "").strip()


def mongo_available() -> bool:
    """Whether a URI is configured at all. Does not open a connection."""
    return bool(mongo_uri())


def describe_uri(uri: str | None = None) -> str:
    """The connection, with the credentials removed.

    For logs and status screens. A URI printed whole puts the cluster password
    into whatever read it — a terminal scrollback, a screenshot, a support
    thread — and that is the mundane path by which credentials actually leak.
    """
    raw = uri if uri is not None else mongo_uri()
    if not raw:
        return "not configured"
    try:
        parts = urlsplit(raw)
        host = parts.hostname or "unknown"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "unparseable"


@contextmanager
def mongo_client() -> Iterator[object | None]:
    """A client, or None when Mongo is not configured or not reachable.

    Yields None rather than raising, for the same reason `optional_connection`
    does: a monitoring surface that 500s because a database is down tells the
    operator less than one that renders and says the database is down.
    """
    uri = mongo_uri()
    if not uri:
        yield None
        return

    try:
        from pymongo import MongoClient  # noqa: PLC0415 - optional dependency
    except ImportError:
        yield None
        return

    # Construction and use are separated deliberately. Yielding from inside an
    # `except` makes the generator yield twice when an exception propagates in
    # at the yield point, and `contextlib` answers that with
    # "generator didn't stop after throw()" — an error about this function
    # rather than about the database, which is the worst kind to debug.
    client: Any
    try:
        client = MongoClient(
            uri,
            connectTimeoutMS=CONNECT_TIMEOUT_MS,
            serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
            appname="neutron",
        )
    except Exception:  # noqa: BLE001 - any failure means "not available"
        yield None
        return

    try:
        # Verified before it is handed out. Constructing a client does not
        # contact the server, so a wrong password or a blocked IP would
        # otherwise surface as a traceback from whichever query ran first —
        # far from the configuration that actually caused it.
        client.admin.command("ping")
    except Exception:  # noqa: BLE001 - unreachable, unauthorised, all the same here
        client.close()
        yield None
        return

    try:
        yield client
    finally:
        client.close()


@contextmanager
def mongo_database() -> Iterator[object | None]:
    """The `neutron` database, or None when Mongo is unavailable."""
    with mongo_client() as client:
        if client is None:
            yield None
            return
        yield client[database_name()]  # type: ignore[index]
