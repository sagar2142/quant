"""Where accounts are kept — MASTER_PLAN §13.7.

Two backends behind one interface, chosen by configuration: MongoDB when
`MONGODB_URI` is set, Postgres otherwise. Both store the same thing — an scrypt
digest with its parameters, and sessions keyed by a hash of the token — because
the security properties belong to `apps.api.accounts`, not to the database.

**The research ledger is not moved.** Hypotheses and experiments stay in
Postgres, where a CHECK constraint enforces the mechanism floor, a UNIQUE makes
a statement the identity of a hypothesis, and a trigger counts trials on every
insert. Those constraints are the reason a verdict cannot be quietly un-made
(§5.1); in Mongo they would become rules someone has to remember.

Accounts have no such invariants across rows — a handful of documents, read
once per request — so which database holds them is a deployment choice rather
than a correctness one.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from core.clock import utc_now
from ops.mongo import mongo_available, mongo_database

__all__ = ["AccountRecord", "AccountStore", "MongoAccountStore", "active_store"]


@dataclass(frozen=True)
class AccountRecord:
    """One account, as either backend returns it."""

    user_id: str
    email: str
    display_name: str
    settings: dict[str, Any]
    created_at: datetime
    last_login_at: datetime | None
    #: Present only when the caller is verifying a password.
    password_hash: bytes | None = None
    password_salt: bytes | None = None
    scrypt_n: int = 0
    scrypt_r: int = 0
    scrypt_p: int = 0


class AccountStore(Protocol):
    """What the account endpoints need from a database."""

    def available(self) -> bool: ...

    def count(self) -> int: ...

    def find_by_email(self, email: str) -> AccountRecord | None: ...

    def find_by_session(self, token_hash: bytes) -> AccountRecord | None: ...

    def create(self, record: AccountRecord) -> AccountRecord: ...

    def start_session(self, token_hash: bytes, user_id: str, expires_at: datetime) -> None: ...

    def end_session(self, token_hash: bytes) -> None: ...

    def touch_login(self, user_id: str) -> None: ...

    def save_settings(self, user_id: str, settings: dict[str, Any]) -> AccountRecord | None: ...


@dataclass
class MongoAccountStore:
    """Accounts in MongoDB.

    **Emails are matched on a stored lowercased copy**, not with a
    case-insensitive query. A regex or a collation would work until the day the
    unique index does not use the same rule, at which point two accounts can
    exist that the lookup treats as one. The index and the query read the same
    field.
    """

    database_name: str = "neutron"

    def available(self) -> bool:
        if not mongo_available():
            return False
        with mongo_database() as database:
            return database is not None

    def count(self) -> int:
        with mongo_database() as database:
            if database is None:
                return 0
            return int(database["users"].count_documents({}))  # type: ignore[index]

    def find_by_email(self, email: str) -> AccountRecord | None:
        with mongo_database() as database:
            if database is None:
                return None
            found = database["users"].find_one({"email_lower": email.lower()})  # type: ignore[index]
        return _record(found) if found else None

    def find_by_session(self, token_hash: bytes) -> AccountRecord | None:
        with mongo_database() as database:
            if database is None:
                return None
            session = database["sessions"].find_one({"_id": token_hash})  # type: ignore[index]
            if session is None:
                return None
            # The TTL index removes expired sessions, but only when Mongo next
            # sweeps — up to a minute later. Checked here too, so a session is
            # never honoured past its expiry because of sweep timing.
            if session["expires_at"] <= utc_now().replace(tzinfo=None):
                database["sessions"].delete_one({"_id": token_hash})  # type: ignore[index]
                return None
            database["sessions"].update_one(  # type: ignore[index]
                {"_id": token_hash}, {"$set": {"last_seen_at": utc_now().replace(tzinfo=None)}}
            )
            found = database["users"].find_one({"_id": session["user_id"]})  # type: ignore[index]
        return _record(found) if found else None

    def create(self, record: AccountRecord) -> AccountRecord:
        with mongo_database() as database:
            if database is None:
                raise RuntimeError("MongoDB is not reachable")
            document = {
                "_id": record.user_id or str(uuid.uuid4()),
                "email": record.email,
                "email_lower": record.email.lower(),
                "display_name": record.display_name,
                "password_hash": record.password_hash,
                "password_salt": record.password_salt,
                "scrypt_n": record.scrypt_n,
                "scrypt_r": record.scrypt_r,
                "scrypt_p": record.scrypt_p,
                "settings": record.settings,
                "created_at": utc_now().replace(tzinfo=None),
                "last_login_at": None,
            }
            database["users"].insert_one(document)  # type: ignore[index]
        return _record(document)

    def start_session(self, token_hash: bytes, user_id: str, expires_at: datetime) -> None:
        with mongo_database() as database:
            if database is None:
                raise RuntimeError("MongoDB is not reachable")
            database["sessions"].insert_one(  # type: ignore[index]
                {
                    # The token itself is never stored — only its hash, so a
                    # leaked collection cannot be replayed as a live session.
                    "_id": token_hash,
                    "user_id": user_id,
                    "created_at": utc_now().replace(tzinfo=None),
                    "expires_at": expires_at.replace(tzinfo=None),
                    "last_seen_at": utc_now().replace(tzinfo=None),
                }
            )

    def end_session(self, token_hash: bytes) -> None:
        with mongo_database() as database:
            if database is not None:
                database["sessions"].delete_one({"_id": token_hash})  # type: ignore[index]

    def touch_login(self, user_id: str) -> None:
        with mongo_database() as database:
            if database is not None:
                database["users"].update_one(  # type: ignore[index]
                    {"_id": user_id}, {"$set": {"last_login_at": utc_now().replace(tzinfo=None)}}
                )

    def save_settings(self, user_id: str, settings: dict[str, Any]) -> AccountRecord | None:
        with mongo_database() as database:
            if database is None:
                return None
            database["users"].update_one(  # type: ignore[index]
                {"_id": user_id}, {"$set": {"settings": settings}}
            )
            found = database["users"].find_one({"_id": user_id})  # type: ignore[index]
        return _record(found) if found else None


def _record(document: dict[str, Any]) -> AccountRecord:
    return AccountRecord(
        user_id=str(document["_id"]),
        email=str(document.get("email", "")),
        display_name=str(document.get("display_name") or ""),
        settings=dict(document.get("settings") or {}),
        created_at=document.get("created_at") or utc_now(),
        last_login_at=document.get("last_login_at"),
        password_hash=document.get("password_hash"),
        password_salt=document.get("password_salt"),
        scrypt_n=int(document.get("scrypt_n") or 0),
        scrypt_r=int(document.get("scrypt_r") or 0),
        scrypt_p=int(document.get("scrypt_p") or 0),
    )


def active_store() -> MongoAccountStore | None:
    """The Mongo store when one is configured, else None.

    None means the Postgres path in `apps.api.accounts` is used. Chosen by
    configuration rather than by trying Mongo and falling back on error: a
    silent fallback would write an account to one database and look for it in
    the other, and the operator would see a login that "sometimes" fails.
    """
    return MongoAccountStore() if mongo_available() else None
