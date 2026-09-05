"""Accounts on the Mongo backend — MASTER_PLAN §13.7.

`apps/api/account_store.py` existed for a while with nothing importing it:
`accounts.py` went straight to Postgres, so a configured Mongo cluster sat
empty while every account went somewhere else. These cover the wiring that
fixes that, and the properties that must hold identically on both backends.

**Against a fake, not against Atlas.** CI has no cluster, and a test suite that
needs one is a test suite that gets skipped. The fake implements the same
`AccountStore` protocol the real store does, so what is tested is the routing
and the security properties — the parts that were missing — rather than
pymongo, which is not this project's to test.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api import accounts as accounts_module
from apps.api.account_store import AccountRecord
from apps.api.accounts import build_accounts_router
from core.clock import utc_now

PASSWORD = "correct-horse-battery"
EMAIL = "operator@example.invalid"


class FakeStore:
    """An in-memory `AccountStore`, matching the Mongo one's semantics.

    In particular it matches the two that are easy to get wrong: emails are
    looked up on a stored lowercased copy (so uniqueness and lookup use the
    same rule), and an expired session is refused on read rather than waiting
    for a sweep to remove it.
    """

    def __init__(self, reachable: bool = True) -> None:
        self.reachable = reachable
        self.users: dict[str, AccountRecord] = {}
        self.sessions: dict[bytes, tuple[str, datetime]] = {}

    def available(self) -> bool:
        return self.reachable

    def count(self) -> int:
        return len(self.users)

    def find_by_email(self, email: str) -> AccountRecord | None:
        for record in self.users.values():
            if record.email.lower() == email.lower():
                return record
        return None

    def find_by_session(self, token_hash: bytes) -> AccountRecord | None:
        found = self.sessions.get(token_hash)
        if found is None:
            return None
        user_id, expires_at = found
        if expires_at <= utc_now():
            del self.sessions[token_hash]
            return None
        return self.users.get(user_id)

    def create(self, record: AccountRecord) -> AccountRecord:
        self.users[record.user_id] = record
        return record

    def start_session(self, token_hash: bytes, user_id: str, expires_at: datetime) -> None:
        self.sessions[token_hash] = (user_id, expires_at)

    def end_session(self, token_hash: bytes) -> None:
        self.sessions.pop(token_hash, None)

    def touch_login(self, user_id: str) -> None:
        record = self.users.get(user_id)
        if record is not None:
            self.users[user_id] = AccountRecord(**{**record.__dict__, "last_login_at": utc_now()})

    def save_settings(self, user_id: str, settings: dict[str, Any]) -> AccountRecord | None:
        record = self.users.get(user_id)
        if record is None:
            return None
        updated = AccountRecord(**{**record.__dict__, "settings": dict(settings)})
        self.users[user_id] = updated
        return updated


@pytest.fixture
def mongo(monkeypatch):
    """Route the account endpoints at the fake store."""
    store = FakeStore()
    monkeypatch.setattr(accounts_module, "active_store", lambda: store)

    app = FastAPI()
    app.include_router(build_accounts_router())
    return TestClient(app), store


def register(client, email: str = EMAIL, password: str = PASSWORD, **extra):
    return client.post("/account/register", json={"email": email, "password": password, **extra})


class TestRouting:
    def test_registering_writes_to_the_configured_store(self, mongo) -> None:
        """The bug this file exists for: the Mongo backend was written, the
        cluster was configured and reachable, and every account still went to
        Postgres because nothing called `active_store`."""
        client, store = mongo
        assert register(client).status_code == 201
        assert store.count() == 1

    def test_status_counts_from_the_same_store(self, mongo) -> None:
        client, _ = mongo
        register(client)
        body = client.get("/account/status").json()
        assert body["user_count"] == 1
        assert body["accounts_available"] is True

    def test_an_unreachable_store_is_reported_not_hidden(self, mongo) -> None:
        client, store = mongo
        store.reachable = False
        body = client.get("/account/status").json()
        assert body["accounts_available"] is False
        assert body["user_count"] == 0

    def test_registering_against_an_unreachable_store_is_a_503(self, mongo) -> None:
        client, store = mongo
        store.reachable = False
        assert register(client).status_code == 503


class TestLifecycle:
    def test_register_signs_in(self, mongo) -> None:
        client, _ = mongo
        register(client)
        assert client.get("/account/status").json()["signed_in"] is True

    def test_a_duplicate_email_is_refused_case_insensitively(self, mongo) -> None:
        """Uniqueness and lookup must use the same rule. If they diverge, two
        accounts can exist that the lookup treats as one."""
        client, _ = mongo
        register(client)
        assert register(client, email=EMAIL.upper()).status_code == 409

    def test_sign_out_deletes_the_session_rather_than_forgetting_it(self, mongo) -> None:
        client, store = mongo
        register(client)
        client.post("/account/logout")
        assert store.sessions == {}
        assert client.get("/account/status").json()["signed_in"] is False

    def test_sign_in_restores_the_account(self, mongo) -> None:
        client, _ = mongo
        register(client)
        client.post("/account/logout")
        response = client.post("/account/login", json={"email": EMAIL, "password": PASSWORD})
        assert response.status_code == 200
        assert response.json()["email"] == EMAIL

    def test_the_email_case_used_at_sign_in_does_not_matter(self, mongo) -> None:
        client, _ = mongo
        register(client)
        client.post("/account/logout")
        assert (
            client.post(
                "/account/login", json={"email": EMAIL.upper(), "password": PASSWORD}
            ).status_code
            == 200
        )


class TestPasswords:
    def test_the_password_is_never_stored(self, mongo) -> None:
        client, store = mongo
        register(client)
        record = next(iter(store.users.values()))
        assert isinstance(record.password_hash, bytes)
        assert PASSWORD.encode() not in record.password_hash
        assert not hasattr(record, "password")

    def test_a_wrong_password_is_refused(self, mongo) -> None:
        client, _ = mongo
        register(client)
        client.post("/account/logout")
        assert (
            client.post(
                "/account/login", json={"email": EMAIL, "password": "not-the-password"}
            ).status_code
            == 401
        )

    def test_an_unknown_email_gives_the_same_answer_as_a_wrong_password(self, mongo) -> None:
        """Distinguishing them tells a prober which addresses exist."""
        client, _ = mongo
        register(client)
        client.post("/account/logout")
        wrong = client.post("/account/login", json={"email": EMAIL, "password": "nope-nope-nope"})
        unknown = client.post(
            "/account/login", json={"email": "nobody@example.invalid", "password": PASSWORD}
        )
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json() == unknown.json()

    def test_the_scrypt_parameters_travel_with_the_digest(self, mongo) -> None:
        """So they can be raised later without locking out every account."""
        client, store = mongo
        register(client)
        record = next(iter(store.users.values()))
        assert record.scrypt_n == accounts_module.SCRYPT_N
        assert record.scrypt_r == accounts_module.SCRYPT_R
        assert record.scrypt_p == accounts_module.SCRYPT_P

    def test_a_short_password_is_refused(self, mongo) -> None:
        client, _ = mongo
        assert register(client, password="short").status_code == 422


class TestSessions:
    def test_only_the_hash_of_the_token_is_stored(self, mongo) -> None:
        """A leaked sessions collection must not be replayable."""
        client, store = mongo
        register(client)
        token = client.cookies.get(accounts_module.SESSION_COOKIE)
        assert token
        stored = next(iter(store.sessions))
        assert isinstance(stored, bytes)
        assert token.encode() != stored
        assert accounts_module._token_hash(token) == stored

    def test_an_expired_session_is_refused(self, mongo) -> None:
        """Not honoured until a sweep gets round to it."""
        client, store = mongo
        register(client)
        key = next(iter(store.sessions))
        user_id, _ = store.sessions[key]
        store.sessions[key] = (user_id, utc_now() - timedelta(seconds=1))
        assert client.get("/account/status").json()["signed_in"] is False

    def test_an_unknown_token_is_not_signed_in(self, mongo) -> None:
        client, _ = mongo
        client.cookies.set(accounts_module.SESSION_COOKIE, "not-a-real-token")
        assert client.get("/account/status").json()["signed_in"] is False


class TestSettings:
    def test_settings_persist_across_a_sign_out(self, mongo) -> None:
        client, _ = mongo
        register(client)
        client.put("/account/settings", json={"default_venue": "BSE"})
        client.post("/account/logout")
        body = client.post("/account/login", json={"email": EMAIL, "password": PASSWORD}).json()
        assert body["settings"]["default_venue"] == "BSE"

    def test_settings_merge_rather_than_replace(self, mongo) -> None:
        """A console that knows three settings must not erase a fourth it has
        never heard of."""
        client, _ = mongo
        register(client)
        client.put("/account/settings", json={"default_venue": "BSE"})
        body = client.put("/account/settings", json={"refresh_seconds": 5}).json()
        assert body["settings"] == {"default_venue": "BSE", "refresh_seconds": 5}

    def test_changing_settings_needs_a_session(self, mongo) -> None:
        client, _ = mongo
        assert client.put("/account/settings", json={"default_venue": "BSE"}).status_code == 401

    def test_an_invalid_venue_is_refused(self, mongo) -> None:
        client, _ = mongo
        register(client)
        assert client.put("/account/settings", json={"default_venue": "NYSE"}).status_code == 422

    def test_write_settings_reaches_the_same_store(self, mongo) -> None:
        """The broker-key router owns a slice of the same document and must
        not write it to a different database than the one the login used."""
        client, store = mongo
        register(client)
        user_id = next(iter(store.users))
        accounts_module.write_settings(user_id, {"broker": "groww"})
        assert store.users[user_id].settings == {"broker": "groww"}
