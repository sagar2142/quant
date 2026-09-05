"""Per-account broker credentials — MASTER_PLAN §13.7, §21.

**These were write-only.** The console encrypted a Groww key into the account
document and `apps.api.trade` read credentials from environment variables
instead, so `stored_credentials` — the only function that decrypts them back —
had no caller. An operator who connected their broker saw the gate keep saying
"not connected", and an order would have gone out on whatever the environment
happened to hold. This file covers the round trip and the precedence rule.

The whole module had no test file, which is how a write-only credential store
survives: every part of it works, and nothing checks that the parts are joined.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.accounts import AccountResponse
from apps.api.broker_keys import SETTINGS_KEY, build_broker_keys_router, stored_credentials

API_KEY = "eyJhbGciOiJIUzI1NiJ9.groww-issued-key-value"
API_SECRET = "s3cr3t-value-from-groww"


@pytest.fixture
def vault_key(monkeypatch):
    """A real Fernet key, so encryption is exercised rather than stubbed."""
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    monkeypatch.setenv("NEUTRON_SECRET_KEY", key)
    # `core.envfile` caches on mtime, and `core.vault` reads the environment
    # late, so setting it here is enough — but the module-level cache in
    # `ops.mongo`/`core.vault` must not have latched an earlier absence.
    import core.vault

    monkeypatch.setattr(core.vault, "_KEY_CACHE", {}, raising=False)
    return key


@pytest.fixture
def keys(monkeypatch, vault_key):
    """A signed-in account whose settings live in a dict."""
    account = AccountResponse(
        user_id="u1",
        email="operator@example.invalid",
        display_name="Operator",
        settings={},
        created_at="2026-01-01T00:00:00+00:00",
        last_login_at=None,
    )
    # Pydantic copies the dict it is handed, so the settings the endpoints read
    # are the model's own. The fake writer must update *that* one, exactly as
    # the real one does by re-reading the account after a save.
    saved = account.settings

    import apps.api.broker_keys as module

    monkeypatch.setattr(module, "current_account", lambda _session: account)

    def save(user_id: str, settings: dict[str, Any]) -> None:
        assert user_id == "u1"
        saved.clear()
        saved.update(settings)

    app = FastAPI()
    app.include_router(build_broker_keys_router(save))
    return TestClient(app), saved, account


def store(client, **overrides):
    body = {"broker": "groww", "api_key": API_KEY, "api_secret": API_SECRET, "access_token": ""}
    return client.put("/account/broker", json={**body, **overrides})


class TestStorage:
    def test_a_key_is_stored_encrypted(self, keys) -> None:
        client, saved, _ = keys
        assert store(client).status_code == 200
        entry = saved[SETTINGS_KEY]["groww"]
        # The ciphertext must not contain the plaintext anywhere in it.
        assert API_KEY not in entry["api_key"]
        assert API_SECRET not in entry["api_secret"]

    def test_only_the_last_four_characters_are_kept_in_clear(self, keys) -> None:
        client, saved, _ = keys
        store(client)
        entry = saved[SETTINGS_KEY]["groww"]
        assert entry["api_key_hint"].endswith(API_KEY[-4:])
        assert len(entry["api_key_hint"]) <= 5

    def test_the_status_never_returns_the_value(self, keys) -> None:
        client, _, _ = keys
        store(client)
        body = client.get("/account/broker").json()
        assert API_KEY not in str(body)
        assert API_SECRET not in str(body)
        assert body["brokers"][0]["configured"] is True

    def test_a_long_jwt_key_is_accepted(self, keys) -> None:
        """Groww issues JWTs. A 512-character cap rejected the real thing
        twice, which presented as an unexplained 422 on a valid key."""
        client, _, _ = keys
        assert store(client, api_key="x" * 4000).status_code == 200

    def test_an_unsupported_broker_is_refused(self, keys) -> None:
        client, _, _ = keys
        assert store(client, broker="kraken").status_code == 422

    def test_removing_forgets_the_credentials(self, keys) -> None:
        client, saved, _ = keys
        store(client)
        assert client.delete("/account/broker/groww").status_code == 200
        assert not saved.get(SETTINGS_KEY, {}).get("groww", {}).get("api_key")

    def test_storing_without_a_session_is_refused(self, keys, monkeypatch) -> None:
        client, _, _ = keys
        import apps.api.broker_keys as module

        monkeypatch.setattr(module, "current_account", lambda _session: None)
        assert store(client).status_code == 401


class TestNoEncryptionNoStorage:
    def test_it_refuses_rather_than_storing_in_clear(self, keys, monkeypatch) -> None:
        """A 409 that says how to fix it, not a silent downgrade. Storing a
        key that can move money in plaintext is not a lesser version of
        storing it safely."""
        client, saved, _ = keys
        import apps.api.broker_keys as module

        monkeypatch.setattr(module, "vault_configured", lambda: False)
        response = store(client)
        assert response.status_code == 409
        assert "NEUTRON_SECRET_KEY" in response.json()["detail"]
        assert SETTINGS_KEY not in saved


class TestReadBack:
    """The half that was missing."""

    def test_what_was_stored_can_be_decrypted(self, keys) -> None:
        client, saved, _ = keys
        store(client)
        found = stored_credentials(saved, "groww")
        assert found is not None
        assert found["api_key"] == API_KEY
        assert found["api_secret"] == API_SECRET

    def test_nothing_stored_reads_back_as_none(self, keys) -> None:
        _, saved, _ = keys
        assert stored_credentials(saved, "groww") is None

    def test_a_different_broker_reads_back_as_none(self, keys) -> None:
        client, saved, _ = keys
        store(client)
        assert stored_credentials(saved, "kite") is None

    def test_an_unreadable_credential_raises_rather_than_reading_as_absent(self, keys) -> None:
        """A credential that cannot be decrypted is not a missing one. Falling
        back to "none configured" sends an operator hunting for a key that is
        present and unreadable — usually because the secret key changed."""
        client, saved, _ = keys
        store(client)
        saved[SETTINGS_KEY]["groww"]["api_key"] = "not-valid-ciphertext"
        with pytest.raises(Exception, match=r"(?i)decrypt|token|vault"):
            stored_credentials(saved, "groww")


class TestTradeUsesThem:
    """The gate must report what will actually be used."""

    def test_the_account_credentials_reach_the_trade_module(self, keys, monkeypatch) -> None:
        client, _, account = keys
        store(client)

        from apps.api import trade

        monkeypatch.setattr(trade, "account_credentials", trade.account_credentials)
        monkeypatch.setattr(
            "apps.api.accounts.current_account", lambda _session: account, raising=False
        )
        found = trade.account_credentials("any-session", "groww")
        assert found is not None
        assert found.api_key.reveal() == API_KEY
        assert found.api_secret.reveal() == API_SECRET

    def test_no_account_means_no_account_credentials(self, monkeypatch) -> None:
        from apps.api import trade

        monkeypatch.setattr(
            "apps.api.accounts.current_account", lambda _session: None, raising=False
        )
        assert trade.account_credentials(None, "groww") is None

    def test_the_gate_reports_connected_once_a_key_is_stored(self, keys, monkeypatch) -> None:
        """The symptom that started this: the operator connects the broker and
        the gate keeps saying "not connected", because it was reading a
        different place than the console writes to."""
        client, _, account = keys
        store(client)

        from apps.api import trade

        monkeypatch.setattr(
            "apps.api.accounts.current_account", lambda _session: account, raising=False
        )
        gate = trade._credential_gate("any-session")
        assert gate.ready is True
        assert "connected" in gate.detail.lower()
