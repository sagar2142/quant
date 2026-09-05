"""Per-account broker credentials — MASTER_PLAN §21.

**A reversal, stated openly.** The account module says broker keys stay in the
environment, because a database row is backed up, replicated and readable by
whoever can query it. That is still true. What changed is that keeping them
only in `.env` means editing a file and restarting a process every morning,
which is a workflow people abandon — and an abandoned safeguard protects
nothing.

So credentials may be stored, under conditions that address the original
objection:

    encrypted at rest    Fernet, with a key from `NEUTRON_SECRET_KEY` that is
                         not in the database. A dumped collection is useless
                         without it.
    never returned       Reads answer "configured" and the last four characters.
                         Nothing here can hand a secret back to a browser.
    refused unencrypted  No key configured means storing is refused, not
                         downgraded to plaintext. A system at its least
                         protected when nobody is watching is worse than one
                         that says no.
    per account          Keys belong to the operator who added them, and are
                         used only for that operator's orders.

**The environment still wins.** If `GROWW_ACCESS_TOKEN` is set in the process,
it is used ahead of anything stored — a deployment that pins credentials
explicitly should not have them silently overridden by whatever is in a
database.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from apps.api.accounts import SessionCookie, current_account
from core.vault import VaultError, decrypt, encrypt, vault_configured

__all__ = ["build_broker_keys_router", "stored_credentials"]

#: Which brokers an operator may configure from the console.
#:
#: Groww alone. Kite was offered because the adapter exists, but a second
#: broker in the picker is a second thing to get wrong for an operator who has
#: one account — and its credential shape differs, so the form would have to
#: explain which fields apply to which. `NEUTRON_BROKER=kite` still works for
#: anyone who wants it; it simply is not offered here.
SUPPORTED = ("groww",)

#: Where they live inside the account's settings document.
SETTINGS_KEY = "broker_keys"

#: How much of a key is shown back. Enough to tell two keys apart, far too
#: little to reconstruct one.
HINT_LENGTH = 4


#: Upper bound on any stored credential.
#:
#: One number for all three, because guessing per-field lengths was wrong twice
#: in a row. A Groww access token is a JWT and ran past 512; then the API key
#: turned out to be a JWT as well — it is passed straight through as a bearer
#: token — and ran past it too. Each time a valid credential was refused by a
#: limit the operator had no reason to know about.
#:
#: The bound exists to stop something absurd being written to a database, not
#: to second-guess a broker's credential format. Brokers change those; this
#: does not need to care.
MAX_CREDENTIAL_LENGTH = 8192


class BrokerKeysRequest(BaseModel):
    broker: str = Field(pattern="^(groww)$")
    api_key: str = Field(min_length=1, max_length=MAX_CREDENTIAL_LENGTH)
    #: Groww has no separate secret; Kite does. Nullable as well as optional,
    #: because a form that does not render the field sends `null` rather than
    #: omitting it, and refusing that is refusing the ordinary case.
    api_secret: str | None = Field(default="", max_length=MAX_CREDENTIAL_LENGTH)
    #: Optional. Groww mints this from the key and secret and refreshes it
    #: automatically, so an operator normally never sees one. Accepted for
    #: anyone who has minted a token out of band.
    access_token: str = Field(default="", max_length=MAX_CREDENTIAL_LENGTH)


class BrokerKeysStatus(BaseModel):
    broker: str
    configured: bool
    #: Last four characters only, so an operator can tell which key is stored
    #: without the value being recoverable from the screen.
    api_key_hint: str
    access_token_hint: str
    updated_at: str | None


class VaultStatus(BaseModel):
    #: Whether anything can be stored at all.
    encryption_available: bool
    brokers: list[BrokerKeysStatus]
    #: True when the process environment pins credentials, which take priority.
    environment_overrides: bool


def _hint(value: str) -> str:
    """The last four characters, or nothing. Never the whole value."""
    return f"…{value[-HINT_LENGTH:]}" if len(value) >= HINT_LENGTH else ""


def _environment_pins(broker: str) -> bool:
    import os  # noqa: PLC0415 - read late, so `.env` has been applied

    return bool(os.environ.get(f"{broker.upper()}_ACCESS_TOKEN", "").strip())


def stored_credentials(settings: dict[str, Any], broker: str) -> dict[str, str] | None:
    """Decrypt one broker's stored credentials, or None.

    Returns None rather than raising when nothing is stored. A decryption
    failure *does* raise: a credential that cannot be read is not an absent
    credential, and silently falling back to "none configured" would send an
    operator hunting for a missing key that is actually there and unreadable.
    """
    keys = settings.get(SETTINGS_KEY)
    if not isinstance(keys, dict):
        return None
    entry = keys.get(broker)
    # A key and secret are enough; the token is derived from them.
    if not isinstance(entry, dict) or not entry.get("api_key"):
        return None
    return {
        "api_key": decrypt(str(entry.get("api_key", ""))),
        "api_secret": decrypt(str(entry.get("api_secret", ""))),
        "access_token": decrypt(str(entry.get("access_token", ""))),
    }


def build_broker_keys_router(
    save_settings: Callable[[str, dict[str, Any]], None],
) -> APIRouter:
    """Store, inspect and remove per-account broker credentials.

    Args:
        save_settings: The account module's settings writer, injected so this
            router does not need to know which database holds accounts.
    """
    router = APIRouter(prefix="/account/broker", tags=["account"])

    @router.get("", response_model=VaultStatus)
    def status(session: SessionCookie = None) -> VaultStatus:
        """What is configured. Never what the values are."""
        account = current_account(session)
        settings = account.settings if account else {}
        keys = settings.get(SETTINGS_KEY) if isinstance(settings, dict) else None
        stored = keys if isinstance(keys, dict) else {}

        return VaultStatus(
            encryption_available=vault_configured(),
            environment_overrides=any(_environment_pins(b) for b in SUPPORTED),
            brokers=[
                BrokerKeysStatus(
                    broker=broker,
                    configured=bool(stored.get(broker, {}).get("api_key")),
                    api_key_hint=str(stored.get(broker, {}).get("api_key_hint", "")),
                    access_token_hint=str(stored.get(broker, {}).get("access_token_hint", "")),
                    updated_at=stored.get(broker, {}).get("updated_at"),
                )
                for broker in SUPPORTED
            ],
        )

    @router.put("", response_model=VaultStatus)
    def store(request: BrokerKeysRequest, session: SessionCookie = None) -> VaultStatus:
        """Encrypt and store one broker's credentials."""
        account = current_account(session)
        if account is None:
            raise HTTPException(status_code=401, detail="sign in to store broker credentials")
        if not vault_configured():
            raise HTTPException(
                status_code=409,
                detail=(
                    "NEUTRON_SECRET_KEY is not set, so credentials cannot be encrypted. "
                    "Generate one with: python -m apps.cli.vault --generate. "
                    "Storing them unencrypted is not offered."
                ),
            )

        from core.clock import utc_now  # noqa: PLC0415

        try:
            entry = {
                "api_key": encrypt(request.api_key),
                "api_secret": encrypt(request.api_secret or ""),
                "access_token": encrypt(request.access_token),
                # Hints are stored in clear on purpose: they are four
                # characters, they are what makes "which key is this"
                # answerable, and they are useless to anyone else.
                "api_key_hint": _hint(request.api_key),
                "access_token_hint": _hint(request.access_token),
                "updated_at": utc_now().isoformat(),
            }
        except VaultError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        existing = account.settings.get(SETTINGS_KEY)
        keys = dict(existing) if isinstance(existing, dict) else {}
        keys[request.broker] = entry
        save_settings(account.user_id, {**account.settings, SETTINGS_KEY: keys})
        return status(session)

    @router.delete("/{broker}", response_model=VaultStatus)
    def remove(broker: str, session: SessionCookie = None) -> VaultStatus:
        """Forget one broker's credentials."""
        account = current_account(session)
        if account is None:
            raise HTTPException(status_code=401, detail="sign in to remove broker credentials")
        existing = account.settings.get(SETTINGS_KEY)
        keys = dict(existing) if isinstance(existing, dict) else {}
        keys.pop(broker, None)
        save_settings(account.user_id, {**account.settings, SETTINGS_KEY: keys})
        return status(session)

    return router
