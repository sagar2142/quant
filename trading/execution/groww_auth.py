"""Minting a Groww access token — MASTER_PLAN §21.

**Groww issues an API key and a secret; the access token is derived from them.**
Asking an operator to paste a token was asking for the wrong thing: the token
is not something Groww hands out, it is something you mint, and it expires
every morning at 06:00 IST. A console that demands a fresh one daily is a
console nobody uses by Thursday.

The exchange, exactly as Groww's own client performs it:

    POST https://api.groww.in/v1/token/api/access
    Authorization: Bearer <api key>
    {"key_type": "approval",
     "checksum": sha256(secret + str(timestamp)),
     "timestamp": <unix seconds>}

The checksum is what proves possession of the secret without sending it, and
the timestamp is what stops a captured checksum being replayed tomorrow. The
secret never leaves this process.

**There is a hard budget: 150 mints per 24 hours.** So the token is cached
until it expires rather than fetched per request, and a 401 triggers exactly
one re-mint. Minting on every call would exhaust the day's allowance before
lunch and lock the account out of trading — a self-inflicted outage during
market hours.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

from core.clock import UTC, utc_now

__all__ = ["MintedToken", "groww_expiry", "mint_access_token"]

TOKEN_URL = "https://api.groww.in/v1/token/api/access"  # noqa: S105 - a URL, not a secret
REQUEST_TIMEOUT_SECONDS = 15.0

#: Groww invalidates every token at 06:00 IST, which is 00:30 UTC.
EXPIRY_HOUR_UTC = 0
EXPIRY_MINUTE_UTC = 30


class GrowwAuthError(RuntimeError):
    """A token could not be minted. Never swallowed: without one, nothing trades."""


@dataclass(frozen=True)
class MintedToken:
    token: str
    expires_at: datetime


def _checksum(secret: str, timestamp: int) -> str:
    """SHA-256 of the secret concatenated with the timestamp.

    Proves possession of the secret without transmitting it, and is worthless
    tomorrow because the timestamp is part of what was hashed.
    """
    return hashlib.sha256(f"{secret}{timestamp}".encode()).hexdigest()


def groww_expiry(now: datetime | None = None) -> datetime:
    """The next 06:00 IST, as UTC.

    Computed rather than assumed to be "in 24 hours": a token minted at 05:50
    IST is valid for ten minutes, and treating it as a day old is how a
    position gets opened with a credential that expired mid-order.
    """
    moment = now or utc_now()
    expiry = moment.replace(
        hour=EXPIRY_HOUR_UTC, minute=EXPIRY_MINUTE_UTC, second=0, microsecond=0, tzinfo=UTC
    )
    if expiry <= moment:
        expiry += timedelta(days=1)
    return expiry


def _raise_for_status(response: httpx.Response) -> None:
    """Turn a failed mint into a message that says what to do about it."""
    if response.status_code == httpx.codes.BAD_REQUEST:
        # Groww puts the useful part in a nested display message. Surfaced,
        # because "Bad Request" alone does not distinguish a wrong secret from
        # a key whose approval was never granted.
        try:
            detail = response.json().get("error", {}).get("displayMessage", "")
        except ValueError:
            detail = ""
        raise GrowwAuthError(f"Groww rejected the key and secret: {detail or 'bad request'}")
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        raise GrowwAuthError(
            "Groww's token limit is 150 mints per 24 hours and it has been reached. "
            "The existing token stays valid until 06:00 IST."
        )
    if not response.is_success:
        raise GrowwAuthError(f"Groww returned HTTP {response.status_code} minting a token")


def mint_access_token(api_key: str, secret: str, client: httpx.Client | None = None) -> MintedToken:
    """Exchange an API key and secret for a live access token.

    Raises:
        GrowwAuthError: on any failure. Never returns a placeholder — a caller
            holding a fabricated token would send orders that are rejected at
            the venue, one at a time, with no obvious cause.
    """
    if not api_key.strip():
        raise GrowwAuthError("Groww API key is empty")
    if not secret.strip():
        raise GrowwAuthError("Groww API secret is empty")

    timestamp = int(time.time())
    payload = {
        "key_type": "approval",
        "checksum": _checksum(secret.strip(), timestamp),
        "timestamp": timestamp,
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "x-request-id": str(uuid.uuid4()),
        "x-client-id": "growwapi",
        "x-api-version": "1.0",
    }

    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        response = http.post(TOKEN_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise GrowwAuthError(f"could not reach Groww to mint a token: {exc}") from exc
    finally:
        if client is None:
            http.close()

    _raise_for_status(response)

    try:
        token = response.json()["token"]
    except (ValueError, KeyError, TypeError) as exc:
        raise GrowwAuthError("Groww's token response did not contain a token") from exc
    if not isinstance(token, str) or not token:
        raise GrowwAuthError("Groww returned an empty token")

    return MintedToken(token=token, expires_at=groww_expiry())
