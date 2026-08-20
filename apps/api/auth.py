"""API authentication — MASTER_PLAN §13.7, §21.

**The design decision, stated plainly: reads are open on loopback, mutations
are closed until you configure a token.**

The threat model is not a stranger on the internet — nothing binds beyond
127.0.0.1. It is a *browser*, which is already inside that boundary. A page the
operator visits can reach a loopback API, and the Host check added alongside
this closes the rebinding path but is not an authorisation decision.

So the split follows consequence rather than convenience:

    reads       analytics, screens, the book. Worth nothing to an attacker and
                needed constantly by the operator. Open when no token is set.
    mutations   the kill switch. Refused outright until `NEUTRON_API_TOKEN`
                exists, and then required on every request.

**Refusing rather than defaulting is the whole point.** A generated-on-boot
token would be convenient and would mean a fresh install silently exposes
`/kill/release` to anything that can guess or read it. An install that has
never been configured cannot release a halt at all — which is the fail-safe
direction, because engaging a halt is recoverable and releasing one is not.

**Setting a token closes everything.** Once configured it is required on reads
too, so the same deployment that would be reachable from another machine is
not quietly serving its position book to it.

Compared with `secrets.compare_digest`, so a wrong token takes the same time as
a right one and cannot be recovered a byte at a time.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from core.config import Settings, settings

__all__ = ["require_read", "require_write", "token_is_configured"]

#: What an unauthenticated caller is told. Deliberately identical for "no token
#: sent" and "wrong token": distinguishing them confirms to a prober that a
#: token exists and that theirs was merely wrong.
_DENIED = "authentication required"


def token_is_configured(config: Settings | None = None) -> bool:
    return bool((config or settings).api_token)


def _presented(header: str | None) -> str:
    """The token from an Authorization header, or empty.

    Accepts `Bearer <token>` and a bare token. The bare form exists because a
    curl one-liner is the normal way this API gets poked during operations, and
    a scheme that only works from the console is a scheme that gets bypassed.
    """
    if not header:
        return ""
    prefix, _, rest = header.partition(" ")
    return rest.strip() if prefix.lower() == "bearer" else header.strip()


def _matches(header: str | None, config: Settings) -> bool:
    expected = config.api_token
    if not expected:
        return False
    return secrets.compare_digest(_presented(header), expected)


def require_read(authorization: Annotated[str | None, Header()] = None) -> None:
    """Guard a read endpoint.

    Open while no token is configured — a fresh checkout can browse its own
    analytics without ceremony. Once a token exists it is required, because a
    deployment worth protecting is protected uniformly.
    """
    if not token_is_configured():
        return
    if not _matches(authorization, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_DENIED,
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_write(authorization: Annotated[str | None, Header()] = None) -> None:
    """Guard a mutating endpoint. The kill switch is the only one.

    Raises:
        HTTPException: 503 when no token is configured, because the endpoint is
            genuinely unavailable rather than the caller unauthorised — and the
            message says how to enable it. 401 when a token exists and the
            caller did not present it.
    """
    if not token_is_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "the kill switch is disabled because no API token is set. "
                "Put NEUTRON_API_TOKEN in .env and restart. Releasing a halt "
                "must be a deliberate act by a configured operator (§21)."
            ),
        )
    if not _matches(authorization, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_DENIED,
            headers={"WWW-Authenticate": "Bearer"},
        )


#: Ready-made dependencies, so an endpoint declares its class of access rather
#: than repeating the wiring.
ReadAccess = Depends(require_read)
WriteAccess = Depends(require_write)
