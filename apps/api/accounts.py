"""Operator accounts — MASTER_PLAN §13.7, §21.

**What this is, and what it is not.** This identifies which operator is acting
and carries their preferences. It is not, on its own, a security boundary: the
API binds to loopback and anything already on this machine can call it
directly. `NEUTRON_API_TOKEN` is what decides whether the API answers at all
(see `apps.api.auth`), and an account screen in front of an open API would be
theatre — which on a system that can place real orders is worse than nothing.

That distinction is reported honestly by `GET /account/status`, so the console
can say "signed in, and the API is open to anything on this machine" rather
than implying a lock that is not there.

**Passwords are hashed with scrypt from the standard library.** Memory-hard, so
a stolen table cannot be brute-forced at the speed a plain digest allows. The
parameters are stored per row, so they can be raised later without invalidating
every existing password.

**Session tokens are never stored.** What is stored is their SHA-256, so a
leaked `sessions` table cannot be replayed. Comparison is constant-time
throughout: a wrong password and a wrong token both take the same time as a
right one.

**No broker credentials here.** Kite keys stay in the environment. A database
row is backed up, replicated and readable by anyone who can query it, and a key
that can move money should be in none of those places.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Cookie, HTTPException, Response, status
from pydantic import BaseModel, Field

from apps.api.account_store import AccountRecord, MongoAccountStore, active_store
from apps.api.auth import token_is_configured
from core.clock import utc_now
from ops.db import optional_connection

__all__ = [
    "SESSION_COOKIE",
    "SessionCookie",
    "StoredPassword",
    "build_accounts_router",
    "current_account",
    "hash_password",
    "verify_password",
    "write_settings",
]

#: Where the session token lives. HttpOnly so page scripts cannot read it, and
#: SameSite=Lax so another site cannot ride it.
SESSION_COOKIE = "neutron_session"

#: How long a session lasts. Deliberately short for something that fronts a
#: trading system; a desk left unlocked overnight should not still be signed in.
SESSION_HOURS = 12

#: scrypt work factors. `n` dominates both time and memory; 2^14 with r=8 is
#: roughly 16MB per hash, which is slow enough to matter to an attacker and
#: unnoticeable on a single interactive login.
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32

MIN_PASSWORD_LENGTH = 12

#: Enough to reject a typo, not enough to reject a real address.
#:
#: Deliberately not `EmailStr`: that needs `email-validator`, which is not in
#: this project's dependencies, and adding a package so a single-operator
#: console can pedantically parse an address it will never send mail to is a
#: poor trade. Anything with a local part, an @ and a dot in the domain passes.
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class RegisterRequest(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)
    display_name: str = Field(default="", max_length=80)


class LoginRequest(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class SettingsRequest(BaseModel):
    """Display and workflow preferences.

    Nothing here decides money. Position limits and the kill switch belong to
    the risk engine, not to a per-user blob an account could quietly widen.
    """

    #: Which exchange every screen opens on.
    default_venue: str | None = Field(default=None, pattern="^(NSE|BSE)$")
    #: Capital the order ticket sizes against. A string, because it is money
    #: and money is never a float (§14.1.2).
    default_capital: str | None = None
    #: Sessions of history a chart loads by default. 0 is everything.
    default_sessions: int | None = Field(default=None, ge=0, le=10_000)
    #: Candle size charts open at.
    default_interval: str | None = Field(default=None, pattern="^(1m|5m|10m|1h|4h|1d|1w)$")
    #: Moving averages drawn over price. Empty means none.
    chart_overlays: list[int] | None = None
    #: Logarithmic price axis by default — right for multi-year spans, where a
    #: linear axis makes an old 10% move look smaller than a recent one.
    default_log_scale: bool | None = None
    #: Order type the ticket opens on.
    default_order_type: str | None = Field(default=None, pattern="^(MARKET|LIMIT)$")
    #: Names the watchlist starts with.
    watchlist: list[str] | None = None
    #: Seconds between vitals polls. Zero stops polling entirely.
    refresh_seconds: int | None = Field(default=None, ge=0, le=3600)


class AccountResponse(BaseModel):
    user_id: str
    email: str
    display_name: str
    settings: dict[str, Any]
    created_at: str
    last_login_at: str | None


class StatusResponse(BaseModel):
    signed_in: bool
    account: AccountResponse | None
    #: Whether accounts can be used at all — they need the database.
    accounts_available: bool
    #: Whether the API itself requires a token. False means an account is
    #: identification, not protection, and the console says so.
    api_token_configured: bool
    #: How many accounts exist. A fresh install shows "register", not "sign in".
    user_count: int
    #: Which database holds accounts: "mongodb" or "postgres". Reported so the
    #: console can name the right thing to start. It used to tell every
    #: operator to start Postgres, which is wrong advice on a Mongo install and
    #: sends them to debug a database the system is not using.
    backend: str = "postgres"


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Hash a password with scrypt. Returns (digest, salt)."""
    chosen = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=chosen,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )
    return digest, chosen


@dataclass(frozen=True)
class StoredPassword:
    """A password as the database holds it, parameters included.

    The work factors travel with the digest rather than being read from the
    current constants, so raising them later re-hashes new passwords without
    locking out every existing one.
    """

    digest: bytes
    salt: bytes
    n: int
    r: int
    p: int


def verify_password(password: str, stored: StoredPassword) -> bool:
    """Constant-time check against a stored digest."""
    candidate = hashlib.scrypt(
        password.encode("utf-8"),
        salt=stored.salt,
        n=stored.n,
        r=stored.r,
        p=stored.p,
        dklen=len(stored.digest),
        maxmem=64 * 1024 * 1024,
    )
    return hmac.compare_digest(candidate, stored.digest)


def _token_hash(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _account(row: tuple[Any, ...]) -> AccountResponse:
    return AccountResponse(
        user_id=str(row[0]),
        email=str(row[1]),
        display_name=str(row[2] or ""),
        settings=row[3] or {},
        created_at=row[4].isoformat(),
        last_login_at=row[5].isoformat() if row[5] else None,
    )


def _from_record(record: AccountRecord) -> AccountResponse:
    """A Mongo document, in the same shape the Postgres path returns.

    One response model for both backends, so the console cannot tell which
    database it is talking to — and so a bug in one backend shows up as a
    difference in behaviour rather than a difference in schema.
    """
    return AccountResponse(
        user_id=record.user_id,
        email=record.email,
        display_name=record.display_name,
        settings=record.settings,
        created_at=record.created_at.isoformat(),
        last_login_at=record.last_login_at.isoformat() if record.last_login_at else None,
    )


def _set_session_cookie(response: Response, token: str) -> None:
    """Put the session token in a cookie the page's scripts cannot read."""
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        # Not `secure`: this is served over http on loopback, and a Secure
        # cookie would simply never be sent. Stated rather than left as an
        # oversight — behind TLS this must become True.
        secure=False,
        path="/",
    )


def _mongo_session(store: MongoAccountStore, user_id: str, response: Response) -> None:
    """Mint a session in Mongo and set the cookie. Only the hash is stored."""
    token = secrets.token_urlsafe(32)
    store.start_session(_token_hash(token), user_id, utc_now() + timedelta(hours=SESSION_HOURS))
    _set_session_cookie(response, token)


def _current(token: str | None) -> AccountResponse | None:
    """The signed-in account, or None. Expired sessions are removed as seen."""
    if not token:
        return None
    store = active_store()
    if store is not None:
        record = store.find_by_session(_token_hash(token))
        return _from_record(record) if record else None
    with optional_connection() as connection:
        if connection is None:
            return None
        with connection.cursor() as cur:
            cur.execute(
                "SELECT u.user_id, u.email, u.display_name, u.settings, u.created_at, "
                "u.last_login_at, s.expires_at FROM sessions s "
                "JOIN users u ON u.user_id = s.user_id WHERE s.token_hash = %s",
                [_token_hash(token)],
            )
            found = cur.fetchone()
            if found is None:
                return None
            if found[6] <= utc_now():
                cur.execute("DELETE FROM sessions WHERE token_hash = %s", [_token_hash(token)])
                connection.commit()
                return None
            cur.execute(
                "UPDATE sessions SET last_seen_at = now() WHERE token_hash = %s",
                [_token_hash(token)],
            )
            connection.commit()
    return _account(found[:6])


def _issue_session(connection: Any, user_id: uuid.UUID, response: Response) -> None:
    """Mint a session in Postgres, store only its hash, and set the cookie."""
    token = secrets.token_urlsafe(32)
    expires = utc_now() + timedelta(hours=SESSION_HOURS)
    with connection.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
            [_token_hash(token), str(user_id), expires],
        )
    connection.commit()
    _set_session_cookie(response, token)


#: The session cookie, as FastAPI reads it off a request.
SessionCookie = Annotated[str | None, Cookie(alias=SESSION_COOKIE)]


def current_account(session: str | None) -> AccountResponse | None:
    """The signed-in account, for other routers that need it."""
    return _current(session)


def write_settings(user_id: str, settings: dict[str, Any]) -> None:
    """Replace an account's settings blob.

    Used by the broker-key router, which owns a slice of the same document.
    Replacement rather than merge, because the caller has already merged what
    it wanted to keep — merging twice would make a deletion impossible.
    """
    store = active_store()
    if store is not None:
        if store.save_settings(user_id, settings) is None:
            raise HTTPException(status_code=503, detail="accounts need the database")
        return
    with optional_connection() as connection:
        if connection is None:
            raise HTTPException(status_code=503, detail="accounts need the database")
        with connection.cursor() as cur:
            cur.execute(
                "UPDATE users SET settings = %s WHERE user_id = %s",
                [json.dumps(settings), user_id],
            )
        connection.commit()


#: How long the account count is trusted without asking again.
#:
#: Short on purpose. This exists to collapse a burst of polls into one query,
#: not to hold an answer across a meaningful stretch of time.
USER_COUNT_TTL_SECONDS = 2.0


@dataclass
class _UserCount:
    """The last count, and when it stops being trusted."""

    deadline: float = 0.0
    value: int = 0


_count = _UserCount()


def forget_user_count() -> None:
    """Drop the cached count. Called wherever an account is created."""
    _count.deadline = 0.0


def _user_count(store: Any) -> int:
    """How many accounts exist, asked at most once every few seconds.

    **Only the count is cached, and deliberately not availability.** Both were
    at first, and it was wrong twice over. Whether the database is reachable is
    a health signal, and a health signal that reports "up" for two seconds
    after the database went down is worse than a slow one. It is also
    unnecessary: `available()` costs a round trip only when the client has to
    be built, and the client is cached now — measured at 2ms against 60ms for
    the count. There was nothing to win and a true answer to lose.

    **Nothing about the caller is cached.** Whether a particular session is
    signed in is resolved from its own cookie on every request and never stored
    here; a shared cache of that would eventually hand one session's identity
    to another, which is a security bug wearing a performance improvement's
    clothes.

    The count cannot change except through a request this module also serves,
    and that path clears the cache outright, so the two-second window only
    applies to a second browser registering an account elsewhere.
    """
    now = time.monotonic()
    if now < _count.deadline:
        return _count.value
    _count.value = store.count()
    _count.deadline = now + USER_COUNT_TTL_SECONDS
    return _count.value


def account_status(session: str | None) -> StatusResponse:
    """Who is signed in, and whether accounts mean anything on this install."""
    store = active_store()
    if store is not None:
        available = store.available()
        count = _user_count(store) if available else 0
        current = _current(session) if available else None
        return StatusResponse(
            signed_in=current is not None,
            account=current,
            accounts_available=available,
            api_token_configured=token_is_configured(),
            user_count=count,
            backend="mongodb",
        )
    with optional_connection() as connection:
        available = connection is not None
        count = 0
        if connection is not None:
            with connection.cursor() as cur:
                cur.execute("SELECT count(*) FROM users")
                row = cur.fetchone()
                count = int(row[0]) if row else 0
    current = _current(session) if available else None
    return StatusResponse(
        signed_in=current is not None,
        account=current,
        accounts_available=available,
        api_token_configured=token_is_configured(),
        user_count=count,
        backend="postgres",
    )


def create_account(request: RegisterRequest, response: Response) -> AccountResponse:
    """Create an account and sign it in."""
    # The user count is about to change, and a status poll a moment later
    # showing the old one would tell a first-time operator their account was
    # not created. Cleared before the write rather than after, so a failure
    # part-way through cannot leave a stale count behind it.
    forget_user_count()
    digest, salt = hash_password(request.password)
    store = active_store()
    if store is not None:
        if not store.available():
            raise HTTPException(
                status_code=503, detail="accounts need the database; start it and retry"
            )
        if store.find_by_email(request.email) is not None:
            raise HTTPException(status_code=409, detail="that email is already registered")
        record = store.create(
            AccountRecord(
                user_id=str(uuid.uuid4()),
                email=request.email,
                display_name=request.display_name,
                settings={},
                created_at=utc_now(),
                last_login_at=None,
                password_hash=digest,
                password_salt=salt,
                scrypt_n=SCRYPT_N,
                scrypt_r=SCRYPT_R,
                scrypt_p=SCRYPT_P,
            )
        )
        _mongo_session(store, record.user_id, response)
        return _from_record(record)
    with optional_connection() as connection:
        if connection is None:
            raise HTTPException(
                status_code=503, detail="accounts need the database; start it and retry"
            )
        with connection.cursor() as cur:
            cur.execute("SELECT 1 FROM users WHERE lower(email) = lower(%s)", [request.email])
            if cur.fetchone():
                raise HTTPException(status_code=409, detail="that email is already registered")
            cur.execute(
                "INSERT INTO users (email, display_name, password_hash, password_salt, "
                "scrypt_n, scrypt_r, scrypt_p) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "RETURNING user_id, email, display_name, settings, created_at, last_login_at",
                [request.email, request.display_name, digest, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P],
            )
            created = cur.fetchone()
        if created is None:  # pragma: no cover - RETURNING always yields
            raise HTTPException(status_code=500, detail="account was not created")
        connection.commit()
        _issue_session(connection, created[0], response)
    return _account(created)


def sign_in(request: LoginRequest, response: Response) -> AccountResponse:
    """Sign in.

    The failure is identical for an unknown email and a wrong password, and
    takes the same time: distinguishing them tells a prober which addresses
    exist, and returning early tells them by the clock instead.
    """
    denied = HTTPException(status_code=401, detail="email or password is incorrect")
    store = active_store()
    if store is not None:
        record = store.find_by_email(request.email)
        if record is None or record.password_hash is None or record.password_salt is None:
            # Hashed anyway, so an unknown address and a wrong password cost
            # the same time. Returning early here tells a prober by the clock
            # what the identical message refuses to tell them in words.
            hash_password(request.password)
            raise denied
        if not verify_password(
            request.password,
            StoredPassword(
                digest=record.password_hash,
                salt=record.password_salt,
                n=record.scrypt_n,
                r=record.scrypt_r,
                p=record.scrypt_p,
            ),
        ):
            raise denied
        store.touch_login(record.user_id)
        _mongo_session(store, record.user_id, response)
        return _from_record(record)
    with optional_connection() as connection:
        if connection is None:
            raise HTTPException(status_code=503, detail="accounts need the database")
        with connection.cursor() as cur:
            cur.execute(
                "SELECT user_id, email, display_name, settings, created_at, last_login_at, "
                "password_hash, password_salt, scrypt_n, scrypt_r, scrypt_p "
                "FROM users WHERE lower(email) = lower(%s)",
                [request.email],
            )
            found = cur.fetchone()
        if found is None:
            hash_password(request.password)
            raise denied
        if not verify_password(
            request.password,
            StoredPassword(
                digest=bytes(found[6]),
                salt=bytes(found[7]),
                n=int(found[8]),
                r=int(found[9]),
                p=int(found[10]),
            ),
        ):
            raise denied

        with connection.cursor() as cur:
            cur.execute(
                "UPDATE users SET last_login_at = now() WHERE user_id = %s", [str(found[0])]
            )
        connection.commit()
        _issue_session(connection, found[0], response)
    return _account(found[:6])


def sign_out(response: Response, session: str | None) -> dict[str, bool]:
    """Sign out, deleting the session rather than merely forgetting it."""
    store = active_store()
    if session and store is not None:
        store.end_session(_token_hash(session))
    elif session:
        with optional_connection() as connection:
            if connection is not None:
                with connection.cursor() as cur:
                    cur.execute(
                        "DELETE FROM sessions WHERE token_hash = %s", [_token_hash(session)]
                    )
                connection.commit()
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"signed_out": True}


def save_settings(request: SettingsRequest, session: str | None) -> AccountResponse:
    """Merge preferences into the account.

    Merged rather than replaced, so a console that knows about three settings
    does not erase a fourth it has never heard of.
    """
    account = _current(session)
    if account is None:
        raise HTTPException(status_code=401, detail="sign in to change settings")

    supplied = {k: v for k, v in request.model_dump().items() if v is not None}
    merged = {**account.settings, **supplied}
    store = active_store()
    if store is not None:
        record = store.save_settings(account.user_id, merged)
        if record is None:
            raise HTTPException(status_code=503, detail="accounts need the database")
        return _from_record(record)
    with optional_connection() as connection:
        if connection is None:
            raise HTTPException(status_code=503, detail="accounts need the database")
        with connection.cursor() as cur:
            cur.execute(
                "UPDATE users SET settings = %s WHERE user_id = %s "
                "RETURNING user_id, email, display_name, settings, created_at, last_login_at",
                [json.dumps(merged), account.user_id],
            )
            updated = cur.fetchone()
        connection.commit()
    if updated is None:  # pragma: no cover - the account was just read
        raise HTTPException(status_code=404, detail="account not found")
    return _account(updated)


def build_accounts_router() -> APIRouter:
    """Register, sign in, sign out, and per-operator settings."""
    router = APIRouter(prefix="/account", tags=["account"])

    @router.get("/status", response_model=StatusResponse)
    def status_(session: SessionCookie = None) -> StatusResponse:
        """Who is signed in, and whether accounts mean anything here."""
        return account_status(session)

    @router.post("/register", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
    def register(request: RegisterRequest, response: Response) -> AccountResponse:
        """Create an account and sign it in."""
        return create_account(request, response)

    @router.post("/login", response_model=AccountResponse)
    def login(request: LoginRequest, response: Response) -> AccountResponse:
        """Sign in."""
        return sign_in(request, response)

    @router.post("/logout")
    def logout(response: Response, session: SessionCookie = None) -> dict[str, bool]:
        """Sign out."""
        return sign_out(response, session)

    @router.put("/settings", response_model=AccountResponse)
    def update_settings(request: SettingsRequest, session: SessionCookie = None) -> AccountResponse:
        """Merge preferences into the account."""
        return save_settings(request, session)

    return router
