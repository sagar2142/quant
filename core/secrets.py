"""Credential handling — MASTER_PLAN §23, §13.7, Appendix B.

**Production credentials never enter AI research.** That is one of the plan's
core design principles, and it is enforced structurally: the import-linter
contract `ai-is-sandboxed` forbids anything under `ai/` from importing this
module at all. A generated strategy cannot reach a broker key because it cannot
reach the code that loads one.

**Research and trading credentials are separate objects, not separate fields.**
A read-only market-data key and a key that can place orders are different kinds
of thing, and putting them in one bag means every consumer of the first also
holds the second.

**Secrets never appear in `repr`.** A credential that leaks into a log line, a
traceback or a debugger session has leaked. `SecretValue` renders as `***` in
every context except an explicit `.reveal()`, which is deliberately awkward to
type and easy to grep for.

**Live credentials refuse to load unless live trading is explicitly enabled.**
Both `env=live` and `live_enabled=true`, checked at load time rather than at
send time, so a misconfigured process fails at startup rather than on its first
order.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.config import Environment, settings

__all__ = [
    "BrokerCredentials",
    "MarketDataCredentials",
    "MissingCredentialError",
    "SecretValue",
    "load_broker_credentials",
    "load_market_data_credentials",
]


class MissingCredentialError(RuntimeError):
    """A required credential is absent.

    Raised rather than falling back to an empty string (§14.1.5): an empty API
    key produces a confusing authentication failure at the venue instead of a
    clear configuration error at startup.
    """

    def __init__(self, name: str, purpose: str) -> None:
        super().__init__(
            f"{name} is not set. Required for {purpose}. "
            "Copy .env.example to .env and fill it in; never commit .env."
        )


class SecretValue:
    """A string that does not print itself.

    Wrapping is not paranoia. Credentials leak through the mundane paths —
    an f-string in a log, a dataclass repr in a traceback, a notebook cell that
    echoes a config object. Every one of those renders `***` here.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """The actual secret. Grep for this to audit every use."""
        return self._value

    @property
    def is_set(self) -> bool:
        return bool(self._value.strip())

    def __repr__(self) -> str:
        return "SecretValue(***)"

    def __str__(self) -> str:
        return "***"

    def __bool__(self) -> bool:
        return self.is_set

    def __eq__(self, other: object) -> bool:
        # Constant-ish comparison against another SecretValue only. Comparing a
        # secret to a plain string is almost always a test shortcut that would
        # not survive review.
        if not isinstance(other, SecretValue):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)


@dataclass(frozen=True)
class MarketDataCredentials:
    """Read-only market data access.

    Safe for research processes: it can read prices and cannot place an order.
    """

    provider: str
    api_key: SecretValue
    api_secret: SecretValue

    def __repr__(self) -> str:
        return f"MarketDataCredentials(provider={self.provider!r}, api_key=***)"


@dataclass(frozen=True)
class BrokerCredentials:
    """Credentials that can move money.

    Loaded only when live trading is explicitly enabled, and never by anything
    under `ai/` or `quant/` (§3.2).
    """

    broker: str
    api_key: SecretValue
    api_secret: SecretValue
    access_token: SecretValue

    def __repr__(self) -> str:
        return f"BrokerCredentials(broker={self.broker!r}, api_key=***, access_token=***)"


def _require(name: str, purpose: str) -> SecretValue:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingCredentialError(name, purpose)
    return SecretValue(value)


def load_market_data_credentials(provider: str = "alpaca") -> MarketDataCredentials:
    """Load read-only market data credentials.

    Alpaca paper keys grant market data and paper order entry — no real money —
    which is why they are classified as market-data rather than broker
    credentials (§0.0). Kite is read-only here; its order-placing form is
    `load_broker_credentials`.
    """
    if provider == "alpaca":
        return MarketDataCredentials(
            provider=provider,
            api_key=_require("ALPACA_API_KEY", "US market data and paper trading"),
            api_secret=_require("ALPACA_API_SECRET", "US market data and paper trading"),
        )
    if provider == "kite":
        return MarketDataCredentials(
            provider=provider,
            api_key=_require("KITE_API_KEY", "NSE market data"),
            api_secret=_require("KITE_API_SECRET", "NSE market data"),
        )
    raise ValueError(f"unknown market data provider: {provider}")


def load_broker_credentials(broker: str = "kite") -> BrokerCredentials:
    """Load credentials that can place real orders.

    Raises:
        PermissionError: unless the environment is LIVE *and* live trading was
            explicitly enabled. Both, not either — checked here at load time so
            a misconfigured process dies at startup rather than mid-session.
        MissingCredentialError: if a required value is absent.
    """
    settings.require_live_permission()

    if broker != "kite":
        raise ValueError(f"unknown broker: {broker}")

    return BrokerCredentials(
        broker=broker,
        api_key=_require("KITE_API_KEY", "live order placement"),
        api_secret=_require("KITE_API_SECRET", "live order placement"),
        access_token=_require("KITE_ACCESS_TOKEN", "live order placement"),
    )


def assert_not_live(context: str) -> None:
    """Guard a code path that must never run against real money.

    Used by research and paper components as a second line of defence behind
    the import boundaries.
    """
    if settings.env is Environment.LIVE and settings.live_enabled:
        raise PermissionError(
            f"{context} must not run with live trading enabled. "
            "This code path is for research and paper only."
        )
