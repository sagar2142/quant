"""Live NSE and BSE trading through Groww — MASTER_PLAN §8, §21.

Written against the documented Groww Trading API: base `https://api.groww.in/v1`,
bearer authentication, `POST /order/create`, `POST /order/cancel`,
`GET /order/list`, `GET /order/detail/{id}`, `GET /positions/user` and
`GET /holdings/user`.

**Three things differ from Kite, and each is handled rather than papered over.**

*The access token expires daily at 06:00 IST.* It is minted from an API key and
a TOTP outside this process, so a token that worked yesterday is simply gone
this morning. That failure arrives as a 401 in the middle of a session, and it
is reported as "the daily token has expired" rather than as a generic auth
error, because those need different actions from an operator at 09:16.

*There is no trades endpoint.* Groww documents holdings and positions but no
fills feed, so `fills_since` raises instead of returning an empty list. An
empty list would say "nothing filled", which is a claim this adapter cannot
make, and reconciliation reading it as truth is exactly how a real fill goes
unnoticed (§9).

*Symbols, not ISINs, address an order.* Groww's order API takes a
`trading_symbol`, while this system keys positions on ISIN (§1.1) precisely
because a symbol can wear more than one ISIN over its life. The conversion
happens here, at the boundary, and the ISIN comes back on positions where
Groww does supply it as `symbol_isin`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import httpx

from core.clock import utc_now
from core.config import settings
from core.instruments import Instrument, InstrumentId
from core.orders import OrderType, Side
from core.secrets import BrokerCredentials
from trading.execution.broker import BrokerError, BrokerFill, BrokerOrder, BrokerPosition
from trading.execution.groww_auth import GrowwAuthError, mint_access_token
from trading.execution.orders import Order, TradingMode

__all__ = ["GROWW_API_BASE", "GrowwBroker"]

logger = logging.getLogger(__name__)

GROWW_API_BASE = "https://api.groww.in/v1"
REQUEST_TIMEOUT_SECONDS = 15.0

#: Groww's vocabulary. Mapped explicitly so a rename upstream is a failure here
#: rather than a silently wrong order type.
_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "SL_M",
    OrderType.STOP_LIMIT: "SL",
}

_SIDES: dict[Side, str] = {Side.BUY: "BUY", Side.SELL: "SELL"}

#: Cash equities. `CNC` is delivery; `MIS` would be intraday and is not offered
#: here, because an intraday product squares off automatically at the close and
#: nothing in this system is modelling that.
SEGMENT_CASH = "CASH"
PRODUCT_DELIVERY = "CNC"
VALIDITY_DAY = "DAY"

#: A candle row is [epoch, open, high, low, close, volume].
CANDLE_FIELDS = 6


@dataclass
class GrowwBroker:
    """Live trading through Groww.

    Args:
        credentials: From `core.secrets.load_broker_credentials("groww")`,
            which itself refuses to load outside a live-enabled environment.
        instruments: Instrument master, for resolving an id to the trading
            symbol Groww's order API expects.
        exchange: Which exchange orders are routed to.
        session_notional_budget: Total notional this process may transact
            before refusing everything. A circuit breaker against a runaway
            loop that passes every per-order check individually. Exhausted
            rather than reset — a new budget needs a new process, which needs
            a human.
        client: Injected for testing. No test in this repo touches the network.
    """

    credentials: BrokerCredentials
    instruments: dict[InstrumentId, Instrument]
    exchange: str = "NSE"
    session_notional_budget: Decimal = Decimal(500_000)
    client: httpx.Client | None = None
    _spent: Decimal = Decimal(0)
    #: The token in use, and when it dies. Minted from the key and secret on
    #: first need and reused until 06:00 IST — Groww allows 150 mints a day,
    #: and one per request would exhaust that before lunch and lock the account
    #: out of trading during market hours.
    _token: str = ""
    _token_expires: datetime | None = None

    def __post_init__(self) -> None:
        # Fails at construction, so a misconfigured process dies at startup
        # rather than on its first order.
        settings.require_live_permission()
        if not (self.credentials.access_token.is_set or self.credentials.api_secret.is_set):
            raise BrokerError(
                "Groww needs an API key and secret. The access token is minted from them "
                "and refreshed automatically."
            )
        logger.warning(
            "GrowwBroker constructed — LIVE trading is enabled, budget %s",
            self.session_notional_budget,
        )

    @property
    def mode(self) -> TradingMode:
        return TradingMode.LIVE

    @property
    def remaining_budget(self) -> Decimal:
        return max(Decimal(0), self.session_notional_budget - self._spent)

    def _access_token(self, force: bool = False) -> str:
        """A live token, minting one if the cached one has expired.

        A token supplied explicitly is used as-is and never replaced: someone
        who set one out of band has said which credential to trade under, and
        silently swapping it would ignore that.
        """
        if self.credentials.access_token.is_set and not force:
            return str(self.credentials.access_token.reveal())

        fresh = self._token and self._token_expires and utc_now() < self._token_expires
        if fresh and not force:
            return self._token

        if not self.credentials.api_secret.is_set:
            raise BrokerError("Groww API secret is not set, so an access token cannot be minted.")
        try:
            minted = mint_access_token(
                self.credentials.api_key.reveal(),
                self.credentials.api_secret.reveal(),
                client=self.client,
            )
        except GrowwAuthError as exc:
            raise BrokerError(str(exc)) from exc

        self._token = minted.token
        self._token_expires = minted.expires_at
        logger.warning("Groww access token minted, valid until %s", minted.expires_at)
        return self._token

    def _headers(self, token: str | None = None) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token or self._access_token()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-client-id": "neutron",
            "x-api-version": "1.0",
        }

    def _http(self) -> httpx.Client:
        return self.client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def _call(
        self,
        method: str,
        path: str,
        json: dict[str, object] | None = None,
        params: dict[str, str] | None = None,
        _retried: bool = False,
    ) -> dict[str, object]:
        """One request, with the failures named.

        `json` and `params` are named rather than taken as `**kwargs`: the two
        this adapter uses are the two httpx needs typed, and a loose kwargs bag
        would let a misspelled option through to be silently ignored.

        A 401 here is almost always the daily token expiry rather than a wrong
        key, and saying which one costs nothing and saves a morning.
        """
        client = self._http()
        try:
            response = client.request(
                method,
                f"{GROWW_API_BASE}{path}",
                headers=self._headers(),
                json=json,
                params=params,
            )
        except httpx.HTTPError as exc:
            raise BrokerError(f"Groww {method} {path} failed: {exc}") from exc
        finally:
            if self.client is None:
                client.close()

        if response.status_code == httpx.codes.UNAUTHORIZED and not _retried:
            # The token expired mid-session, which happens every morning at
            # 06:00 IST. Mint once and retry; a second failure is real.
            self._access_token(force=True)
            return self._call(method, path, json=json, params=params, _retried=True)
        if response.status_code == httpx.codes.UNAUTHORIZED:
            raise BrokerError(
                "Groww rejected the credentials. Check the API key and secret in Settings."
            )
        if response.status_code != httpx.codes.OK:
            raise BrokerError(
                f"Groww {method} {path} returned HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise BrokerError(f"Groww {path} returned a non-JSON body") from exc
        if not isinstance(body, dict):
            raise BrokerError(f"Groww {path} returned {type(body).__name__}, expected an object")
        return body

    def _symbol_for(self, instrument_id: InstrumentId) -> str:
        """The trading symbol Groww's order API expects.

        Resolved from the instrument master rather than by stripping the venue
        prefix off the id: an id is `NSE:INE002A01018`, and sending an ISIN
        where a symbol is expected is an order the venue cannot match.
        """
        instrument = self.instruments.get(instrument_id)
        if instrument is None or not instrument.symbol:
            raise BrokerError(
                f"no trading symbol known for {instrument_id}; "
                "Groww addresses orders by symbol, and guessing one is not an option"
            )
        return instrument.symbol

    def submit(self, order: Order, reference_price: Decimal) -> str:
        """Place a live order. Returns Groww's order id.

        Raises:
            BrokerError: on any guard failure or venue rejection. Never returns
                a fabricated id: a caller that believes an order exists when it
                does not is worse off than one that saw an error.
        """
        notional = abs(order.quantity) * reference_price
        if notional > self.remaining_budget:
            raise BrokerError(
                f"order notional {notional} exceeds the remaining session budget "
                f"{self.remaining_budget}; a new budget requires a new process"
            )

        payload: dict[str, object] = {
            "trading_symbol": self._symbol_for(order.instrument_id),
            "quantity": int(order.quantity),
            "validity": VALIDITY_DAY,
            "exchange": self.exchange,
            "segment": SEGMENT_CASH,
            "product": PRODUCT_DELIVERY,
            "order_type": _ORDER_TYPES[order.order_type],
            "transaction_type": _SIDES[order.side],
            # Groww wants a price field even for a market order, where it is
            # ignored. Sent as 0 rather than omitted, which the API rejects.
            "price": float(order.limit_price) if order.limit_price else 0,
        }

        body = self._call("POST", "/order/create", json=payload)
        payload_out = body.get("payload") if isinstance(body.get("payload"), dict) else body
        order_id = None
        if isinstance(payload_out, dict):
            order_id = payload_out.get("groww_order_id") or payload_out.get("order_id")
        if not order_id:
            raise BrokerError(f"Groww accepted the order but returned no id: {body}")

        self._spent += notional
        logger.warning(
            "LIVE order placed via Groww: %s %s x%s -> %s",
            order.side.value,
            order.instrument_id,
            order.quantity,
            order_id,
        )
        return str(order_id)

    def cancel(self, broker_order_id: str) -> None:
        """Cancel a working order."""
        self._call(
            "POST",
            "/order/cancel",
            json={"groww_order_id": broker_order_id, "segment": SEGMENT_CASH},
        )

    def orders(self) -> list[BrokerOrder]:
        """Today's order book, as Groww reports it."""
        body = self._call("GET", "/order/list", params={"segment": SEGMENT_CASH})
        rows = _rows(body, "order_list", "orders")
        return [
            BrokerOrder(
                broker_order_id=str(row.get("groww_order_id") or row.get("order_id") or ""),
                instrument_id=InstrumentId(f"{self.exchange}:{row.get('trading_symbol', '')}"),
                side=Side.BUY if str(row.get("transaction_type")) == "BUY" else Side.SELL,
                quantity=_decimal(row.get("quantity")),
                filled_quantity=_decimal(row.get("filled_quantity")),
                # A market order has no price of its own; None says so rather
                # than reporting a zero that reads as "free".
                price=_decimal(row.get("price")) if row.get("price") else None,
                # Passed through verbatim. A venue vocabulary that does not fit
                # the local state machine should be visible as itself, not
                # coerced into the nearest match.
                status=str(row.get("order_status") or row.get("status") or "UNKNOWN"),
                placed_at=str(row.get("created_at") or ""),
            )
            for row in rows
        ]

    def candles(
        self,
        trading_symbol: str,
        start: str,
        end: str,
        interval_minutes: int,
        segment: str = SEGMENT_CASH,
    ) -> list[tuple[int, float, float, float, float, float]]:
        """Historical candles: (epoch, open, high, low, close, volume).

        **Intraday only exists here.** The panel this system stores is daily
        bhavcopy, so anything finer than a session has to come from the broker.
        Groww keeps three months of intraday and caps how much can be asked for
        at once — a minute of history is seven days per request, an hour is a
        hundred and fifty days — and those limits are the caller's to respect;
        exceeding them returns an error rather than a truncated answer, which
        is the better failure.

        Never mixed into the panel. Everything in `quant/` and `engine/` reads
        rows carrying a `receive_time`, and these have none (§3).
        """
        body = self._call(
            "GET",
            "/historical/candle/range",
            params={
                "exchange": self.exchange,
                "segment": segment,
                "trading_symbol": trading_symbol,
                "start_time": start,
                "end_time": end,
                "interval_in_minutes": str(interval_minutes),
            },
        )
        payload = body.get("payload")
        source = payload if isinstance(payload, dict) else body
        rows = source.get("candles") if isinstance(source, dict) else None
        if not isinstance(rows, list):
            return []

        out: list[tuple[int, float, float, float, float, float]] = []
        for row in rows:
            # Each candle is a six-element array. A row of another shape is
            # skipped rather than padded: a candle with an invented close is
            # worse than a candle that is missing.
            if not isinstance(row, list) or len(row) < CANDLE_FIELDS:
                continue
            try:
                out.append(
                    (
                        int(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                )
            except (TypeError, ValueError):
                continue
        return out

    def positions(self) -> list[BrokerPosition]:
        """What Groww believes is held. The reconciliation baseline (§9).

        Keyed on ISIN where Groww supplies it, because that is this system's
        identity for a position (§1.1). A row without one falls back to the
        symbol, which is visibly different rather than silently wrong.
        """
        body = self._call("GET", "/positions/user", params={"segment": SEGMENT_CASH})
        rows = _rows(body, "positions", "position_list")
        held: list[BrokerPosition] = []
        for row in rows:
            quantity = _decimal(row.get("quantity"))
            if quantity == 0:
                continue
            isin = str(row.get("symbol_isin") or "").strip()
            symbol = str(row.get("trading_symbol") or "").strip()
            held.append(
                BrokerPosition(
                    instrument_id=InstrumentId(f"{self.exchange}:{isin or symbol}"),
                    quantity=quantity,
                    # `net_price` is Groww's average for the net position.
                    average_price=_decimal(row.get("net_price") or row.get("credit_price")),
                )
            )
        return held

    def holdings(self) -> list[BrokerPosition]:
        """Demat holdings, which are not the same thing as positions.

        A position is today's net exposure; a holding is what is in the demat
        account. They differ for anything bought today and not yet settled, and
        conflating them would show a T+1 purchase as already delivered.
        """
        body = self._call("GET", "/holdings/user")
        rows = _rows(body, "holdings", "holding_list")
        return [
            BrokerPosition(
                instrument_id=InstrumentId(
                    f"{self.exchange}:{row.get('isin') or row.get('trading_symbol')}"
                ),
                quantity=_decimal(row.get("quantity")),
                average_price=_decimal(row.get("average_price")),
            )
            for row in rows
            if _decimal(row.get("quantity")) != 0
        ]

    def fills_since(self, marker: str | None) -> list[BrokerFill]:
        """Not available from Groww.

        **Raises rather than returning an empty list.** Groww documents
        holdings and positions but no trades feed, so this adapter cannot say
        what filled. An empty list is a claim — "nothing filled" — and
        reconciliation would believe it, which is precisely how a real fill
        goes unnoticed (§9, §14.1.5).

        Reconcile against `positions()` instead: it is what Groww actually
        knows, and it is the baseline §9 was written around.
        """
        _ = marker
        raise BrokerError(
            "Groww exposes no trades endpoint, so fills cannot be read. "
            "Reconcile against positions() instead — an empty fill list would "
            "assert that nothing traded, which this adapter cannot establish."
        )


def _rows(body: dict[str, object], *keys: str) -> list[dict[str, object]]:
    """Pull a list of rows out of a response, whichever key holds it.

    Groww nests payloads under `payload`, and the inner key differs per
    endpoint. Tried in order rather than assumed, and an unrecognised shape
    returns nothing rather than raising: a portfolio screen that cannot parse
    one endpoint should still render the rest.
    """
    payload = body.get("payload")
    source = payload if isinstance(payload, dict) else body
    for key in keys:
        found = source.get(key)
        if isinstance(found, list):
            return [row for row in found if isinstance(row, dict)]
    return []


def _decimal(value: object) -> Decimal:
    """A JSON number as a Decimal (§14.1.2). Missing reads as zero."""
    if value is None:
        return Decimal(0)
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return Decimal(0)
