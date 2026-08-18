"""Zerodha Kite live broker adapter — MASTER_PLAN §20, §21.

The only code in this system that can move real money, and it is written to be
hard to invoke by accident.

**Four independent guards, all of which must pass.** They are deliberately
redundant, because the failure they prevent is unrecoverable:

    1. `settings.require_live_permission()` — env=live AND live_enabled=true
    2. constructor rejects any order whose mode is not LIVE
    3. `submit` re-checks permission on every single order, not just at startup
    4. a per-session notional budget, exhausted rather than reset

Guard 3 looks redundant against guard 1 and is not: a long-running process can
have its configuration change underneath it, and the cost of one extra check
per order is nothing against the cost of one unintended order.

**No `cancel_all`, no `close_all_positions`.** Convenience methods on a live
adapter are how a typo liquidates a book. Flattening is a runbook procedure
executed deliberately, not a function call available to anything holding a
reference.

**Every rejection is loud.** A silently dropped live order leaves the system
believing it holds a position it does not, which is exactly the state
reconciliation exists to catch and exactly the state you do not want to be in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

import httpx

from core.clock import utc_now
from core.config import settings
from core.instruments import Instrument, InstrumentId
from core.orders import OrderType, Side
from core.secrets import BrokerCredentials
from trading.execution.broker import BrokerError, BrokerFill, BrokerPosition
from trading.execution.orders import Order, TradingMode

__all__ = ["KITE_API_BASE", "KiteBroker"]

logger = logging.getLogger(__name__)

KITE_API_BASE = "https://api.kite.trade"
REQUEST_TIMEOUT_SECONDS = 15.0

#: Kite's own vocabulary. Mapping is explicit so a rename upstream is a
#: compile-time concern rather than a silently wrong order type.
_ORDER_TYPES: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP: "SL-M",
    OrderType.STOP_LIMIT: "SL",
}

_SIDES: dict[Side, str] = {Side.BUY: "BUY", Side.SELL: "SELL"}


@dataclass
class KiteBroker:
    """Live NSE trading through Kite Connect.

    Args:
        credentials: Loaded via `core.secrets.load_broker_credentials`, which
            itself refuses to load outside a live-enabled environment.
        instruments: Instrument master, for tick and lot validation.
        session_notional_budget: Total notional this process may transact
            before refusing everything. A circuit breaker against a runaway
            loop that passes every per-order check individually. Exhausted
            rather than reset — a new budget requires a new process, which
            requires a human.
        client: Injected for testing. No test in this repo touches the network.
    """

    credentials: BrokerCredentials
    instruments: dict[InstrumentId, Instrument]
    session_notional_budget: Decimal = Decimal(500_000)
    client: httpx.Client | None = None
    _spent: Decimal = field(default=Decimal(0), init=False)

    def __post_init__(self) -> None:
        # Guard 1: fails at construction, so a misconfigured process dies at
        # startup rather than on its first order.
        settings.require_live_permission()
        if not self.credentials.access_token.is_set:
            raise BrokerError("Kite access token is empty; complete the daily login flow")
        logger.warning(
            "KiteBroker constructed — LIVE trading is enabled, budget %s",
            self.session_notional_budget,
        )

    @property
    def mode(self) -> TradingMode:
        return TradingMode.LIVE

    @property
    def remaining_budget(self) -> Decimal:
        return max(Decimal(0), self.session_notional_budget - self._spent)

    def _headers(self) -> dict[str, str]:
        return {
            "X-Kite-Version": "3",
            "Authorization": (
                f"token {self.credentials.api_key.reveal()}:"
                f"{self.credentials.access_token.reveal()}"
            ),
        }

    def _http(self) -> httpx.Client:
        return self.client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    def submit(self, order: Order, reference_price: Decimal) -> str:
        """Place a live order.

        Raises:
            BrokerError: on any guard failure or venue rejection. Never returns
                a sentinel — a caller that cannot tell success from failure
                will assume success.
        """
        # Guard 3: re-checked per order, because configuration can change
        # under a long-running process.
        settings.require_live_permission()

        # Guard 2: a mode mismatch is how a paper order reaches a real venue.
        if order.mode is not TradingMode.LIVE:
            raise BrokerError(
                f"KiteBroker refuses a {order.mode.value} order — "
                "only LIVE orders may reach a real venue"
            )

        instrument = self.instruments.get(order.instrument_id)
        if instrument is None:
            raise BrokerError(f"unknown instrument {order.instrument_id}")

        notional = order.quantity * reference_price * instrument.multiplier

        # Guard 4: the session budget.
        if notional > self.remaining_budget:
            raise BrokerError(
                f"session notional budget exhausted: order {notional} exceeds "
                f"remaining {self.remaining_budget}. Restart the process "
                "deliberately to grant a new budget."
            )

        self._validate_lot_size(order, instrument)

        payload = {
            "tradingsymbol": instrument.symbol,
            "exchange": instrument.exchange.value,
            "transaction_type": _SIDES[order.side],
            "order_type": _ORDER_TYPES[order.order_type],
            "quantity": str(int(order.quantity)),
            "product": "CNC",
            "validity": "DAY",
            # Kite rejects a duplicate tag, which is a second line of defence
            # behind our own database UNIQUE constraint (§19).
            "tag": order.idempotency_key[:20],
        }
        if order.limit_price is not None:
            payload["price"] = str(instrument.round_to_tick(order.limit_price))
        if order.stop_price is not None:
            payload["trigger_price"] = str(instrument.round_to_tick(order.stop_price))

        broker_order_id = self._post_order(payload, order)
        self._spent += notional
        logger.warning(
            "LIVE order placed: %s %s %s @ ~%s (budget left %s)",
            order.side.value,
            order.quantity,
            instrument.symbol,
            reference_price,
            self.remaining_budget,
        )
        return broker_order_id

    def _validate_lot_size(self, order: Order, instrument: Instrument) -> None:
        if instrument.lot_size > 1 and order.quantity % instrument.lot_size != 0:
            raise BrokerError(
                f"quantity {order.quantity} is not a multiple of lot size "
                f"{instrument.lot_size} for {instrument.symbol}"
            )

    def _post_order(self, payload: dict[str, str], order: Order) -> str:
        client = self._http()
        try:
            response = client.post(
                f"{KITE_API_BASE}/orders/regular",
                headers=self._headers(),
                data=payload,
            )
        except httpx.HTTPError as exc:
            # The order may or may not have reached the venue. This is exactly
            # the UNKNOWN case (§19): the caller must mark it and reconcile,
            # never assume either outcome.
            raise BrokerError(
                f"network failure submitting order {order.order_id}; state is "
                f"UNKNOWN and must be reconciled, not assumed: {exc}"
            ) from exc
        finally:
            if self.client is None:
                client.close()

        if response.status_code != httpx.codes.OK:
            raise BrokerError(
                f"Kite rejected order {order.order_id}: "
                f"HTTP {response.status_code} {response.text[:300]}"
            )

        body = response.json()
        order_id = body.get("data", {}).get("order_id")
        if not order_id:
            raise BrokerError(f"Kite returned no order_id: {body}")
        return str(order_id)

    def cancel(self, broker_order_id: str) -> None:
        settings.require_live_permission()
        client = self._http()
        try:
            response = client.delete(
                f"{KITE_API_BASE}/orders/regular/{broker_order_id}",
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise BrokerError(f"network failure cancelling {broker_order_id}: {exc}") from exc
        finally:
            if self.client is None:
                client.close()

        if response.status_code != httpx.codes.OK:
            raise BrokerError(f"cancel of {broker_order_id} failed: HTTP {response.status_code}")

    def positions(self) -> list[BrokerPosition]:
        """What Kite believes is held. The reconciliation baseline (§9)."""
        client = self._http()
        try:
            response = client.get(f"{KITE_API_BASE}/portfolio/positions", headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"could not fetch positions: {exc}") from exc
        finally:
            if self.client is None:
                client.close()

        if response.status_code != httpx.codes.OK:
            raise BrokerError(f"positions fetch failed: HTTP {response.status_code}")

        net = response.json().get("data", {}).get("net", [])
        return [
            BrokerPosition(
                instrument_id=InstrumentId(f"NSE:{row['tradingsymbol']}"),
                quantity=Decimal(str(row["quantity"])),
                average_price=Decimal(str(row["average_price"])),
            )
            for row in net
            if Decimal(str(row["quantity"])) != 0
        ]

    def fills_since(self, marker: str | None) -> list[BrokerFill]:
        """Executed trades. `marker` is the last seen trade id."""
        client = self._http()
        try:
            response = client.get(f"{KITE_API_BASE}/trades", headers=self._headers())
        except httpx.HTTPError as exc:
            raise BrokerError(f"could not fetch trades: {exc}") from exc
        finally:
            if self.client is None:
                client.close()

        if response.status_code != httpx.codes.OK:
            raise BrokerError(f"trades fetch failed: HTTP {response.status_code}")

        trades = response.json().get("data", [])
        fills = [
            BrokerFill(
                broker_fill_id=str(row["trade_id"]),
                broker_order_id=str(row["order_id"]),
                instrument_id=InstrumentId(f"NSE:{row['tradingsymbol']}"),
                side=Side.BUY if row["transaction_type"] == "BUY" else Side.SELL,
                quantity=Decimal(str(row["quantity"])),
                price=Decimal(str(row["average_price"])),
                fees=Decimal(0),  # itemised separately on the contract note
                event_time=utc_now(),
            )
            for row in trades
        ]
        if marker is None:
            return fills
        ids = [f.broker_fill_id for f in fills]
        return fills[ids.index(marker) + 1 :] if marker in ids else fills
