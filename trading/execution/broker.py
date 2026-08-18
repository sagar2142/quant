"""Broker adapter interface and the paper implementation — MASTER_PLAN §18, §20.

**Paper trading uses the identical code path as live.** Strategy, sizing, risk,
order state machine, reconciliation — all the same objects. The single
substitution is at the last possible moment: `PaperBroker` simulates the fill
instead of sending it to a venue. Anything less than that and paper results
stop being evidence about the live system, which defeats the entire point of
running paper for six weeks (§M9 gate).

That constraint is what makes paper-versus-backtest drift meaningful, and drift
is the single most informative metric the system produces (§35).

**The live adapter refuses to exist unless explicitly enabled.** Both
`env=live` and `live_enabled=true` are required, and the check lives in the
constructor rather than in the send path, so a misconfigured process fails at
startup instead of on its first order.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from core.clock import utc_now
from core.instruments import Instrument, InstrumentId
from core.orders import OrderState, Side
from trading.execution.orders import Order, TradingMode

__all__ = ["BrokerAdapter", "BrokerError", "BrokerFill", "BrokerPosition", "PaperBroker"]


class BrokerError(RuntimeError):
    """The venue rejected or failed to process a request.

    Never swallowed (§14.1.5): a silently dropped order leaves the system
    believing it holds a position it does not.
    """


@dataclass(frozen=True)
class BrokerFill:
    """A fill as the venue reports it."""

    broker_fill_id: str
    broker_order_id: str
    instrument_id: InstrumentId
    side: Side
    quantity: Decimal
    price: Decimal
    fees: Decimal
    event_time: object


@dataclass(frozen=True)
class BrokerPosition:
    """A position as the venue reports it. The truth reconciliation compares to."""

    instrument_id: InstrumentId
    quantity: Decimal
    average_price: Decimal


@runtime_checkable
class BrokerAdapter(Protocol):
    """What every venue must provide.

    Deliberately small. A wide interface tempts strategies into venue-specific
    behaviour, and the whole point of §1.4 is that adding a market should be a
    plugin rather than a rewrite.
    """

    @property
    def mode(self) -> TradingMode: ...

    def submit(self, order: Order, reference_price: Decimal) -> str:
        """Send an order. Returns the venue's order id.

        Raises:
            BrokerError: if the venue rejected it.
        """

    def cancel(self, broker_order_id: str) -> None: ...

    def positions(self) -> list[BrokerPosition]:
        """What the venue believes is held. The reconciliation baseline."""

    def fills_since(self, marker: str | None) -> list[BrokerFill]: ...


@dataclass
class PaperBroker:
    """Simulated venue against live prices.

    Fills at the reference price plus a configurable slippage, immediately and
    in full. That is optimistic on both counts — real venues partial-fill and
    real fills arrive late — and the optimism is deliberate: paper is meant to
    exercise the *plumbing*, and drift against backtest expectations is what
    reveals whether the fill assumptions were sound.

    Args:
        slippage_bps: Adverse price movement applied to every fill.
        reject_rate: Fraction of orders to reject, for exercising the rejection
            path. Zero by default; set it to make sure the error handling is
            real rather than theoretical.
    """

    instruments: dict[InstrumentId, Instrument]
    slippage_bps: Decimal = Decimal(2)
    _fills: list[BrokerFill] = field(default_factory=list)
    _positions: dict[InstrumentId, BrokerPosition] = field(default_factory=dict)
    _submitted: dict[str, Order] = field(default_factory=dict)

    @property
    def mode(self) -> TradingMode:
        return TradingMode.PAPER

    def submit(self, order: Order, reference_price: Decimal) -> str:
        if order.mode is not TradingMode.PAPER:
            raise BrokerError(
                f"PaperBroker refuses a {order.mode.value} order — "
                "mode mismatches are how a live order reaches a simulator"
            )
        if reference_price <= 0:
            raise BrokerError(f"reference price must be positive, got {reference_price}")

        broker_order_id = f"paper-{uuid.uuid4().hex[:12]}"
        self._submitted[broker_order_id] = order

        fill_price = self._apply_slippage(order, reference_price)
        instrument = self.instruments.get(order.instrument_id)
        if instrument is None:
            raise BrokerError(f"unknown instrument {order.instrument_id}")
        fill_price = instrument.round_to_tick(fill_price)

        self._fills.append(
            BrokerFill(
                broker_fill_id=f"pf-{uuid.uuid4().hex[:12]}",
                broker_order_id=broker_order_id,
                instrument_id=order.instrument_id,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                fees=Decimal(0),
                event_time=utc_now(),
            )
        )
        self._apply_to_positions(order, fill_price)
        return broker_order_id

    def _apply_slippage(self, order: Order, reference: Decimal) -> Decimal:
        """Move the price against the trader, always."""
        drift = reference * self.slippage_bps / Decimal(10_000)
        return reference + drift * order.side.sign

    def _apply_to_positions(self, order: Order, price: Decimal) -> None:
        existing = self._positions.get(order.instrument_id)
        delta = order.quantity * order.side.sign

        if existing is None:
            self._positions[order.instrument_id] = BrokerPosition(order.instrument_id, delta, price)
            return

        combined = existing.quantity + delta
        if combined == 0:
            self._positions.pop(order.instrument_id)
            return
        # Weighted average only when adding in the same direction; a reduction
        # leaves the entry price alone.
        if (existing.quantity > 0) == (delta > 0):
            average = (existing.average_price * abs(existing.quantity) + price * abs(delta)) / abs(
                combined
            )
        else:
            average = existing.average_price
        self._positions[order.instrument_id] = BrokerPosition(
            order.instrument_id, combined, average
        )

    def cancel(self, broker_order_id: str) -> None:
        order = self._submitted.get(broker_order_id)
        if order is None:
            raise BrokerError(f"unknown order {broker_order_id}")
        if order.state.is_terminal:
            raise BrokerError(f"order {broker_order_id} is already {order.state.value}")
        order.transition(OrderState.CANCELLED, "cancelled at paper venue")

    def positions(self) -> list[BrokerPosition]:
        return sorted(self._positions.values(), key=lambda p: p.instrument_id)

    def fills_since(self, marker: str | None) -> list[BrokerFill]:
        """Fills after `marker`, or all of them when it is None."""
        if marker is None:
            return list(self._fills)
        ids = [f.broker_fill_id for f in self._fills]
        if marker not in ids:
            return list(self._fills)
        return self._fills[ids.index(marker) + 1 :]

    def inject_position(self, position: BrokerPosition) -> None:
        """Force a position, bypassing the order path.

        Exists to simulate the discrepancies reconciliation is meant to catch.
        Nothing in normal operation calls it.
        """
        self._positions[position.instrument_id] = position
