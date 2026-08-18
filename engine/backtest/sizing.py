"""Order sizing and funding — MASTER_PLAN §17.

Turning *"I want 3% of NAV in this name"* into *"buy 47 shares"* is a distinct
responsibility from replaying bars, and it is where two easily-missed rules
live:

**Lot sizes are real.** NSE derivatives trade in lots; a target of 1.4 lots is
1 lot. Rounding at sizing time rather than at fill time keeps the backtest
honest about what could actually have been placed.

**Sells are planned before buys, and buys are sized against what remains.**
That is what a portfolio manager does, and it matters more than it looks: a
fully-invested book whose equity has grown wants to buy more on *every single
bar*, and without this the engine emits — then rejects — an unfundable order
every time. An order the account cannot fund was never an order, and counting
it as a rejection buries real rejections in noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from core.instruments import Instrument, InstrumentId
from engine.accounting import Portfolio

__all__ = ["OrderPlanner", "SizingConfig"]


@dataclass(frozen=True)
class SizingConfig:
    """Turnover and funding controls."""

    #: Skip rebalancing a name whose target differs from its holding by less
    #: than this fraction of NAV. Turnover is the most reliable way to lose
    #: money to costs (§7.1).
    rebalance_threshold: Decimal = Decimal("0.005")
    #: Below this, an order is not worth its fixed costs — DP charges alone
    #: would dominate (§7.1).
    min_order_value: Decimal = Decimal(500)
    #: Cash held back when sizing buys, covering fees and the gap between the
    #: decision price and the fill price.
    cost_headroom: Decimal = Decimal("0.005")


class OrderPlanner:
    """Converts target weights into fundable, signed quantity deltas."""

    def __init__(
        self,
        instruments: dict[InstrumentId, Instrument],
        config: SizingConfig | None = None,
    ) -> None:
        self.instruments = instruments
        self.config = config or SizingConfig()

    def plan(
        self,
        portfolio: Portfolio,
        weights: dict[InstrumentId, Decimal],
        marks: dict[InstrumentId, Decimal],
        equity: Decimal,
    ) -> list[tuple[InstrumentId, Decimal]]:
        """Signed quantity deltas the account can actually execute.

        Held positions are unioned into the target set so that names the
        strategy has dropped get closed rather than silently retained.
        """
        planned: list[tuple[InstrumentId, Decimal]] = []

        for instrument_id in sorted(set(weights) | set(portfolio.open_positions())):
            price = marks.get(instrument_id)
            instrument = self.instruments.get(instrument_id)
            if price is None or instrument is None or price <= 0:
                continue

            target_qty = self._target_quantity(
                instrument, equity * weights.get(instrument_id, Decimal(0)), price
            )
            delta = target_qty - portfolio.position(instrument_id).quantity
            if delta == 0:
                continue

            trade_value = abs(delta) * price * instrument.multiplier
            if trade_value < self.config.min_order_value:
                continue
            if equity > 0 and trade_value / equity < self.config.rebalance_threshold:
                continue

            planned.append((instrument_id, delta))

        return self._fund(portfolio, planned, marks)

    @staticmethod
    def _target_quantity(instrument: Instrument, target_value: Decimal, price: Decimal) -> Decimal:
        """Value to quantity, rounded to the venue's lot size.

        **A cash equity's lot is one share, not "no lot".** Skipping the
        rounding when lot_size is 1 produces targets like 148.0198 shares —
        an order no exchange accepts, and a backtest that quietly fills it is
        flattering itself by the fractional remainder on every position.
        Rounded toward zero: the conservative direction for longs and shorts
        alike, since rounding away from zero would exceed the intended weight.
        """
        quantity = target_value / (price * instrument.multiplier)
        lot = instrument.lot_size if instrument.lot_size > 1 else Decimal(1)
        lots = (quantity / lot).to_integral_value(rounding=ROUND_DOWN)
        return lots * lot

    def _unit_value(self, instrument_id: InstrumentId, price: Decimal) -> Decimal:
        return price * self.instruments[instrument_id].multiplier

    def _fund(
        self,
        portfolio: Portfolio,
        planned: list[tuple[InstrumentId, Decimal]],
        marks: dict[InstrumentId, Decimal],
    ) -> list[tuple[InstrumentId, Decimal]]:
        """Drop or trim buys the account cannot fund.

        Sell proceeds are credited to the running balance first, since they
        settle into the same session.
        """
        sells = [(i, q) for i, q in planned if q < 0]
        buys = [(i, q) for i, q in planned if q > 0]

        available = portfolio.cash + portfolio.margin_allowance
        for instrument_id, quantity in sells:
            available += abs(quantity) * self._unit_value(instrument_id, marks[instrument_id])

        funded: list[tuple[InstrumentId, Decimal]] = list(sells)
        # Deterministic ordering so a cash shortfall always trims the same
        # names (§14.1.1). Largest first: the strategy's biggest conviction is
        # funded before its smallest.
        for instrument_id, quantity in sorted(
            buys, key=lambda kv: (-abs(kv[1]) * marks[kv[0]], kv[0])
        ):
            # unit_value is positive by construction: `plan` already dropped
            # non-positive prices, and Instrument.multiplier is constrained
            # gt=0. A second guard here would be unreachable code.
            unit_value = self._unit_value(instrument_id, marks[instrument_id])
            budget = available * (Decimal(1) - self.config.cost_headroom)
            affordable = min(quantity, budget / unit_value)
            affordable = self._round_down_to_lot(self.instruments[instrument_id], affordable)
            if affordable * unit_value < self.config.min_order_value:
                continue
            funded.append((instrument_id, affordable))
            available -= affordable * unit_value
        return funded

    @staticmethod
    def _round_down_to_lot(instrument: Instrument, quantity: Decimal) -> Decimal:
        """Round down, never up: rounding up would exceed the funded budget.

        The lot floor is one share for the same reason as `_target_quantity` —
        a trimmed buy of 42.7 shares is still not an order.
        """
        lot = instrument.lot_size if instrument.lot_size > 1 else Decimal(1)
        lots = (quantity / lot).to_integral_value(rounding=ROUND_DOWN)
        return lots * lot

    def affordable(
        self,
        portfolio: Portfolio,
        instrument: Instrument,
        price: Decimal,
        wanted: Decimal,
        cost_of: Callable[[Decimal, Decimal], Decimal],
    ) -> Decimal:
        """Largest buy the account can pay for at `price`.

        Called twice on the way to a fill, at successively better estimates of
        the price: once against the fill model's reference, and once against the
        realised fill price after slippage. Each earlier estimate is one adverse
        tick from being too optimistic, and the consequence of the last one
        being wrong is an overdrawn account.

        Args:
            cost_of: (quantity, price) -> total cost. Injected so this stays
                independent of any particular cost model.
        """
        available = portfolio.cash + portfolio.margin_allowance
        unit_value = price * instrument.multiplier
        if unit_value <= 0:
            return Decimal(0)

        estimated = cost_of(wanted, price)
        if wanted * unit_value + estimated <= available:
            return wanted

        room = (available - estimated) / unit_value
        if room <= 0:
            return Decimal(0)
        room = self._round_down_to_lot(instrument, room)
        return max(Decimal(0), room.to_integral_value(rounding=ROUND_DOWN))
