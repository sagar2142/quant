"""Position and cash accounting — MASTER_PLAN §14.1.2, §17.

The ledger. Every rupee that enters or leaves a simulated or real portfolio
passes through here, which is why it is `Decimal` end to end and why §14.5
demands 100% test coverage: a rounding error here does not crash, it quietly
produces a P&L that fails reconciliation months later.

Lives in `engine/` rather than `trading/` because the backtester needs it and
§3.2 forbids `engine/` importing `trading/`. The live trading layer builds on
this same primitive, so simulated and real accounting cannot drift apart —
which is the only way paper-versus-backtest drift analysis means anything.

**Shorts are negative quantities**, not a separate flag. One signed number
removes an entire class of branch-and-forget bugs, and it makes a long-short
portfolio's net exposure a plain sum.

**Corporate actions are applied to positions, never to prices** (see
`data.corpactions`). A split multiplies your share count and divides your
average price on the ex-date, exactly as it happens to a real holding.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from decimal import Decimal

from core.clock import require_utc
from core.instruments import InstrumentId
from core.orders import Side
from engine.costs.model import CostBreakdown, quantize_money

__all__ = ["Fill", "InsufficientCashError", "Portfolio", "Position"]


class InsufficientCashError(RuntimeError):
    """A buy would take cash below the permitted floor.

    Raised rather than silently allowing negative cash (§14.1.5): an unnoticed
    overdraft in a backtest is free leverage, and free leverage flatters every
    metric downstream.
    """

    def __init__(self, needed: Decimal, available: Decimal) -> None:
        super().__init__(
            f"insufficient cash: need {needed}, have {available}. "
            "Enable margin explicitly if leverage is intended."
        )


@dataclass(frozen=True)
class Fill:
    """An executed trade. The only thing that moves a portfolio."""

    instrument_id: InstrumentId
    side: Side
    quantity: Decimal
    price: Decimal
    costs: CostBreakdown
    event_time: datetime
    multiplier: Decimal = Decimal(1)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"fill quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"fill price must be positive, got {self.price}")
        object.__setattr__(self, "event_time", require_utc(self.event_time))

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity * self.side.sign

    @property
    def gross_value(self) -> Decimal:
        """Trade value before costs. Always positive."""
        return self.price * self.quantity * self.multiplier

    @property
    def cash_delta(self) -> Decimal:
        """Change in cash. Buys consume, sells release; costs always consume."""
        direction = -self.side.sign
        return quantize_money(self.gross_value * direction - self.costs.total)


@dataclass(frozen=True)
class Position:
    """A holding in one instrument. Immutable; every update returns a new one."""

    instrument_id: InstrumentId
    quantity: Decimal = Decimal(0)
    average_price: Decimal = Decimal(0)
    realised_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)
    multiplier: Decimal = Decimal(1)

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def cost_basis(self) -> Decimal:
        """Signed capital committed. Negative for shorts."""
        return self.quantity * self.average_price * self.multiplier

    def market_value(self, price: Decimal) -> Decimal:
        return quantize_money(self.quantity * price * self.multiplier)

    def unrealised_pnl(self, price: Decimal) -> Decimal:
        """Mark-to-market gain. Correct for shorts without a special case,
        because quantity carries the sign."""
        if self.is_flat:
            return Decimal(0)
        return quantize_money((price - self.average_price) * self.quantity * self.multiplier)

    def apply(self, fill: Fill) -> tuple[Position, Decimal]:
        """Apply a fill.

        Returns:
            The new position, and the P&L realised by this fill (zero when the
            fill only opens or increases exposure).

        Three cases, and the third is the one implementations get wrong:

        1. **Opening or increasing** — weighted-average the entry price.
        2. **Reducing or closing** — realise P&L on the closed portion; the
           average price is untouched, because the remaining shares were bought
           at that average.
        3. **Flipping through zero** — realise the *entire* old position first,
           then open the remainder at the fill price. Treating a flip as a
           simple net leaves a phantom average price mixing long and short
           entries, and the error persists for the life of the position.
        """
        new_fees = self.fees_paid + fill.costs.total
        old_qty = self.quantity
        delta = fill.signed_quantity
        new_qty = old_qty + delta

        # Case 1: flat, or moving further from zero in the same direction.
        if old_qty == 0 or (old_qty > 0) == (delta > 0):
            total_cost = abs(old_qty) * self.average_price + abs(delta) * fill.price
            avg = total_cost / abs(new_qty) if new_qty != 0 else Decimal(0)
            return (
                replace(
                    self,
                    quantity=new_qty,
                    average_price=avg,
                    fees_paid=new_fees,
                    multiplier=fill.multiplier,
                ),
                Decimal(0),
            )

        closing_qty = min(abs(delta), abs(old_qty))
        direction = Decimal(1) if old_qty > 0 else Decimal(-1)
        realised = quantize_money(
            (fill.price - self.average_price) * closing_qty * direction * fill.multiplier
        )

        # Case 3: flipped past zero — the remainder opens a fresh position.
        if (new_qty > 0) != (old_qty > 0) and new_qty != 0:
            return (
                replace(
                    self,
                    quantity=new_qty,
                    average_price=fill.price,
                    realised_pnl=self.realised_pnl + realised,
                    fees_paid=new_fees,
                    multiplier=fill.multiplier,
                ),
                realised,
            )

        # Case 2: reduced or exactly closed.
        return (
            replace(
                self,
                quantity=new_qty,
                average_price=self.average_price if new_qty != 0 else Decimal(0),
                realised_pnl=self.realised_pnl + realised,
                fees_paid=new_fees,
                multiplier=fill.multiplier,
            ),
            realised,
        )

    def apply_split(self, ratio: Decimal) -> Position:
        """Apply a split or bonus on its ex-date.

        Share count multiplies, average price divides, so position value is
        preserved — which is what actually happens to a holding.
        """
        if ratio <= 0:
            raise ValueError(f"split ratio must be positive, got {ratio}")
        return replace(
            self,
            quantity=self.quantity * ratio,
            average_price=self.average_price / ratio,
        )


@dataclass
class Portfolio:
    """Cash plus positions. The complete state of an account."""

    cash: Decimal
    positions: dict[InstrumentId, Position] = field(default_factory=dict)
    realised_pnl: Decimal = Decimal(0)
    fees_paid: Decimal = Decimal(0)
    #: Cash may fall this far below zero. Zero means no leverage at all.
    margin_allowance: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        self.starting_cash = self.cash

    def position(self, instrument_id: InstrumentId) -> Position:
        """Current holding, or a flat one. Never raises — absence is flat."""
        return self.positions.get(instrument_id, Position(instrument_id))

    def apply_fill(self, fill: Fill) -> Decimal:
        """Apply a fill to cash and positions.

        Returns:
            Realised P&L from this fill.

        Raises:
            InsufficientCashError: if the fill would breach the margin
                allowance.
        """
        new_cash = self.cash + fill.cash_delta
        if new_cash < -self.margin_allowance:
            raise InsufficientCashError(abs(fill.cash_delta), self.cash)

        position, realised = self.position(fill.instrument_id).apply(fill)
        self.positions[fill.instrument_id] = position
        self.cash = quantize_money(new_cash)
        self.realised_pnl = quantize_money(self.realised_pnl + realised)
        self.fees_paid = quantize_money(self.fees_paid + fill.costs.total)
        return realised

    def apply_split(self, instrument_id: InstrumentId, ratio: Decimal) -> None:
        """Apply a split or bonus to a held position. No-op when flat."""
        position = self.position(instrument_id)
        if not position.is_flat:
            self.positions[instrument_id] = position.apply_split(ratio)

    def apply_dividend(self, instrument_id: InstrumentId, cash_per_share: Decimal) -> Decimal:
        """Credit a dividend. A short position *pays* it.

        Returns the cash movement, which is negative when short.
        """
        position = self.position(instrument_id)
        if position.is_flat:
            return Decimal(0)
        amount = quantize_money(position.quantity * cash_per_share)
        self.cash = quantize_money(self.cash + amount)
        return amount

    def apply_funding(self, amount: Decimal) -> None:
        """Apply a perpetual funding cash flow. Positive `amount` is paid."""
        self.cash = quantize_money(self.cash - amount)
        self.fees_paid = quantize_money(self.fees_paid + amount)

    def market_value(self, prices: dict[InstrumentId, Decimal]) -> Decimal:
        """Total value of open positions at the given marks.

        Raises:
            KeyError: if a held instrument has no price. Substituting a stale or
                zero mark would silently misstate equity (§14.1.5).
        """
        total = Decimal(0)
        for instrument_id, position in self.positions.items():
            if position.is_flat:
                continue
            if instrument_id not in prices:
                raise KeyError(f"no mark for held position {instrument_id}")
            total += position.market_value(prices[instrument_id])
        return quantize_money(total)

    def equity(self, prices: dict[InstrumentId, Decimal]) -> Decimal:
        """Cash plus position value. The number a drawdown is measured on."""
        return quantize_money(self.cash + self.market_value(prices))

    def unrealised_pnl(self, prices: dict[InstrumentId, Decimal]) -> Decimal:
        total = Decimal(0)
        for instrument_id, position in self.positions.items():
            if not position.is_flat and instrument_id in prices:
                total += position.unrealised_pnl(prices[instrument_id])
        return quantize_money(total)

    def gross_exposure(self, prices: dict[InstrumentId, Decimal]) -> Decimal:
        """Sum of absolute position values — long plus short."""
        total = Decimal(0)
        for instrument_id, position in self.positions.items():
            if not position.is_flat and instrument_id in prices:
                total += abs(position.market_value(prices[instrument_id]))
        return quantize_money(total)

    def net_exposure(self, prices: dict[InstrumentId, Decimal]) -> Decimal:
        """Long minus short. A market-neutral book sits near zero."""
        return self.market_value(prices)

    def open_positions(self) -> dict[InstrumentId, Position]:
        return {k: v for k, v in self.positions.items() if not v.is_flat}
