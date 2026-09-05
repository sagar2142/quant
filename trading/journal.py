"""Execution quality — MASTER_PLAN §9, §M7.

**A backtest's costs are a model; a journal's costs are what happened.** The
cost model charges an estimate at every fill, and until something measures the
real thing against it, the estimate is only ever checked by whether the equity
curve looks plausible — which it will, because the model produced it.

This computes the difference, decomposed the way Perold's implementation
shortfall does, because the parts have different owners:

    decision price  →  intended price     delay: how long you sat on it
    intended price  →  average fill       execution: how you were filled
    fees                                  the schedule, and lot sizes
    unfilled quantity                     opportunity: the trade you missed

Lumping these into one "slippage" number is the usual mistake. Delay cost is a
process problem, execution cost is a broker or an order-type problem, and
opportunity cost is a sizing problem; a single figure tells you something is
wrong and nothing about which of the three.

**Nothing is inferred when it is missing.** An order with no recorded decision
price has no measurable delay cost, and that is reported as unmeasured rather
than as zero — a zero would average into the totals and quietly flatter them.

**Pure.** This takes rows and returns numbers; the database read lives in the
API. That keeps the arithmetic testable without a Postgres, and keeps the
trading layer from reaching sideways into ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

__all__ = [
    "BPS",
    "Execution",
    "JournalSummary",
    "summarise",
]

#: One basis point. Slippage is quoted in bps because it is compared across
#: names whose prices differ by two orders of magnitude.
BPS = Decimal(10_000)


def _sign(side: str) -> Decimal:
    """+1 for a buy, -1 for a sell.

    Cost is *adverse* movement, so both sides must be positive when they hurt:
    paying 1 above the decision price on a buy and receiving 1 below it on a
    sell are the same loss, and a signless difference calls one of them a gain.
    """
    return Decimal(1) if side.upper() == "BUY" else Decimal(-1)


@dataclass(frozen=True)
class Execution:
    """One order and what it actually cost.

    All prices are per unit. Quantities are in units, not lots — the database
    stores them that way, and converting here would silently double-apply a
    lot size the caller may already have handled.
    """

    order_id: str
    instrument_id: str
    symbol: str
    side: str
    strategy_id: str
    mode: str
    decision_time: datetime
    quantity_ordered: Decimal
    quantity_filled: Decimal
    average_fill: Decimal | None
    fees: Decimal
    #: The market price when the decision was taken. `None` when no mark was
    #: recorded — delay cost is then unmeasured, not zero.
    decision_price: Decimal | None = None
    #: What the order aimed at when it was sent.
    intended_price: Decimal | None = None
    first_fill_time: datetime | None = None
    last_fill_time: datetime | None = None

    @property
    def is_filled(self) -> bool:
        return self.quantity_filled > 0

    @property
    def unfilled_quantity(self) -> Decimal:
        return max(Decimal(0), self.quantity_ordered - self.quantity_filled)

    @property
    def fill_rate(self) -> float:
        if self.quantity_ordered == 0:
            return 0.0
        return float(self.quantity_filled / self.quantity_ordered)

    @property
    def delay_cost(self) -> Decimal | None:
        """Money lost between deciding and sending.

        The price moved while the order sat. Positive is a cost. `None` when
        either mark is missing — an unmeasured delay is not a zero one.
        """
        if self.decision_price is None or self.intended_price is None or not self.is_filled:
            return None
        return (self.intended_price - self.decision_price) * _sign(self.side) * self.quantity_filled

    @property
    def execution_cost(self) -> Decimal | None:
        """Money lost between sending and being filled.

        This is the broker's and the order type's contribution, and the one a
        different venue or a limit instead of a market would actually change.
        """
        if self.intended_price is None or self.average_fill is None or not self.is_filled:
            return None
        return (self.average_fill - self.intended_price) * _sign(self.side) * self.quantity_filled

    @property
    def slippage(self) -> Decimal | None:
        """Total price cost against the decision, excluding fees.

        Falls back to the intended price when no decision mark exists, and
        says as much by way of `decision_price` being None: the number is then
        execution cost alone, which is a smaller claim than shortfall.
        """
        # `is None`, not `or`: a zero reference is not a missing one, and
        # silently falling through on it would measure the wrong quantity
        # rather than declining to measure.
        reference = self.decision_price if self.decision_price is not None else self.intended_price
        if reference is None or self.average_fill is None or not self.is_filled:
            return None
        return (self.average_fill - reference) * _sign(self.side) * self.quantity_filled

    @property
    def slippage_bps(self) -> float | None:
        """Slippage as basis points of the reference notional."""
        # `is None`, not `or`: a zero reference is not a missing one, and
        # silently falling through on it would measure the wrong quantity
        # rather than declining to measure.
        reference = self.decision_price if self.decision_price is not None else self.intended_price
        cost = self.slippage
        if cost is None or reference is None or reference == 0:
            return None
        return float(cost / (reference * self.quantity_filled) * BPS)

    @property
    def implementation_shortfall(self) -> Decimal | None:
        """Price cost plus fees, against the decision price.

        The number that belongs beside a backtest's modelled cost, because the
        backtest also charges from a decision price and also pays fees.
        """
        cost = self.slippage
        return None if cost is None else cost + self.fees

    def opportunity_cost(self, later_price: Decimal | None) -> Decimal | None:
        """What the unfilled part would have made, had it filled.

        Args:
            later_price: A mark after the order went terminal.

        Returns:
            None when nothing went unfilled, or when no later mark or decision
            price is available. Positive means the miss was expensive.
        """
        if later_price is None or self.decision_price is None or self.unfilled_quantity == 0:
            return None
        return (later_price - self.decision_price) * _sign(self.side) * self.unfilled_quantity


@dataclass(frozen=True)
class JournalSummary:
    """Execution quality across a set of orders."""

    orders: int
    filled: int
    #: Orders with enough marks to measure shortfall. The rest are counted so
    #: a total computed from a third of the book cannot be mistaken for the
    #: whole book's.
    measured: int
    notional: Decimal
    fees: Decimal
    delay_cost: Decimal | None
    execution_cost: Decimal | None
    slippage: Decimal | None
    implementation_shortfall: Decimal | None
    #: Volume-weighted, so a large order counts for more than a small one —
    #: the equal-weighted average is the number that looks fine while the size
    #: that matters is being filled badly.
    shortfall_bps: float | None
    worst: list[Execution]

    @property
    def coverage(self) -> float:
        """Share of filled orders that could be measured at all."""
        return self.measured / self.filled if self.filled else 0.0

    def format(self) -> str:
        def money(value: Decimal | None) -> str:
            return "—" if value is None else f"{value:>14,.2f}"

        lines = [
            f"  {self.orders:,} order(s), {self.filled:,} filled,"
            f" {self.measured:,} measurable ({self.coverage:.0%})",
            f"  notional            {self.notional:>14,.2f}",
            "",
            f"  delay cost          {money(self.delay_cost)}",
            f"  execution cost      {money(self.execution_cost)}",
            f"  fees                {money(self.fees)}",
            "  ─────────────────────────────────",
            f"  shortfall           {money(self.implementation_shortfall)}",
        ]
        if self.shortfall_bps is not None:
            lines.append(f"  shortfall (bps)     {self.shortfall_bps:>14.2f}")
        if self.measured < self.filled:
            lines.append(
                f"\n  {self.filled - self.measured} filled order(s) had no decision mark;"
                " their cost is unmeasured, not zero."
            )
        return "\n".join(lines)


def _total(values: list[Decimal | None]) -> Decimal | None:
    """Sum, or None if nothing was measurable.

    None rather than zero: "no order could be measured" and "the orders cost
    nothing" are opposite findings that a zero would render identically.
    """
    present = [v for v in values if v is not None]
    return sum(present, Decimal(0)) if present else None


def summarise(executions: list[Execution], worst_n: int = 5) -> JournalSummary:
    """Aggregate execution quality.

    Args:
        executions: The orders to summarise.
        worst_n: How many of the most expensive to name. Named rather than
            merely counted, because a shortfall concentrated in two orders and
            one spread evenly across two hundred call for different fixes.
    """
    filled = [e for e in executions if e.is_filled]
    measurable = [e for e in filled if e.implementation_shortfall is not None]

    notional = sum((e.average_fill or Decimal(0)) * e.quantity_filled for e in filled) or Decimal(0)

    shortfall = _total([e.implementation_shortfall for e in filled])
    shortfall_bps = (
        float(shortfall / notional * BPS) if shortfall is not None and notional else None
    )

    worst = sorted(
        measurable,
        key=lambda e: e.implementation_shortfall or Decimal(0),
        reverse=True,
    )[:worst_n]

    return JournalSummary(
        orders=len(executions),
        filled=len(filled),
        measured=len(measurable),
        notional=Decimal(notional),
        fees=sum((e.fees for e in filled), Decimal(0)),
        delay_cost=_total([e.delay_cost for e in filled]),
        execution_cost=_total([e.execution_cost for e in filled]),
        slippage=_total([e.slippage for e in filled]),
        implementation_shortfall=shortfall,
        shortfall_bps=shortfall_bps,
        worst=worst,
    )
