"""Cost of holding a short — MASTER_PLAN §7.3.

**Shorting is not free, and on the NSE cash segment it is mostly not possible.**
A retail account cannot carry an overnight short in cash equities at all:
intraday positions are squared off before the close, and holding one past it
requires Securities Lending and Borrowing, which is a separate market with its
own participants, its own availability, and its own price. The alternative is
single-stock futures, which exist for a few hundred names rather than three
thousand.

That matters here because this system can now build market-neutral books, and a
backtest that shorts freely is not describing a portfolio anyone could hold. The
long-short results measured before this existed were optimistic by an amount
nobody could size — which is the worst kind of wrong, because it is invisible
rather than merely large.

**The charge is per session held, not per trade.** Everything else in
`engine.costs` prices a transaction; this prices *time*. A short carried for
sixty sessions pays sixty times, whether or not it traded, which is exactly why
it cannot live in `CostModel.cost`.

**The default rate is deliberately not a market rate.** SLB fees are set by
scarcity and range from under 1% annualised on an abundant large-cap to well
past 20% on a name everyone wants to short — the very names a signal is most
likely to select. A single default cannot be right; what it can be is high
enough that a strategy surviving it is not surviving on an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from core.instruments import InstrumentId
from engine.costs.model import quantize_money

__all__ = [
    "DEFAULT_ANNUAL_RATE",
    "HARD_TO_BORROW_RATE",
    "SESSIONS_PER_YEAR",
    "BorrowModel",
]

#: Sessions per year, for turning an annual rate into a per-session one.
SESSIONS_PER_YEAR = 252

#: Annualised SLB fee assumed for a name with no quoted rate.
#:
#: Six percent is toward the expensive end of ordinary and nowhere near the
#: worst. It is chosen to be pessimistic on purpose: the cost of assuming too
#: little is a strategy that looks tradeable and is not, and this system exists
#: to reject ideas rather than to flatter them.
DEFAULT_ANNUAL_RATE = Decimal("0.06")

#: What a scarce name costs. Not a default — supplied per instrument when the
#: borrow is known to be tight, and recorded so the assumption is visible.
HARD_TO_BORROW_RATE = Decimal("0.25")


@dataclass(frozen=True)
class BorrowModel:
    """Per-session financing charge on short positions.

    Args:
        annual_rate: Applied to any instrument without a specific rate.
        rates: Known annualised rates per instrument. A name absent from this
            map is charged the default rather than nothing — an unknown borrow
            cost is not a zero borrow cost.
        borrowable: Instruments that can be shorted overnight at all. Empty
            means *no restriction is modelled*, which is the permissive default
            and is documented rather than assumed: on the NSE cash segment the
            truthful set for a retail account is close to empty, and a caller
            testing a market-neutral book should say which names it can
            actually borrow.
    """

    annual_rate: Decimal = DEFAULT_ANNUAL_RATE
    rates: dict[InstrumentId, Decimal] = field(default_factory=dict)
    borrowable: frozenset[InstrumentId] = frozenset()

    def rate_for(self, instrument_id: InstrumentId) -> Decimal:
        """Annualised borrow rate for one name."""
        return self.rates.get(instrument_id, self.annual_rate)

    def can_short(self, instrument_id: InstrumentId) -> bool:
        """Whether an overnight short is possible at all.

        An empty `borrowable` set means the restriction is not being modelled,
        so everything is allowed. That is a deliberate permissive default with
        a loud docstring rather than a silent one: making it restrictive by
        default would break every existing long-only backtest for a reason
        those backtests do not have.
        """
        return not self.borrowable or instrument_id in self.borrowable

    def session_charge(self, positions: dict[InstrumentId, Decimal], sessions: int = 1) -> Decimal:
        """Total borrow cost for holding these positions.

        Args:
            positions: Signed market value per instrument. Longs are ignored;
                only the short side pays to borrow.
            sessions: How many sessions the book was held for.

        Returns:
            A positive cost. Zero when nothing is short, which is why a
            long-only backtest is unaffected by this existing.
        """
        if sessions <= 0:
            return Decimal(0)

        total = Decimal(0)
        for instrument_id, value in positions.items():
            if value >= 0:
                continue
            daily = self.rate_for(instrument_id) / Decimal(SESSIONS_PER_YEAR)
            total += abs(value) * daily * Decimal(sessions)
        return quantize_money(total)
