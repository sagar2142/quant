"""What the paper book is right now — MASTER_PLAN §8, §12.7.

**Why this module exists.** Two surfaces were reporting numbers they had not
measured. `/vitals` returned literal zeros with a docstring calling them
placeholders, and the console's Risk screen hardcoded `observed: 0` and
`passed: true` for every limit. Both predate the paper daemon; neither was
updated when it started writing real state.

A monitoring surface that reports zero when it means "not measured" is the
worst failure mode available to it. Day P&L of exactly ₹0.00 is a plausible
quiet session. A drawdown of 0.0% is a plausible healthy book. Staleness of
0.0s with every feed green is a plausible live system — and the real state on
disk was three days old with a 0.15% drawdown. Nothing looked wrong, which is
precisely the problem: the operator has no way to tell a calm book from a
screen that is not reading one.

So this derives every observable quantity from the paper state file once, and
says `None` for anything it genuinely cannot know. `None` renders as an em
dash, not as zero. A per-order limit like the fat-finger price band has no
value at rest, and claiming one would be inventing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from core.clock import utc_now
from core.instruments import InstrumentId
from engine.accounting import Position
from trading.paper.state import PaperStateStore, StateCorruptError

__all__ = ["BookSnapshot", "book_snapshot"]

#: Cycles needed before a day-over-day change exists at all. One cycle has
#: nothing to difference against, and comparing it to starting capital would
#: report the whole book's P&L as a single session's.
CYCLES_FOR_A_DIFFERENCE = 2


@dataclass(frozen=True)
class BookSnapshot:
    """Derived state of the paper account, or `present=False` if there is none.

    Every field that cannot be measured is `None` rather than zero. Callers are
    expected to render that distinction rather than collapse it.
    """

    present: bool
    equity: Decimal | None = None
    cash: Decimal | None = None
    peak_equity: Decimal | None = None
    #: Fraction below the high-water mark, **negative**: -0.0015 is a 0.15%
    #: drawdown. Negative because `LadderRung.drawdown_pct` is validated
    #: negative and both the engine and the console's ladder meter compare
    #: against it directly — a positive convention here would leave every rung
    #: unlit through an arbitrarily deep drawdown, and colour the loss green.
    drawdown: Decimal | None = None
    #: Change in equity across the last two recorded cycles. None until two
    #: exist — a single cycle has nothing to difference against, and comparing
    #: it to starting capital would report the whole book's P&L as one day's.
    day_pnl: Decimal | None = None
    day_pnl_pct: Decimal | None = None
    #: Long plus short, as a fraction of equity.
    gross_exposure: Decimal | None = None
    #: Long minus short, as a fraction of equity.
    net_exposure: Decimal | None = None
    #: Largest single position as a fraction of equity.
    largest_position_pct: Decimal | None = None
    last_cycle_at: datetime | None = None
    #: Seconds since the last completed cycle. None when none has run.
    staleness_seconds: float | None = None
    cycles: int = 0
    halted: bool = False
    halt_reason: str = ""


def _exposures(
    positions: Mapping[InstrumentId, Position],
    marks: Mapping[InstrumentId, Decimal],
    equity: Decimal,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    """Gross, net and largest-position fractions of equity.

    Returns three Nones on a non-positive equity: dividing by it would produce
    a number with no meaning, and a risk screen showing an infinite exposure is
    less informative than one admitting it cannot say.
    """
    if equity <= 0:
        return None, None, None

    values = []
    for instrument_id, position in positions.items():
        if position.is_flat:
            continue
        mark = marks.get(instrument_id, position.average_price)
        values.append(position.market_value(mark))

    if not values:
        return Decimal(0), Decimal(0), Decimal(0)

    gross = sum((abs(v) for v in values), Decimal(0)) / equity
    net = sum(values, Decimal(0)) / equity
    largest = max(abs(v) for v in values) / equity
    return gross, net, largest


def book_snapshot(
    state_dir: Path, marks: dict[InstrumentId, Decimal] | None = None
) -> BookSnapshot:
    """Read the paper state and derive what can be observed from it.

    Args:
        state_dir: Where `apps.cli.paper` writes.
        marks: Latest close per instrument. Positions fall back to average
            price when a mark is missing, which shows them at cost — visibly
            wrong rather than invisibly stale.

    A missing or corrupt state file yields `present=False` rather than raising.
    A monitoring surface that 500s tells the operator less than one that says
    nothing is there, and the CLI is where corruption gets diagnosed.
    """
    store = PaperStateStore(state_dir)
    if not store.exists():
        return BookSnapshot(present=False)

    try:
        state = store.restore()
    except StateCorruptError:
        return BookSnapshot(present=False)

    marks = marks or {}
    positions = state.portfolio.positions
    position_value = Decimal(0)
    for instrument_id, position in positions.items():
        if position.is_flat:
            continue
        position_value += position.market_value(marks.get(instrument_id, position.average_price))

    equity = state.portfolio.cash + position_value
    gross, net, largest = _exposures(positions, marks, equity)

    drawdown: Decimal | None = None
    if state.peak_equity > 0:
        # Clamped at zero from above: a book at a new high is flat, not in a
        # positive drawdown. Negative below that, matching the rungs.
        drawdown = min(Decimal(0), (equity - state.peak_equity) / state.peak_equity)

    day_pnl: Decimal | None = None
    day_pnl_pct: Decimal | None = None
    history = store.equity_history()
    if len(history) >= CYCLES_FOR_A_DIFFERENCE:
        previous = Decimal(history[-2]["equity"])
        latest = Decimal(history[-1]["equity"])
        day_pnl = latest - previous
        if previous > 0:
            day_pnl_pct = day_pnl / previous

    staleness: float | None = None
    if state.last_cycle_at is not None:
        staleness = max(0.0, (utc_now() - state.last_cycle_at).total_seconds())

    return BookSnapshot(
        present=True,
        equity=equity,
        cash=state.portfolio.cash,
        peak_equity=state.peak_equity,
        drawdown=drawdown,
        day_pnl=day_pnl,
        day_pnl_pct=day_pnl_pct,
        gross_exposure=gross,
        net_exposure=net,
        largest_position_pct=largest,
        last_cycle_at=state.last_cycle_at,
        staleness_seconds=staleness,
        cycles=state.cycles,
        halted=state.halted,
        halt_reason=state.halt_reason,
    )
