"""Position and cash reconciliation — MASTER_PLAN §9, §24.

Runs every day, without exception, in paper and live alike.

**An unexplained break is a system-down event, not a rounding issue.** The plan
is unambiguous about this and it is worth restating: if the broker thinks you
hold 100 shares and you think you hold 90, one of those numbers is being used
to size your next order, and you do not know which is right. The correct
response is to halt new orders and find out — not to adjust your own record to
match and carry on.

This is also how order-logic bugs get caught *before* they cost money. A
double-counted fill or a missed partial shows up here first, as a discrepancy
of exactly the size of the error.

**Decimal throughout, and the tolerance is zero by default.** Share counts are
integers and cash settles to the paisa; a "small" mismatch is not small, it is
a mismatch whose cause you have not found yet. A non-zero tolerance is
available for venues that genuinely round, and using it is a decision to be
made explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from core.clock import utc_now
from core.instruments import InstrumentId
from engine.accounting import Portfolio
from trading.execution.broker import BrokerPosition

__all__ = ["BreakKind", "ReconciliationBreak", "ReconciliationReport", "reconcile_positions"]


class BreakKind(str, Enum):
    """What kind of disagreement was found."""

    #: We hold it, the broker does not.
    PHANTOM = "PHANTOM"
    #: The broker holds it, we do not.
    UNRECORDED = "UNRECORDED"
    #: Both hold it, in different sizes.
    QUANTITY = "QUANTITY"
    #: Same size, different average price.
    PRICE = "PRICE"

    @property
    def is_critical(self) -> bool:
        """Whether this break can misstate exposure.

        A price disagreement misstates P&L, which matters. A quantity
        disagreement misstates *risk*, which matters more: the next order is
        sized against a position that does not exist.
        """
        return self is not BreakKind.PRICE


@dataclass(frozen=True)
class ReconciliationBreak:
    """One disagreement between internal records and the broker."""

    instrument_id: InstrumentId
    kind: BreakKind
    internal: Decimal
    broker: Decimal
    field_name: str = "quantity"

    @property
    def difference(self) -> Decimal:
        return self.internal - self.broker

    def format(self) -> str:
        return (
            f"  [{self.kind.value:<10}] {self.instrument_id:<28} "
            f"{self.field_name}: internal {self.internal} vs broker {self.broker} "
            f"(diff {self.difference})"
        )


@dataclass
class ReconciliationReport:
    """The outcome of one reconciliation run."""

    as_of: datetime = field(default_factory=utc_now)
    breaks: list[ReconciliationBreak] = field(default_factory=list)
    instruments_checked: int = 0

    @property
    def is_clean(self) -> bool:
        return not self.breaks

    @property
    def critical_breaks(self) -> list[ReconciliationBreak]:
        return [b for b in self.breaks if b.kind.is_critical]

    @property
    def should_halt(self) -> bool:
        """Whether trading must stop until this is explained (§9).

        Any critical break halts. This is not a judgement call the operator
        makes in the moment — it is the pre-committed response.
        """
        return bool(self.critical_breaks)

    def format(self) -> str:
        if self.is_clean:
            return (
                f"reconciliation {self.as_of.isoformat()}: CLEAN "
                f"({self.instruments_checked} instrument(s))"
            )
        verdict = "HALT REQUIRED" if self.should_halt else "breaks found"
        lines = [
            f"reconciliation {self.as_of.isoformat()}: {verdict} "
            f"— {len(self.breaks)} break(s) across {self.instruments_checked} instrument(s)"
        ]
        lines.extend(b.format() for b in self.breaks)
        if self.should_halt:
            lines.append(
                "\n  An unexplained break is a system-down event (§9). Halt new "
                "orders and find the cause; do not adjust the internal record to match."
            )
        return "\n".join(lines)


def reconcile_positions(
    portfolio: Portfolio,
    broker_positions: list[BrokerPosition],
    quantity_tolerance: Decimal = Decimal(0),
    price_tolerance: Decimal = Decimal(0),
) -> ReconciliationReport:
    """Compare internal positions against what the broker reports.

    Args:
        portfolio: Internal record.
        broker_positions: The venue's view. Treated as the reference, but never
            as automatically correct — a discrepancy means *find out which*.
        quantity_tolerance: Permitted absolute difference. Zero by default:
            share counts are integers, and a "small" mismatch is a mismatch
            whose cause has not been found.
        price_tolerance: Permitted average-price difference. Venues genuinely
            round here, so a small value is defensible.

    Returns:
        A report. Callers must consult `should_halt` before placing anything.
    """
    by_instrument = {p.instrument_id: p for p in broker_positions}
    internal = portfolio.open_positions()
    report = ReconciliationReport(instruments_checked=len(set(by_instrument) | set(internal)))

    for instrument_id in sorted(set(internal) | set(by_instrument)):
        ours = internal.get(instrument_id)
        theirs = by_instrument.get(instrument_id)

        if ours is not None and theirs is None:
            report.breaks.append(
                ReconciliationBreak(instrument_id, BreakKind.PHANTOM, ours.quantity, Decimal(0))
            )
            continue

        if ours is None and theirs is not None:
            report.breaks.append(
                ReconciliationBreak(
                    instrument_id, BreakKind.UNRECORDED, Decimal(0), theirs.quantity
                )
            )
            continue

        if ours is None or theirs is None:  # pragma: no cover — both-absent is impossible
            continue

        if abs(ours.quantity - theirs.quantity) > quantity_tolerance:
            report.breaks.append(
                ReconciliationBreak(
                    instrument_id, BreakKind.QUANTITY, ours.quantity, theirs.quantity
                )
            )
        elif abs(ours.average_price - theirs.average_price) > price_tolerance:
            report.breaks.append(
                ReconciliationBreak(
                    instrument_id,
                    BreakKind.PRICE,
                    ours.average_price,
                    theirs.average_price,
                    field_name="average_price",
                )
            )

    return report
