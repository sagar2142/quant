"""Order lifecycle — MASTER_PLAN §19.

The state machine from the plan, verbatim, with two properties that matter more
than the diagram.

**Idempotency is structural.** Every order carries a client-generated key.
Networks fail mid-submit; a retry that produces a second fill is not a bug you
can recover from by apologising to yourself. The key is a database UNIQUE
constraint, so a duplicate submit fails at the storage layer rather than
relying on any caller remembering.

**UNKNOWN is the load-bearing state.** When a submit times out you do not know
whether the venue received it. The wrong answers are "assume rejected" (you
resend, and now hold double) and "assume filled" (you do not resend, and hold
nothing while believing otherwise). The right answer is UNKNOWN, which is never
resolved by inference — only by reconciliation against what the broker actually
holds (§9).

**Illegal transitions raise.** A state machine that tolerates an undefined
transition is a state machine that has stopped describing reality.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from core.clock import require_utc, utc_now
from core.instruments import InstrumentId
from core.orders import LEGAL_TRANSITIONS, OrderState, OrderType, Side

__all__ = ["IllegalTransitionError", "Order", "OrderTransition", "TradingMode"]


class IllegalTransitionError(RuntimeError):
    """An order was asked to move between states that cannot connect.

    Never suppressed. If the venue reports a transition the machine does not
    model, the model is wrong and continuing would mean acting on a fiction.
    """

    def __init__(self, order_id: uuid.UUID, current: OrderState, target: OrderState) -> None:
        legal = sorted(s.value for s in LEGAL_TRANSITIONS[current])
        super().__init__(
            f"order {order_id}: cannot move {current.value} -> {target.value}. "
            f"Legal from {current.value}: {legal or 'nothing (terminal)'}"
        )
        self.current = current
        self.target = target


class TradingMode(str, Enum):
    """Which plane an order belongs to. Recorded on every row."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"

    @property
    def touches_real_money(self) -> bool:
        return self is TradingMode.LIVE


@dataclass(frozen=True)
class OrderTransition:
    """One state change, appended and never rewritten.

    Reconstructing an order's history is how a fill you did not expect gets
    diagnosed.
    """

    from_state: OrderState | None
    to_state: OrderState
    occurred_at: datetime
    detail: str = ""


@dataclass
class Order:
    """A single order and its history.

    Quantities and prices are Decimal (§14.1.2): these are reconciled against a
    broker's contract note, and a float rounding error is a reconciliation
    break at 2am.
    """

    strategy_id: str
    instrument_id: InstrumentId
    side: Side
    quantity: Decimal
    order_type: OrderType
    mode: TradingMode
    decision_time: datetime

    limit_price: Decimal | None = None
    stop_price: Decimal | None = None

    order_id: uuid.UUID = field(default_factory=uuid.uuid4)
    #: Client-generated. The venue may reject a duplicate; so does our database.
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)
    broker_order_id: str | None = None

    state: OrderState = OrderState.CREATED
    filled_quantity: Decimal = Decimal(0)
    average_fill_price: Decimal = Decimal(0)
    history: list[OrderTransition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type.needs_limit_price and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} requires a limit price")
        if self.order_type.needs_stop_price and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} requires a stop price")
        object.__setattr__(self, "decision_time", require_utc(self.decision_time))
        if not self.history:
            self.history.append(OrderTransition(None, OrderState.CREATED, utc_now(), "created"))

    # ── state ───────────────────────────────────────────────────────────────

    @property
    def remaining(self) -> Decimal:
        return self.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    @property
    def is_live(self) -> bool:
        """Whether this order can still consume risk budget at the venue."""
        return self.state.is_live

    @property
    def needs_reconciliation(self) -> bool:
        return self.state is OrderState.UNKNOWN

    def transition(self, target: OrderState, detail: str = "") -> None:
        """Move to `target`, recording the change.

        Raises:
            IllegalTransitionError: if the move is not in `LEGAL_TRANSITIONS`.
        """
        if target not in LEGAL_TRANSITIONS[self.state]:
            raise IllegalTransitionError(self.order_id, self.state, target)
        self.history.append(OrderTransition(self.state, target, utc_now(), detail))
        self.state = target

    def apply_fill(self, quantity: Decimal, price: Decimal) -> None:
        """Record a fill and advance the state.

        Raises:
            ValueError: on a non-positive fill, or one exceeding the remainder.
                Over-filling is a venue or adapter bug and must never be
                absorbed silently — it would corrupt the position permanently.
        """
        if quantity <= 0:
            raise ValueError(f"fill quantity must be positive, got {quantity}")
        if price <= 0:
            raise ValueError(f"fill price must be positive, got {price}")
        if quantity > self.remaining:
            raise ValueError(
                f"fill of {quantity} exceeds remaining {self.remaining} on order {self.order_id}"
            )

        filled = self.filled_quantity + quantity
        # Weighted average across partials.
        self.average_fill_price = (
            self.average_fill_price * self.filled_quantity + price * quantity
        ) / filled
        self.filled_quantity = filled

        target = OrderState.FILLED if self.remaining == 0 else OrderState.PARTIALLY_FILLED
        self.transition(target, f"filled {quantity} @ {price}")

    def mark_unknown(self, detail: str) -> None:
        """Record that the venue's view of this order is not known.

        Deliberately not a terminal state and deliberately not inferred from a
        timeout. Only reconciliation resolves it.
        """
        self.transition(OrderState.UNKNOWN, detail)

    def format_history(self) -> str:
        lines = [f"order {self.order_id} [{self.state.value}]"]
        for t in self.history:
            origin = t.from_state.value if t.from_state else "-"
            lines.append(
                f"  {t.occurred_at.isoformat()}  {origin:<18} -> {t.to_state.value:<18} {t.detail}"
            )
        return "\n".join(lines)
