"""Order vocabulary — MASTER_PLAN §19.

Only the *nouns* live here: side, type, state, and the legal transitions between
states. The machine that drives them lives in `trading/execution`, which the
research and cost layers may not import (§3.2). Putting the vocabulary in `core`
lets a cost model reason about buy-versus-sell without reaching into the
trading plane.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["LEGAL_TRANSITIONS", "TERMINAL_STATES", "OrderState", "OrderType", "Side"]


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for buys, -1 for sells. Position deltas are always signed."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

    @property
    def needs_limit_price(self) -> bool:
        return self in {OrderType.LIMIT, OrderType.STOP_LIMIT}

    @property
    def needs_stop_price(self) -> bool:
        return self in {OrderType.STOP, OrderType.STOP_LIMIT}


class OrderState(str, Enum):
    """The state machine from §19, verbatim.

    UNKNOWN is the load-bearing one: networks fail mid-submit, and "did that
    order reach the broker?" must be answerable without guessing. An order in
    UNKNOWN is never assumed dead — it is reconciled.
    """

    CREATED = "CREATED"
    RISK_CHECKED = "RISK_CHECKED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def is_live(self) -> bool:
        """Whether the order can still consume risk budget at the venue."""
        return self in {
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.UNKNOWN,
        }


TERMINAL_STATES = frozenset({OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED})

#: Every legal transition. Anything absent is a bug, not an edge case, and the
#: execution layer raises rather than tolerating it (§14.1.5).
LEGAL_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.RISK_CHECKED, OrderState.REJECTED}),
    OrderState.RISK_CHECKED: frozenset({OrderState.SUBMITTED, OrderState.REJECTED}),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
            # Venues may fill instantly without a distinct ack.
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.CANCEL_REQUESTED: frozenset(
        {
            OrderState.CANCELLED,
            # A cancel can lose the race with a fill.
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.UNKNOWN,
        }
    ),
    # Reconciliation resolves UNKNOWN into whatever the broker actually holds.
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.CANCELLED: frozenset(),
}


def is_legal_transition(current: OrderState, target: OrderState) -> bool:
    return target in LEGAL_TRANSITIONS[current]
