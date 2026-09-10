"""Risk limits and state — MASTER_PLAN §8.

Three independent layers, each catching what the one before cannot:

    1. pre-trade    one order, synchronously — size, price sanity, rate
    2. portfolio    the whole book, continuously — exposure, loss, concentration
    3. strategic    the drawdown ladder — pre-committed de-risking

**The risk engine is independent by construction** (§3.2). It imports no
strategy code and no backtest code; a strategy cannot read its limits, raise
them, or reason about them. A strategy that can modify its own limit has no
limit.

**Decimal throughout, enforced by lint.** A limit is a comparison between two
money amounts, and a float rounding error here is the difference between a
blocked order and an allowed one.

**The correlation-cluster limit is the one people miss.** Ten positions in
correlated PSU banks is one bet, not ten. A gross-exposure limit sees ten
diversified holdings; a cluster limit sees the single bet that is actually
there, and it is what stops the Fundamental Law being defeated silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from core.instruments import InstrumentId

__all__ = [
    "DrawdownLadder",
    "LadderRung",
    "PortfolioState",
    "ProposedOrder",
    "RiskCheck",
    "RiskDecision",
    "RiskLimits",
]


class RiskDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"

    @property
    def is_blocked(self) -> bool:
        return self is RiskDecision.BLOCK


@dataclass(frozen=True)
class RiskCheck:
    """One limit, evaluated."""

    name: str
    passed: bool
    observed: Decimal | None = None
    threshold: Decimal | None = None
    message: str = ""
    #: Whether the limit was actually evaluated against a number.
    #:
    #: A check can allow an order without having measured anything — the
    #: liquidity limit deliberately lets a fresh listing through, because
    #: blocking every instrument with no ADV history would be wrong. But
    #: reporting that as "ok" is the failure this console has been chasing all
    #: along: it renders identically to a limit that was checked and cleared,
    #: so nobody can tell protection from its absence.
    measured: bool = True

    def format(self) -> str:
        if not self.measured:
            return f"    [ --  ] {self.name:<24} {self.message}"
        mark = "ok  " if self.passed else "FAIL"
        if self.observed is None:
            return f"    [{mark}] {self.name:<24} {self.message}"
        return (
            f"    [{mark}] {self.name:<24} {self.observed:>14,.2f} "
            f"vs {self.threshold:>14,.2f}  {self.message}"
        )


@dataclass(frozen=True)
class ProposedOrder:
    """An order the risk engine is being asked to approve."""

    strategy_id: str
    instrument_id: InstrumentId
    quantity: Decimal
    price: Decimal
    #: Signed: positive to buy, negative to sell.
    multiplier: Decimal = Decimal(1)
    #: Correlation group. Names sharing one count as a single bet.
    cluster: str = ""
    #: Sessions until this name's announced results meeting, when one is known.
    #:
    #: `None` is "not measured", not "nothing announced" — the calendar may be
    #: absent, stale, or unable to resolve the symbol, and a check that treated
    #: those as "no event" would wave through exactly the order it exists to
    #: catch. The engine reports it as unmeasured instead.
    days_to_results: int | None = None

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("proposed order has zero quantity")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")

    @property
    def notional(self) -> Decimal:
        """Absolute cash value of the order."""
        return abs(self.quantity) * self.price * self.multiplier

    @property
    def signed_notional(self) -> Decimal:
        return self.quantity * self.price * self.multiplier


@dataclass(frozen=True)
class PortfolioState:
    """A snapshot the engine evaluates against.

    Passed in rather than queried, so the engine stays pure and testable and
    holds no connection to the trading system it polices.
    """

    equity: Decimal
    cash: Decimal
    #: Highest equity ever reached. The drawdown denominator.
    peak_equity: Decimal
    #: Equity at the start of the current session, for the daily loss limit.
    day_start_equity: Decimal

    #: Signed notional per instrument.
    positions: dict[InstrumentId, Decimal] = field(default_factory=dict)
    #: Signed notional per correlation cluster.
    clusters: dict[str, Decimal] = field(default_factory=dict)
    #: Live orders already at the venue, for the rate limit.
    open_orders: int = 0
    orders_this_minute: int = 0
    #: Most recent traded price per instrument, for the fat-finger band.
    last_prices: dict[InstrumentId, Decimal] = field(default_factory=dict)
    #: 20-day average daily traded value, for the liquidity cap.
    adv: dict[InstrumentId, Decimal] = field(default_factory=dict)

    @property
    def gross_exposure(self) -> Decimal:
        return sum((abs(v) for v in self.positions.values()), start=Decimal(0))

    @property
    def net_exposure(self) -> Decimal:
        return sum(self.positions.values(), start=Decimal(0))

    @property
    def drawdown(self) -> Decimal:
        """Current drawdown from peak, as a negative fraction."""
        if self.peak_equity <= 0:
            return Decimal(0)
        return (self.equity - self.peak_equity) / self.peak_equity

    @property
    def day_pnl_pct(self) -> Decimal:
        """Session P&L as a fraction of the session's opening equity."""
        if self.day_start_equity <= 0:
            return Decimal(0)
        return (self.equity - self.day_start_equity) / self.day_start_equity


@dataclass(frozen=True)
class RiskLimits:
    """Every threshold, in one auditable place.

    Owned by the risk engine. Strategies have no write path to these — the
    `risk_limits` table records who set each value and when (§8).
    """

    # ── layer 1: per order ──────────────────────────────────────────────────
    max_order_notional: Decimal = Decimal(500_000)
    max_position_pct: Decimal = Decimal("0.10")
    #: Reject an order priced further than this from the last trade. The
    #: fat-finger guard: a decimal-point slip is caught before the venue sees it.
    price_band_pct: Decimal = Decimal("0.05")
    max_orders_per_minute: int = 20
    max_open_orders: int = 50

    # ── layer 2: portfolio ──────────────────────────────────────────────────
    max_gross_exposure_pct: Decimal = Decimal("1.50")
    max_net_exposure_pct: Decimal = Decimal("1.00")
    max_cluster_pct: Decimal = Decimal("0.30")
    #: Session loss that halts new orders, as a negative fraction.
    daily_loss_limit_pct: Decimal = Decimal("-0.03")
    #: Order may not exceed this fraction of the instrument's ADV.
    max_adv_participation: Decimal = Decimal("0.05")

    #: Sessions before an announced results meeting during which a position may
    #: not be *increased*. Zero disables the check.
    #:
    #: **Only increases.** Reducing or closing into an event is always allowed,
    #: and a rule that blocked it would trap a book in the position it was
    #: trying to get out of — the opposite of a risk control.
    #:
    #: Two sessions rather than the day itself: a results print moves the open,
    #: so an order placed the evening before is already exposed. This system
    #: decides on a close and fills on the next bar, which spends one of them.
    #:
    #: The reason this is a limit and not a strategy's business: an earnings
    #: gap is not the volatility GARCH forecasts from history. The forecast is
    #: built from sessions where nothing was scheduled, so it understates a day
    #: whose distribution is known in advance to be bimodal.
    event_blackout_days: int = 2

    def __post_init__(self) -> None:
        if self.daily_loss_limit_pct >= 0:
            raise ValueError("daily_loss_limit_pct must be negative")
        for name in ("max_order_notional", "max_position_pct", "max_gross_exposure_pct"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class LadderRung:
    """One step of the drawdown ladder."""

    #: Drawdown at which this rung engages, as a negative fraction.
    drawdown_pct: Decimal
    #: Fraction of normal size to scale to. 0 means flat.
    scale_to: Decimal
    #: Whether reaching this rung halts trading entirely.
    halt: bool = False

    def __post_init__(self) -> None:
        if self.drawdown_pct >= 0:
            raise ValueError(f"rung drawdown must be negative, got {self.drawdown_pct}")
        if not 0 <= self.scale_to <= 1:
            raise ValueError(f"scale_to must be in [0, 1], got {self.scale_to}")


@dataclass(frozen=True)
class DrawdownLadder:
    """Pre-committed de-risking — MASTER_PLAN §8.

    The entire purpose is to **remove the operator from the decision at the
    exact moment they are least capable of making it.** Deciding "should I cut?"
    during a 9% drawdown is the worst possible time to think about it; the
    decision belongs to the version of you that wrote these numbers down while
    calm, and it belongs in code so it executes without asking.
    """

    rungs: tuple[LadderRung, ...] = (
        LadderRung(Decimal("-0.05"), Decimal("0.50")),
        LadderRung(Decimal("-0.08"), Decimal("0.25")),
        LadderRung(Decimal("-0.10"), Decimal("0.00"), halt=True),
    )

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("a ladder with no rungs is not a ladder")
        ordered = sorted(self.rungs, key=lambda r: r.drawdown_pct, reverse=True)
        if list(ordered) != list(self.rungs):
            raise ValueError("rungs must be ordered from shallowest to deepest drawdown")
        scales = [r.scale_to for r in self.rungs]
        if scales != sorted(scales, reverse=True):
            raise ValueError("deeper rungs must scale to a smaller size, never larger")

    def rung_for(self, drawdown: Decimal) -> LadderRung | None:
        """The deepest rung this drawdown has reached, or None if above them all."""
        engaged = [r for r in self.rungs if drawdown <= r.drawdown_pct]
        return engaged[-1] if engaged else None

    def scale_for(self, drawdown: Decimal) -> Decimal:
        """Position scale at this drawdown. 1 when no rung has engaged."""
        rung = self.rung_for(drawdown)
        return rung.scale_to if rung else Decimal(1)

    def halts_at(self, drawdown: Decimal) -> bool:
        rung = self.rung_for(drawdown)
        return rung.halt if rung else False
