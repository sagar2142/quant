"""Cost model interface — MASTER_PLAN §7.

This module decides which strategies are viable, so it is built before any
strategy exists. At ~0.22% round trip on NSE delivery, a weekly-rebalancing
strategy needs ~11% gross annual return merely to break even, while a monthly
one needs ~2.6%. That single fact rules out most of the strategy space and
pushes research toward low-turnover, cross-sectional designs. The cost model is
a *constraint on what may be built*, not a detail applied afterwards.

**Decimal throughout, enforced by lint.** A cost is money, and money is exact:
these numbers are compared against a broker contract note during reconciliation
(§14.1.2). Statistics elsewhere are float64; nothing here is.

**Costs stay itemised.** `total` is available but never stored alone. When
realised costs diverge from modelled ones — and they will — the itemisation is
what tells you whether the error is in the fee schedule, the tax treatment or
the slippage estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol, runtime_checkable

from core.instruments import Instrument
from core.orders import Side

__all__ = ["PAISA", "CostBreakdown", "CostModel", "TradeContext", "quantize_money"]

#: Indian and US venues both settle to two decimal places.
PAISA = Decimal("0.01")


def quantize_money(value: Decimal) -> Decimal:
    """Round to two decimals, half-up.

    Half-up rather than banker's rounding because that is what Indian brokers
    apply on contract notes, and reconciliation compares against those.
    """
    return value.quantize(PAISA, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class TradeContext:
    """Everything a cost model needs about one prospective trade."""

    instrument: Instrument
    side: Side
    quantity: Decimal
    price: Decimal

    #: Intraday (squared off same session) attracts different STT in India.
    intraday: bool = False

    #: Average daily traded value, for the square-root impact term (§7.5).
    adv_value: Decimal | None = None
    #: Quoted bid-ask spread in currency units.
    spread: Decimal | None = None
    #: Daily return volatility as a fraction, e.g. Decimal("0.02") for 2%.
    volatility: Decimal | None = None

    #: Whether this trade closes a position, which triggers Indian DP charges.
    is_exit: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}")

    @property
    def notional(self) -> Decimal:
        """Cash value of the trade, including any contract multiplier."""
        return self.price * self.quantity * self.instrument.multiplier


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised trading costs. Every field is a positive cost to the trader."""

    brokerage: Decimal = Decimal(0)
    stt: Decimal = Decimal(0)
    exchange_fee: Decimal = Decimal(0)
    regulatory_fee: Decimal = Decimal(0)
    stamp_duty: Decimal = Decimal(0)
    gst: Decimal = Decimal(0)
    depository_fee: Decimal = Decimal(0)
    slippage: Decimal = Decimal(0)
    #: Signed: negative means funding was *received* (§7.3).
    funding: Decimal = Decimal(0)

    @property
    def total(self) -> Decimal:
        return quantize_money(
            self.brokerage
            + self.stt
            + self.exchange_fee
            + self.regulatory_fee
            + self.stamp_duty
            + self.gst
            + self.depository_fee
            + self.slippage
            + self.funding
        )

    @property
    def explicit(self) -> Decimal:
        """Fees a broker would itemise. Excludes slippage, which is implicit."""
        return quantize_money(self.total - self.slippage - self.funding)

    def bps_of(self, notional: Decimal) -> Decimal:
        """Total cost in basis points of notional.

        The number that decides whether a strategy is viable: compare it to the
        expected gross edge per trade.
        """
        if notional == 0:
            return Decimal(0)
        return (self.total / notional) * Decimal(10_000)

    def __add__(self, other: CostBreakdown) -> CostBreakdown:
        return CostBreakdown(
            brokerage=self.brokerage + other.brokerage,
            stt=self.stt + other.stt,
            exchange_fee=self.exchange_fee + other.exchange_fee,
            regulatory_fee=self.regulatory_fee + other.regulatory_fee,
            stamp_duty=self.stamp_duty + other.stamp_duty,
            gst=self.gst + other.gst,
            depository_fee=self.depository_fee + other.depository_fee,
            slippage=self.slippage + other.slippage,
            funding=self.funding + other.funding,
        )

    def format(self) -> str:
        rows = [
            ("brokerage", self.brokerage),
            ("STT", self.stt),
            ("exchange", self.exchange_fee),
            ("SEBI", self.regulatory_fee),
            ("stamp duty", self.stamp_duty),
            ("GST", self.gst),
            ("depository", self.depository_fee),
            ("slippage", self.slippage),
            ("funding", self.funding),
        ]
        lines = [f"  {name:<12} {value:>12.2f}" for name, value in rows if value != 0]
        lines.append(f"  {'TOTAL':<12} {self.total:>12.2f}")
        return "\n".join(lines)


@runtime_checkable
class CostModel(Protocol):
    """One cost model per (venue, product, account type).

    Implementations are pure: same context in, same breakdown out, no I/O
    (§14.1.6). That is what makes cost sensitivity testable by simply scaling
    the model.
    """

    @property
    def name(self) -> str:
        """Identifies the fee schedule. Recorded on the experiment row.

        Read-only deliberately: implementations expose it either as a dataclass
        field or as a computed property (`ScaledCostModel` derives it), and a
        settable declaration would reject the latter.
        """

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        """Total cost of executing this trade."""


@dataclass(frozen=True)
class ScaledCostModel:
    """Wraps a model and multiplies every component.

    Gauntlet test 7 (§5.4) requires a strategy to survive **3x** modelled costs.
    Wrapping rather than editing rates keeps the sensitivity sweep honest: the
    underlying schedule stays a single source of truth.
    """

    inner: CostModel
    multiplier: Decimal

    def __post_init__(self) -> None:
        if self.multiplier < 0:
            raise ValueError("cost multiplier cannot be negative")

    @property
    def name(self) -> str:
        return f"{self.inner.name}x{self.multiplier}"

    def cost(self, ctx: TradeContext) -> CostBreakdown:
        base = self.inner.cost(ctx)
        m = self.multiplier
        return CostBreakdown(
            brokerage=base.brokerage * m,
            stt=base.stt * m,
            exchange_fee=base.exchange_fee * m,
            regulatory_fee=base.regulatory_fee * m,
            stamp_duty=base.stamp_duty * m,
            gst=base.gst * m,
            depository_fee=base.depository_fee * m,
            slippage=base.slippage * m,
            funding=base.funding * m,
        )
