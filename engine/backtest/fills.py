"""Fill simulation — MASTER_PLAN §7.6, §14.

**The single most common source of fake backtest profit** is filling at the
close of the same bar whose close generated the signal. It is not a small
optimism; on a mean-reversion strategy it can invent the entire edge, because
the signal fires precisely when the price is extreme and the fill captures that
extreme.

This module makes that impossible by construction rather than by convention:
`simulate` receives only the *execution bar*, which the engine guarantees is
strictly after the decision bar. There is no parameter through which the
decision bar's prices could be passed.

Legitimate fills for a daily strategy deciding at bar T:

    next open       open of T+1                    conservative, honest
    next VWAP       typical price of T+1           models a sliced order
    next close      close of T+1                   models an MOC order

Every model applies slippage in the direction that hurts: buys fill above the
reference, sells below. Slippage that helps is a modelling error, not luck.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from core.instruments import Instrument
from core.orders import Side
from engine.costs.model import CostModel, TradeContext

__all__ = [
    "ExecutionBar",
    "FillModel",
    "NextCloseFill",
    "NextOpenFill",
    "NextVwapFill",
    "NoLiquidityError",
]


class NoLiquidityError(RuntimeError):
    """The execution bar cannot support this trade.

    Raised rather than silently filling (§14.1.5). A backtest that fills into a
    zero-volume bar is trading against nobody, and a strategy that depends on
    such fills will look profitable and be untradeable.
    """

    def __init__(self, instrument_id: str, wanted: Decimal, available: Decimal) -> None:
        super().__init__(f"{instrument_id}: wanted {wanted} but bar supports only {available}")


@dataclass(frozen=True)
class ExecutionBar:
    """The bar a trade executes into. Always strictly after the decision bar."""

    instrument: Instrument
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        if self.low <= 0:
            raise ValueError(f"non-positive low: {self.low}")
        if self.high < self.low:
            raise ValueError(f"high {self.high} below low {self.low}")

    @property
    def typical(self) -> Decimal:
        """(H+L+C)/3 — a reasonable stand-in for VWAP without tick data."""
        return (self.high + self.low + self.close) / 3

    @property
    def is_tradable(self) -> bool:
        """A zero-volume or zero-range bar is not a real trading opportunity."""
        return self.volume > 0 and self.high > self.low


@dataclass(frozen=True)
class SimulatedFill:
    """Result of filling into one bar."""

    price: Decimal
    quantity: Decimal
    side: Side


class FillModel(ABC):
    """Converts a desired trade into an achievable one.

    Args:
        max_participation: Cap on the fraction of the bar's volume this order
            may consume. 10% is already aggressive for a daily bar; anything
            higher is a claim that the market would have absorbed you without
            moving, which is exactly what the impact model says is false.
    """

    def __init__(
        self,
        cost_model: CostModel,
        max_participation: Decimal = Decimal("0.10"),
    ) -> None:
        if not 0 < max_participation <= 1:
            raise ValueError("max_participation must be in (0, 1]")
        self.cost_model = cost_model
        self.max_participation = max_participation

    @abstractmethod
    def reference_price(self, bar: ExecutionBar) -> Decimal:
        """The price this model fills at, before slippage."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    def simulate(
        self,
        bar: ExecutionBar,
        side: Side,
        quantity: Decimal,
        *,
        allow_partial: bool = True,
    ) -> SimulatedFill:
        """Fill `quantity` into `bar`.

        Args:
            allow_partial: When True, an oversized order is trimmed to the
                participation cap. When False it raises, which is the right
                behaviour for a strategy whose sizing assumes full execution.

        Raises:
            NoLiquidityError: if the bar is untradable, or the order exceeds
                the participation cap and partials are disallowed.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if not bar.is_tradable:
            raise NoLiquidityError(bar.instrument.instrument_id, quantity, bar.volume)

        capacity = bar.volume * self.max_participation
        if quantity > capacity:
            if not allow_partial:
                raise NoLiquidityError(bar.instrument.instrument_id, quantity, capacity)
            quantity = capacity

        reference = self.reference_price(bar)
        price = self._apply_slippage(bar, side, quantity, reference)
        # A fill outside the bar's own range never happened.
        price = max(bar.low, min(bar.high, price))
        return SimulatedFill(
            price=bar.instrument.round_to_tick(price), quantity=quantity, side=side
        )

    def _apply_slippage(
        self, bar: ExecutionBar, side: Side, quantity: Decimal, reference: Decimal
    ) -> Decimal:
        """Move the price against the trader, always.

        Slippage is expressed per share here so it can be added to the
        reference; the cost model returns it in currency units for the whole
        order.
        """
        ctx = TradeContext(
            instrument=bar.instrument,
            side=side,
            quantity=quantity,
            price=reference,
            adv_value=bar.volume * bar.typical,
        )
        breakdown = self.cost_model.cost(ctx)
        per_share = breakdown.slippage / (quantity * bar.instrument.multiplier)
        return reference + per_share * side.sign


class NextOpenFill(FillModel):
    """Fill at the next bar's open.

    The default and the most defensible: a decision made on tonight's close is
    acted on at tomorrow's open, which is what a real daily system does.
    """

    @property
    def name(self) -> str:
        return "next_open"

    def reference_price(self, bar: ExecutionBar) -> Decimal:
        return bar.open


class NextVwapFill(FillModel):
    """Fill at the next bar's typical price, modelling a sliced order.

    More favourable than the open on average, and defensible only if you would
    actually slice. Claiming VWAP while sending a single market order at the
    open is optimism, not modelling.
    """

    @property
    def name(self) -> str:
        return "next_vwap"

    def reference_price(self, bar: ExecutionBar) -> Decimal:
        return bar.typical


class NextCloseFill(FillModel):
    """Fill at the next bar's close, modelling a market-on-close order.

    Note this is the *next* bar's close — never the decision bar's, which is
    the look-ahead this module exists to prevent.
    """

    @property
    def name(self) -> str:
        return "next_close"

    def reference_price(self, bar: ExecutionBar) -> Decimal:
        return bar.close
