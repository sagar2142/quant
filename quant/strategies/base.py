"""Strategy interface — MASTER_PLAN §13.

A strategy is a **pure function of observable history**. It receives a view of
the world as of a decision time and returns target weights. It cannot place
orders, cannot see its own P&L, and cannot reach the trading plane at all
(§3.2). Everything downstream — sizing, risk, execution — is somebody else's
job, deliberately.

**Targets, not orders.** A strategy expresses *"I want 3% of NAV in this name"*,
not *"buy 47 shares"*. Converting a target to an order requires the current
position, current price, lot size and risk limits, none of which the strategy
should know. This separation is what lets the same strategy run unchanged in
backtest, paper and live.

**Purity is enforced, not requested.** `generate` takes the observable window
and returns weights. No I/O, no clock reads, no mutable state that survives a
call. That is what makes a backtest reproducible (§M3 gate) and what allows the
shuffle-future test (§5.4 test 2) to be meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import polars as pl

from core.clock import DecisionTime
from core.instruments import InstrumentId

__all__ = ["MarketView", "Strategy", "StrategySpec", "TargetWeights"]


@dataclass(frozen=True)
class StrategySpec:
    """Everything needed to reproduce a strategy's behaviour (§13).

    Stored verbatim on the experiment row. If two runs share a spec, a data
    version and a seed, they must produce identical numbers — that is the M3
    gate.
    """

    name: str
    universe: str
    timeframe: str
    parameters: dict[str, Any] = field(default_factory=dict)

    #: Longest history the strategy needs, in bars. The engine will not call
    #: `generate` until this much data is observable, so a strategy never has
    #: to defend against a short window.
    lookback: int = 1

    #: Maximum absolute weight in any single name, as a fraction of NAV.
    max_position: Decimal = Decimal("0.10")

    #: Gross exposure cap. 1.0 is fully invested and unlevered.
    max_gross: Decimal = Decimal("1.0")

    version: int = 1

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be at least 1 bar")
        if self.max_position <= 0:
            raise ValueError("max_position must be positive")
        if self.max_gross <= 0:
            raise ValueError("max_gross must be positive")

    def fingerprint(self) -> dict[str, Any]:
        """Serialisable identity, for the experiment registry."""
        return {
            "name": self.name,
            "version": self.version,
            "universe": self.universe,
            "timeframe": self.timeframe,
            "parameters": dict(sorted(self.parameters.items())),
            "lookback": self.lookback,
            "max_position": str(self.max_position),
            "max_gross": str(self.max_gross),
        }


@dataclass(frozen=True)
class MarketView:
    """What a strategy may see at one decision point.

    Constructed by the engine from `receive_time <= as_of` data only. There is
    no accessor for anything later, so a strategy cannot read the future even
    by mistake (§14.1.4).
    """

    as_of: DecisionTime
    #: Long-format history: event_time, instrument_id, open, high, low, close, volume.
    history: pl.DataFrame
    #: Universe members at this decision point.
    universe: tuple[InstrumentId, ...]

    def closes(self) -> pl.DataFrame:
        """Wide close-price matrix: one row per timestamp, one column per name.

        The natural shape for cross-sectional work.
        """
        if self.history.is_empty():
            return pl.DataFrame()
        return self.history.pivot(on="instrument_id", index="event_time", values="close").sort(
            "event_time"
        )

    def series(self, instrument_id: InstrumentId) -> pl.DataFrame:
        """One instrument's history, ascending by time."""
        return self.history.filter(pl.col("instrument_id") == instrument_id).sort("event_time")

    def latest_close(self) -> dict[InstrumentId, float]:
        """Most recent observable close per instrument.

        float, not Decimal: these feed statistics, not the ledger (§14.1.2).
        """
        if self.history.is_empty():
            return {}
        latest = (
            self.history.sort("event_time").group_by("instrument_id").agg(pl.col("close").last())
        )
        return dict(
            zip(
                latest["instrument_id"].to_list(),
                latest["close"].to_list(),
                strict=True,
            )
        )

    def bar_count(self) -> int:
        """Distinct timestamps observed. The engine uses this to honour lookback."""
        if self.history.is_empty():
            return 0
        return self.history["event_time"].n_unique()


@dataclass(frozen=True)
class TargetWeights:
    """Desired portfolio as fractions of NAV.

    Positive is long, negative is short, absent is flat. Weights need not sum
    to 1: the remainder is cash, and a long-short book sums near zero.
    """

    as_of: DecisionTime
    weights: dict[InstrumentId, Decimal]

    def __post_init__(self) -> None:
        for instrument_id, weight in self.weights.items():
            if not weight.is_finite():
                raise ValueError(f"non-finite weight for {instrument_id}: {weight}")

    @property
    def gross(self) -> Decimal:
        return sum((abs(w) for w in self.weights.values()), start=Decimal(0))

    @property
    def net(self) -> Decimal:
        return sum(self.weights.values(), start=Decimal(0))

    def clipped(self, max_position: Decimal, max_gross: Decimal) -> TargetWeights:
        """Enforce the spec's exposure limits.

        Per-name clipping first, then proportional scaling if gross still
        exceeds the cap. Scaling preserves the strategy's *relative* views,
        which is what it actually expressed an opinion about.
        """
        clipped = {
            k: max(-max_position, min(max_position, w)) for k, w in self.weights.items() if w != 0
        }
        gross = sum((abs(w) for w in clipped.values()), start=Decimal(0))
        if gross > max_gross and gross > 0:
            scale = max_gross / gross
            clipped = {k: w * scale for k, w in clipped.items()}
        return TargetWeights(self.as_of, clipped)


class Strategy(ABC):
    """Base class. Subclasses implement `generate` and nothing else.

    There is deliberately no `on_fill`, no `on_order`, no portfolio handle. A
    strategy that can see its own fills starts making decisions about its own
    P&L, which is the trading plane's job and is not reproducible from data
    alone.
    """

    def __init__(self, spec: StrategySpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @abstractmethod
    def generate(self, view: MarketView) -> TargetWeights:
        """Target weights given everything observable at `view.as_of`.

        Must be pure: same view in, same weights out, no I/O and no state
        carried between calls (§14.1.6).
        """

    def __call__(self, view: MarketView) -> TargetWeights:
        """Generate and clip to the spec's limits.

        The engine calls this rather than `generate`, so limits are applied
        even if a subclass forgets them.
        """
        raw = self.generate(view)
        return raw.clipped(self.spec.max_position, self.spec.max_gross)
