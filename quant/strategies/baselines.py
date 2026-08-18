"""Baseline strategies — MASTER_PLAN §M3, Part 6.

These exist to validate the *engine*, not to make money. Their value is that
their correct behaviour is known in advance, so when the backtester disagrees
with hand arithmetic the backtester is wrong.

`BuyAndHold` is the calibration instrument: its result can be computed on paper
to the rupee, including costs, and it must also *pass* the validation gauntlet
(§M5 gate) — a gauntlet that rejects buy-and-hold is broken.

`SmaCrossover` is the classic trend baseline, and the plan is blunt that it is
not live-worthy (§6). It is here to exercise turnover, whipsaw and cost drag,
which is precisely what it is good at demonstrating.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import polars as pl

from core.instruments import InstrumentId
from quant.strategies.base import MarketView, Strategy, StrategySpec, TargetWeights

__all__ = [
    "BuyAndHold",
    "CrossSectionalMomentum",
    "EqualWeight",
    "RandomEntry",
    "SmaCrossover",
]


class BuyAndHold(Strategy):
    """Hold a fixed equal-weight allocation from the first bar onward.

    Emits the same target every bar. The engine's rebalance threshold means
    that becomes one initial purchase and then near-silence, which is the
    intended behaviour: any trade after the first is drift correction.
    """

    def __init__(self, universe_size: int | None = None, gross: Decimal = Decimal(1)) -> None:
        super().__init__(
            StrategySpec(
                name="buy_and_hold",
                universe="fixed",
                timeframe="1d",
                parameters={"gross": str(gross)},
                lookback=1,
                max_position=Decimal(1),
                max_gross=gross,
            )
        )
        self.universe_size = universe_size
        self.gross = gross

    def generate(self, view: MarketView) -> TargetWeights:
        names = [n for n in view.universe if n in view.latest_close()]
        if not names:
            return TargetWeights(view.as_of, {})
        count = self.universe_size or len(names)
        weight = self.gross / Decimal(count)
        return TargetWeights(view.as_of, dict.fromkeys(names, weight))


class EqualWeight(BuyAndHold):
    """Alias for readability where the rebalancing intent matters more than
    the buy-and-hold framing."""

    def __init__(self, gross: Decimal = Decimal(1)) -> None:
        super().__init__(gross=gross)
        self.spec = StrategySpec(
            name="equal_weight",
            universe="fixed",
            timeframe="1d",
            parameters={"gross": str(gross)},
            lookback=1,
            max_position=Decimal(1),
            max_gross=gross,
        )


class SmaCrossover(Strategy):
    """Long when fast SMA is above slow SMA, flat otherwise.

    Deliberately simple and deliberately not live-worthy. Its purpose is to
    demonstrate what the plan claims: that turnover multiplied by round-trip
    cost destroys marginal edges (§7.1). Run it with 1x and 3x costs and the
    difference is the lesson.
    """

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        gross: Decimal = Decimal(1),
        allow_short: bool = False,
    ) -> None:
        if fast >= slow:
            raise ValueError(f"fast window ({fast}) must be shorter than slow ({slow})")
        super().__init__(
            StrategySpec(
                name="sma_crossover",
                universe="fixed",
                timeframe="1d",
                parameters={"fast": fast, "slow": slow, "allow_short": allow_short},
                # +1 so the slow average has a full window on the first decision.
                lookback=slow + 1,
                max_position=Decimal(1),
                max_gross=gross,
            )
        )
        self.fast = fast
        self.slow = slow
        self.gross = gross
        self.allow_short = allow_short

    def generate(self, view: MarketView) -> TargetWeights:
        names = list(view.universe)
        if not names:
            return TargetWeights(view.as_of, {})

        signals: dict[InstrumentId, Decimal] = {}
        for name in names:
            series = view.series(name)
            if series.height < self.slow:
                continue
            # Plain Python over a short window: Polars aggregates return a
            # loosely-typed union, and the window is small enough that clarity
            # beats vectorisation here.
            window = [float(c) for c in series["close"].tail(self.slow).to_list()]
            slow_ma = sum(window) / len(window)
            fast_ma = sum(window[-self.fast :]) / self.fast
            if fast_ma > slow_ma:
                signals[name] = Decimal(1)
            elif self.allow_short:
                signals[name] = Decimal(-1)

        if not signals:
            return TargetWeights(view.as_of, {})

        weight = self.gross / Decimal(len(signals))
        return TargetWeights(view.as_of, {k: v * weight for k, v in signals.items()})


class CrossSectionalMomentum(Strategy):
    """Long the top decile by trailing return, short the bottom if permitted.

    The plan's strongest first live candidate (§6): documented across decades
    and markets, monthly rebalance keeps turnover survivable against Indian
    costs, and high breadth puts the Fundamental Law to work
    (``IR ~= IC * sqrt(breadth)``).

    The skip window is not decoration. Classic 12-1 momentum omits the most
    recent month because short-horizon reversal runs against momentum there,
    and including it measurably degrades the signal.
    """

    def __init__(
        self,
        lookback_bars: int = 252,
        skip_bars: int = 21,
        top_fraction: Decimal = Decimal("0.2"),
        gross: Decimal = Decimal(1),
        long_only: bool = True,
    ) -> None:
        if skip_bars >= lookback_bars:
            raise ValueError("skip window must be shorter than the lookback")
        super().__init__(
            StrategySpec(
                name="xs_momentum",
                universe="dynamic",
                timeframe="1d",
                parameters={
                    "lookback_bars": lookback_bars,
                    "skip_bars": skip_bars,
                    "top_fraction": str(top_fraction),
                    "long_only": long_only,
                },
                lookback=lookback_bars + 1,
                max_position=Decimal("0.15"),
                max_gross=gross,
            )
        )
        self.lookback_bars = lookback_bars
        self.skip_bars = skip_bars
        self.top_fraction = top_fraction
        self.gross = gross
        self.long_only = long_only

    def generate(self, view: MarketView) -> TargetWeights:
        scores: dict[InstrumentId, float] = {}
        for name in view.universe:
            series = view.series(name)
            if series.height < self.lookback_bars:
                continue
            closes = [float(c) for c in series["close"].to_list()]
            # Return from lookback ago to skip_bars ago — the "12-1" window.
            start = closes[-self.lookback_bars]
            end = closes[-(self.skip_bars + 1)] if self.skip_bars else closes[-1]
            if start <= 0:
                continue
            scores[name] = end / start - 1

        if not scores:
            return TargetWeights(view.as_of, {})

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        count = max(1, int(len(ranked) * float(self.top_fraction)))
        longs = [name for name, _ in ranked[:count]]
        shorts = [] if self.long_only else [name for name, _ in ranked[-count:]]

        weights: dict[InstrumentId, Decimal] = {}
        if longs:
            leg = self.gross if self.long_only else self.gross / 2
            per_name = leg / Decimal(len(longs))
            for name in longs:
                weights[name] = per_name
        if shorts:
            per_name = (self.gross / 2) / Decimal(len(shorts))
            for name in shorts:
                weights[name] = -per_name
        return TargetWeights(view.as_of, weights)


class RandomEntry(Strategy):
    """Random names, matched holding period and gross exposure — §5.4 test 10.

    The placebo. If a coin flip trading the same instruments at the same cadence
    with the same exposure does about as well, then the market did the work and
    the signal contributed nothing. This is the control group, and a strategy
    without one is an anecdote.

    **Purity is the hard part.** A strategy carrying a mutable RNG returns
    different weights for the same view, which breaks the M3 reproducibility
    gate and quietly invalidates the shuffle-future test. So the draw is derived
    from `(seed, bar_bucket)` on every call and no state survives it: the same
    view always yields the same names.

    Holding period is matched by *quantising the bar count* rather than by
    counting elapsed time. Sessions are unevenly spaced — weekends, holidays —
    so a wall-clock bucket would silently vary the holding period across the
    sample and make the comparison unfair in an uncontrolled direction.
    """

    def __init__(
        self,
        *,
        seed: int,
        n_names: int,
        hold_bars: int = 1,
        gross: Decimal = Decimal(1),
        lookback: int = 1,
    ) -> None:
        if n_names < 1:
            raise ValueError("n_names must be at least 1")
        if hold_bars < 1:
            raise ValueError("hold_bars must be at least 1")
        super().__init__(
            StrategySpec(
                name="random_entry",
                universe="fixed",
                timeframe="1d",
                parameters={"seed": seed, "n_names": n_names, "hold_bars": hold_bars},
                # Matched to the strategy under test, so the placebo starts
                # trading on the same bar and neither gets a head start.
                lookback=lookback,
                max_position=Decimal(1),
                max_gross=gross,
            )
        )
        self.seed = seed
        self.n_names = n_names
        self.hold_bars = hold_bars
        self.gross = gross

    def generate(self, view: MarketView) -> TargetWeights:
        # Sorted, so the draw does not depend on dict or set ordering.
        names = sorted(n for n in view.universe if n in view.latest_close())
        if not names:
            return TargetWeights(view.as_of, {})

        bucket = view.bar_count() // self.hold_bars
        rng = np.random.default_rng([self.seed, bucket])
        count = min(self.n_names, len(names))
        chosen = rng.choice(len(names), size=count, replace=False)

        weight = self.gross / Decimal(count)
        return TargetWeights(view.as_of, {names[int(i)]: weight for i in chosen})


def sma(frame: pl.DataFrame, window: int, column: str = "close") -> float | None:
    """Simple moving average of the last `window` rows. None if too short."""
    if frame.height < window:
        return None
    values = [float(v) for v in frame[column].tail(window).to_list()]
    return sum(values) / len(values) if values else None
