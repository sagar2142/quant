"""Trading a precomputed signal — MASTER_PLAN §6, §13, §17.

**The connector the pipeline was missing.** The factor lab could show that a
composite predicts — IC, quantiles, decay, turnover — and nothing could trade
it. Every other strategy computes its own signal internally, so a new idea
needed a new `Strategy` subclass before it could be backtested at all. A
research loop that ends at "this predicts" and cannot reach "this earns" is
half a loop.

This takes a scored panel and turns it into target weights, so any signal the
lab produces becomes backtestable, gauntlet-testable and paper-tradable without
new code.

**Look-ahead is the whole risk here, and it is guarded twice.**

A precomputed panel is convenient precisely because it holds the entire
history, which is also how a backtest reads the future by accident. Two
defences:

1. Frames carrying forward-return columns are *rejected at construction*.
   `build_factor` returns `fwd_1`, `fwd_5` and friends alongside the signal,
   and passing that frame here would hand the strategy tomorrow's return under
   a column name it could read. This is refused rather than filtered, because
   silently dropping them would let a caller believe they had been used.
2. `generate` reads only rows at or before `view.as_of`, and takes each name's
   most recent score. The signal itself is backward-looking by construction —
   momentum shifts *positive* — so the score stamped at bar *t* was computable
   at *t*.

**Construction is separate from selection**, deliberately. Which names to hold
is a question about the signal; how much of each is a question about risk, and
the plan keeps those apart everywhere else too (§8, §17).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

import numpy as np
import polars as pl

from core.instruments import InstrumentId
from quant.strategies.base import MarketView, Strategy, StrategySpec, TargetWeights

__all__ = ["Construction", "ForwardLeakError", "SignalStrategy"]

#: Columns that must never reach a strategy. `build_factor` attaches these for
#: scoring and they are the future by definition.
FORWARD_PREFIX = "fwd_"

#: Required columns of a scored panel.
REQUIRED = ("event_time", "symbol", "signal")

#: Volatility lookback for inverse-volatility construction. A quarter: long
#: enough to be stable, short enough to react within a regime.
VOL_WINDOW = 60

#: Below this a name has no usable volatility estimate and is equal-weighted
#: rather than sized off noise.
MIN_VOL_BARS = 20


class ForwardLeakError(ValueError):
    """A scored panel carried forward returns into a strategy.

    Raised, never filtered. A caller who passed the frame straight from
    `build_factor` needs to know the columns were there — quietly dropping them
    would leave them believing the backtest had used a frame it had not.
    """

    def __init__(self, columns: list[str]) -> None:
        super().__init__(
            f"scored panel carries forward-return columns {columns}. Those are "
            "the future: drop them before backtesting. "
            "`frame.select('event_time', 'symbol', 'signal')`"
        )


class Construction(str, Enum):
    """How selected names are weighted once chosen.

    Selection answers *which*; construction answers *how much*. Keeping them
    apart means a signal can be tested against several risk treatments without
    touching the signal.
    """

    EQUAL = "equal"
    SCORE = "score"
    INVERSE_VOL = "inverse_vol"

    @property
    def description(self) -> str:
        return {
            Construction.EQUAL: (
                "Equal weight. The honest default: it makes no claim the "
                "signal cannot support, and a signal whose edge disappears "
                "under equal weighting never had one."
            ),
            Construction.SCORE: (
                "Proportional to cross-sectional rank. Expresses conviction, "
                "and concentrates into whatever the signal is most wrong about "
                "when it is wrong."
            ),
            Construction.INVERSE_VOL: (
                "Inverse realised volatility, so each position contributes "
                "comparable risk rather than comparable rupees. Usually the "
                "largest single improvement to a naive cross-sectional book."
            ),
        }[self]


class SignalStrategy(Strategy):
    """Hold the top fraction of a precomputed signal.

    Args:
        scores: Panel of `event_time`, `symbol`, `signal`. Forward-return
            columns are refused.
        top_fraction: Share of the scored universe to hold.
        construction: How the held names are weighted.
        gross: Total exposure.
        max_position: Per-name cap as a fraction of NAV, applied by the base
            class after construction.
        name: Recorded on the spec, so an experiment row says which signal was
            traded rather than "signal".
    """

    def __init__(
        self,
        scores: pl.DataFrame,
        *,
        top_fraction: Decimal = Decimal("0.2"),
        construction: Construction = Construction.EQUAL,
        gross: Decimal = Decimal(1),
        max_position: Decimal = Decimal("0.10"),
        name: str = "signal",
    ) -> None:
        leaked = [c for c in scores.columns if c.startswith(FORWARD_PREFIX)]
        if leaked:
            raise ForwardLeakError(leaked)
        missing = [c for c in REQUIRED if c not in scores.columns]
        if missing:
            raise ValueError(f"scored panel is missing {missing}")
        if not 0 < float(top_fraction) <= 1:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

        super().__init__(
            StrategySpec(
                name=f"signal:{name}",
                universe="dynamic",
                timeframe="1d",
                parameters={
                    "signal": name,
                    "top_fraction": str(top_fraction),
                    "construction": construction.value,
                },
                # One bar: the score is precomputed, so the strategy needs no
                # warm-up of its own. Whatever lookback the *signal* required
                # is already spent — its early rows are simply absent.
                lookback=1,
                max_position=max_position,
                max_gross=gross,
            )
        )
        self.scores = scores.sort("event_time")
        self.top_fraction = top_fraction
        self.construction = construction
        self.gross = gross

    @staticmethod
    def _symbol_to_instrument(view: MarketView) -> dict[str, InstrumentId]:
        """Ticker to instrument id, as the view itself reports it.

        **The scored panel is keyed by symbol and the universe by instrument
        id**, and those are different strings — "TCS" against
        "NSE:INE467B01029". Comparing them directly matches nothing and a
        strategy that silently holds nothing looks exactly like a signal with
        no opportunities.

        Resolved from the view rather than a static table so that a symbol
        which has worn more than one ISIN maps to whichever the engine is
        currently trading (§1.1). 344 NSE symbols have.
        """
        if view.history.is_empty():
            return {}
        pairs = view.history.select("symbol", "instrument_id").unique()
        tradable = {str(i) for i in view.universe}
        return {
            str(symbol): InstrumentId(str(instrument_id))
            for symbol, instrument_id in zip(pairs["symbol"], pairs["instrument_id"], strict=True)
            if str(instrument_id) in tradable
        }

    def scores_at(self, view: MarketView) -> dict[InstrumentId, float]:
        """Each name's most recent score at or before the decision time.

        The second look-ahead guard. Filtering on `as_of` rather than taking
        the whole frame is what makes a precomputed panel safe to hold.
        """
        visible = self.scores.filter(pl.col("event_time") <= view.as_of)
        if visible.is_empty():
            return {}

        latest = (
            visible.group_by("symbol")
            .agg(pl.col("signal").last().alias("signal"))
            .filter(pl.col("signal").is_finite())
        )
        resolved = self._symbol_to_instrument(view)
        return {
            resolved[symbol]: float(score)
            for symbol, score in zip(latest["symbol"], latest["signal"], strict=True)
            if symbol in resolved
        }

    def _realised_vol(self, view: MarketView, instrument_id: InstrumentId) -> float | None:
        series = view.series(instrument_id)
        if series.height < MIN_VOL_BARS:
            return None
        closes = np.asarray(
            [float(c) for c in series["close"].tail(VOL_WINDOW).to_list()], dtype=np.float64
        )
        returns = closes[1:] / closes[:-1] - 1
        deviation = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
        return deviation if deviation > 0 else None

    def _weights(
        self, view: MarketView, chosen: list[tuple[InstrumentId, float]]
    ) -> dict[InstrumentId, Decimal]:
        """Turn selected names into fractions of NAV."""
        if self.construction is Construction.SCORE:
            # Rank, not raw score: a signal's units are arbitrary and one
            # extreme value would otherwise take most of the book.
            ranks = {name: float(i + 1) for i, (name, _) in enumerate(reversed(chosen))}
            total = sum(ranks.values())
            return {name: self.gross * Decimal(str(rank / total)) for name, rank in ranks.items()}

        if self.construction is Construction.INVERSE_VOL:
            inverse = {}
            for name, _ in chosen:
                deviation = self._realised_vol(view, name)
                # A name with no usable estimate takes the median weight rather
                # than being dropped: the signal chose it, and construction is
                # not the place to overrule that.
                inverse[name] = 1.0 / deviation if deviation else 0.0
            usable = [v for v in inverse.values() if v > 0]
            if usable:
                fallback = float(np.median(usable))
                inverse = {k: (v if v > 0 else fallback) for k, v in inverse.items()}
                total = sum(inverse.values())
                return {
                    name: self.gross * Decimal(str(value / total))
                    for name, value in inverse.items()
                }

        weight = self.gross / Decimal(len(chosen))
        return {name: weight for name, _ in chosen}

    def generate(self, view: MarketView) -> TargetWeights:
        scores = self.scores_at(view)
        if not scores:
            return TargetWeights(view.as_of, {})

        # Deterministic ordering: ties break on the instrument id, so two runs
        # over the same data hold the same book (§14.1.1).
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        count = max(1, int(len(ranked) * float(self.top_fraction)))
        chosen = ranked[:count]

        return TargetWeights(view.as_of, self._weights(view, chosen))
