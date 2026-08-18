"""Gauntlet input generators — MASTER_PLAN §5.4.

`checks.py` knows how to *judge* a statistic. This module knows how to *produce*
one, by re-running the backtest under the conditions each check requires.

**A skipped check is not a passed check, and this module is why several were
being skipped.** `GauntletInputs` leaves tests 8, 9 and 10 optional, and an
optional input that nothing ever fills means the strategy is being judged by
nine tests while the report claims twelve. These generators close that gap:

    8  universe dropout — re-run on random subsets of the universe
    9  regimes         — slice the return series by market condition
    10 placebo         — re-run with random entries at matched exposure

**Test 12 is deliberately absent.** The locked test set is touched once per
strategy, ever (§5.3), so generating it automatically on every validation run
would burn the only untouched evidence that exists. It stays SKIP until someone
asks for it explicitly and records the access — see `ExperimentRepository`.

**Every generator is seeded and reports the seed it used.** A dropout sample
that cannot be reproduced is an anecdote, and the whole point of the gauntlet is
to stop accepting anecdotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from engine.validation.report import (
    array_or_none,
)
from quant.math.metrics.performance import sharpe_ratio

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from core.instruments import InstrumentId

__all__ = [
    "REGIME_MIN_BARS",
    "SamplingSpec",
    "market_proxy",
    "placebo_sharpes",
    "regime_slices",
    "universe_dropout_sharpes",
]

#: A regime slice shorter than this is noise, not evidence about a regime.
#: Roughly a trading quarter.
REGIME_MIN_BARS = 60

#: Trailing window used to classify regimes, in bars. About a quarter, long
#: enough that a single week does not flip the label.
REGIME_WINDOW = 60

#: Fraction of the universe removed in each dropout sample. Large enough to
#: dislodge a result that rests on one or two names, small enough that the
#: remaining book is still the same strategy.
DROPOUT_FRACTION = 0.2


class UniverseRunner(Protocol):
    """Runs the strategy over a given universe and returns its per-bar returns.

    Injected rather than imported: this module must not know which strategy,
    cost model or panel is involved, only how to perturb them.
    """

    def __call__(self, universe: tuple[InstrumentId, ...]) -> npt.NDArray[np.float64]: ...


class SeededRunner(Protocol):
    """Runs a randomised strategy for one seed and returns its per-bar returns."""

    def __call__(self, seed: int) -> npt.NDArray[np.float64]: ...


@dataclass(frozen=True)
class SamplingSpec:
    """How many perturbed runs to do, and from which seed.

    Shared by the dropout and placebo generators because both answer the same
    shape of question — *where does the real result sit in a distribution of
    perturbed ones* — and both are worthless if the distribution cannot be
    regenerated. Keeping the seed next to the sample count makes it awkward to
    record one without the other.
    """

    seed: int
    samples: int
    periods_per_year: int = 252

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")


def universe_dropout_sharpes(
    run: UniverseRunner,
    universe: Sequence[InstrumentId],
    spec: SamplingSpec,
    *,
    drop_fraction: float = DROPOUT_FRACTION,
) -> npt.NDArray[np.float64]:
    """Sharpe from each of many random universe subsets — test 8.

    Answers one question: *does this result survive without its best names?* A
    strategy whose entire edge comes from two lucky tickers looks identical to a
    broad one in aggregate, and looks nothing like it here.

    The check reads the 5th percentile, so `spec.samples` has to be large enough
    for that percentile to mean something. `MIN_DROPOUT_SAMPLES` is the floor
    the check will accept, not a recommendation — at ten samples the 5th
    percentile is just the minimum.

    Returns:
        One Sharpe per subset. Subsets that produced no returns — too few names
        left to trade — are dropped rather than recorded as zero, which would
        drag the percentile toward a number no run actually produced.
    """
    if not 0 < drop_fraction < 1:
        raise ValueError(f"drop_fraction must be in (0, 1), got {drop_fraction}")

    members = tuple(sorted(universe))
    keep = max(2, round(len(members) * (1 - drop_fraction)))
    if keep >= len(members):
        raise ValueError(
            f"universe of {len(members)} is too small to drop {drop_fraction:.0%} — "
            "test 8 cannot distinguish a broad edge from a concentrated one here"
        )

    rng = np.random.default_rng(spec.seed)
    sharpes: list[float] = []
    for _ in range(spec.samples):
        chosen = rng.choice(len(members), size=keep, replace=False)
        subset = tuple(members[int(i)] for i in sorted(chosen))
        returns = array_or_none(run(subset))
        if returns is None:
            continue
        sharpes.append(sharpe_ratio(returns, periods_per_year=spec.periods_per_year))
    return np.asarray(sharpes, dtype=np.float64)


def market_proxy(history: pl.DataFrame, universe: Sequence[InstrumentId]) -> pl.DataFrame:
    """Equal-weight return of the universe, per bar. The regime yardstick.

    Built from the same panel the strategy trades rather than from an index
    feed, deliberately: a strategy trading mid-caps judged against a large-cap
    index is being labelled by the wrong market. This is the market *it*
    actually faces.

    Returns:
        Two columns, `event_time` and `market_return`, ascending. The first bar
        is absent — a return needs a prior close.
    """
    import polars as pl  # noqa: PLC0415 - keeps polars off this module's import path

    wanted = set(universe)
    frame = history.filter(pl.col("instrument_id").is_in(list(wanted)))
    if frame.is_empty():
        return pl.DataFrame({"event_time": [], "market_return": []})

    per_name = frame.sort("event_time").with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("instrument_id") - 1).alias("ret")
    )
    return (
        per_name.drop_nulls("ret")
        .group_by("event_time")
        .agg(pl.col("ret").mean().alias("market_return"))
        .sort("event_time")
    )


def regime_slices(
    returns: npt.ArrayLike,
    market_returns: npt.ArrayLike,
    *,
    window: int = REGIME_WINDOW,
    min_bars: int = REGIME_MIN_BARS,
) -> dict[str, npt.NDArray[np.float64]]:
    """Split a return series by market regime — test 9.

    Three labels, from trailing statistics of the market proxy:

        bull      trailing return over `window` bars is positive
        bear      trailing return over `window` bars is negative
        high_vol  trailing volatility is in the top third of the sample

    **The labels overlap on purpose.** `high_vol` is mostly a subset of `bear`,
    and that is fine: "does this survive a downturn" and "does this survive
    turbulence" are different questions, and forcing a partition would answer
    neither cleanly.

    Trailing windows, not centred ones, so a label never depends on data after
    the bar it labels. Nothing here feeds a trading decision, but a look-ahead
    habit in analysis code eventually becomes one in decision code.

    Returns:
        Regime name to return series, omitting any regime with fewer than
        `min_bars` observations. A Sharpe computed on twelve bars is not
        evidence about a regime, and reporting it as though it were is worse
        than reporting nothing.
    """
    strategy: npt.NDArray[np.float64] = np.asarray(returns, dtype=np.float64).ravel()
    market: npt.NDArray[np.float64] = np.asarray(market_returns, dtype=np.float64).ravel()

    size = min(strategy.size, market.size)
    if size < window + min_bars:
        return {}
    # Aligned on the tail: both series end at the last bar of the backtest, so
    # trimming from the front is what lines them up. Trimming the end instead
    # would pair each strategy return with a different bar's market return.
    strategy = strategy[strategy.size - size :]
    market = market[market.size - size :]

    windows = sliding_window_view(market, window)
    trailing_return = _pad(windows.sum(axis=1), market.size, window)
    trailing_vol = _pad(windows.std(axis=1), market.size, window)

    usable = ~np.isnan(trailing_return)
    if not usable.any():
        return {}

    vol_cut = float(np.nanquantile(trailing_vol, 2 / 3))
    masks = {
        "bull": usable & (trailing_return > 0),
        "bear": usable & (trailing_return < 0),
        "high_vol": usable & (trailing_vol >= vol_cut),
    }
    return {name: strategy[mask] for name, mask in masks.items() if int(mask.sum()) >= min_bars}


def placebo_sharpes(run: SeededRunner, spec: SamplingSpec) -> npt.NDArray[np.float64]:
    """Sharpe from each of many random-entry runs — test 10.

    The control group. Each run uses a distinct seed derived from `seed`, so the
    whole distribution reproduces from one number.

    This is the most expensive generator in the module — one full backtest per
    sample — and it is worth it. The check asks whether the real strategy beats
    the 95th percentile of coin flips, and that percentile cannot be estimated
    from a handful of draws.
    """
    rng = np.random.default_rng(spec.seed)
    # Drawn up front so the seed sequence does not depend on how many runs
    # happen to produce returns.
    seeds = rng.integers(0, 2**31 - 1, size=spec.samples)

    sharpes: list[float] = []
    for one in seeds:
        returns = array_or_none(run(int(one)))
        if returns is None:
            continue
        sharpes.append(sharpe_ratio(returns, periods_per_year=spec.periods_per_year))
    return np.asarray(sharpes, dtype=np.float64)


def _pad(reduced: npt.NDArray[np.float64], size: int, window: int) -> npt.NDArray[np.float64]:
    """Right-align a windowed reduction onto the original series, NaN-filled.

    NaN rather than a partial window: a 3-bar "60-bar volatility" is not a small
    inaccuracy, it is a different statistic wearing the same label. Right
    alignment is what makes the window *trailing* — each value labels the bar it
    ends on, never one it precedes.
    """
    out = np.full(size, np.nan, dtype=np.float64)
    out[window - 1 :] = reduced
    return out
