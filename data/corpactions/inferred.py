"""Inferring splits from the price series — MASTER_PLAN §9.

**Why this exists rather than a vendor fetch.** `data.feeds.yahoo` gives real
corporate actions and is the right source for a handful of names. The factor
lab scores two and a half thousand at once, and a per-symbol network call for
each is minutes of rate-limited requests before any research happens. Left
unadjusted, 593 of 2,575 equity names — 23% — carry at least one session move
above 35% in the last thousand sessions, and every one enters a momentum factor
as a real return. A 1:1 bonus reads as -50%.

**The inference is deliberately narrow.** A move is treated as a split only if
it lands close to a *plausible ratio*: 1:2, 1:5, 2:3 and their common
relatives. A genuine crash of -38% is left alone; a move of exactly -50.0% on
a liquid name is not a crash. That narrowness is the point — the failure this
guards against is silently rewriting a real price move, which would be worse
than the contamination it removes.

**This is for research, never for the ledger.** The backtester applies real
corporate actions to *positions* from the real action book (§9); nothing here
touches that path. Factor construction needs a comparable price series, which
is a different and weaker requirement than knowing what a holding actually did.

Where certainty matters — a name you intend to trade — profile it through
`apps.cli.terminal`, which fetches the real action book for that one symbol.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "COMMON_RATIOS",
    "RATIO_TOLERANCE",
    "SUSPECT_MOVE",
    "adjust_for_inferred_splits",
    "inferred_split_factors",
]

#: A session move beyond this is a candidate. Matches the data-quality check's
#: threshold so the two agree about what looks wrong.
SUSPECT_MOVE = 0.35

#: Ratios a split or bonus actually takes, as the price multiplier on the
#: ex-date. A 1:1 bonus halves the price (0.5); a 1-for-5 reverse split
#: quintuples it (5.0).
COMMON_RATIOS: tuple[float, ...] = (
    0.1,  # 1:10 or 9:1 bonus
    0.125,
    0.2,  # 1:5
    0.25,  # 1:4
    1 / 3,  # 1:3
    0.4,
    0.5,  # 1:2 split or 1:1 bonus — much the commonest
    2 / 3,  # 3:2
    1.5,
    2.0,  # reverse
    3.0,
    5.0,
    10.0,
)

#: How close the observed move must be to a listed ratio, proportionally. Two
#: percent: tight enough that a -48% crash is not mistaken for a 1:1 bonus,
#: loose enough to absorb the day's genuine price move on top of the split.
RATIO_TOLERANCE = 0.02


def _matched_ratios(
    step: npt.NDArray[np.float64], same_name: npt.NDArray[np.bool_]
) -> npt.NDArray[np.float64]:
    """The split ratio each session move matches, or 1.0 where none does.

    Vectorised over the whole panel: the candidate mask is one numpy pass and
    the ratio table is thirteen entries, so this is thirteen comparisons rather
    than a Python loop over four million rows. The earlier row-wise version
    made a single factor take three and a half minutes to build, against the
    six seconds the research loop is supposed to cost at this stage.

    First match wins, as a nearest-ratio search would: the table is spaced far
    wider than RATIO_TOLERANCE, so no move matches two entries.
    """
    factors = np.ones(step.size, dtype=np.float64)
    candidate = np.flatnonzero(
        np.isfinite(step) & (step > 0) & (np.abs(step - 1.0) > SUSPECT_MOVE) & same_name
    )
    if candidate.size == 0:
        return factors

    # The ratio table is walked over the candidates alone, not the panel. Large
    # moves are rare — a few thousand rows out of millions — so this is the
    # difference between thirteen passes over the whole panel and thirteen
    # passes over a rounding error's worth of it.
    moves = step[candidate]
    matched = np.ones(moves.size, dtype=np.float64)
    for ratio in COMMON_RATIOS:
        hit = (matched == 1.0) & (np.abs(moves - ratio) / ratio <= RATIO_TOLERANCE)
        matched[hit] = ratio
    factors[candidate] = matched
    return factors


def inferred_split_factors(panel: pl.DataFrame, already_sorted: bool = False) -> pl.DataFrame:
    """Cumulative price adjustment per row, from inferred splits.

    Args:
        already_sorted: The caller guarantees the panel is ordered by
            (symbol, event_time). Sorting three million rows costs more than
            everything else here put together, so a caller that has already
            paid for it should not pay twice. Wrong, it silently pairs each
            price with the previous row of whatever name happens to precede
            it — hence the conservative default.

    Returns:
        The panel with a `split_factor` column. Multiplying a raw close by it
        gives a series continuous across the inferred events — prices *before*
        each split are scaled down, exactly as `back_adjust` does with real
        actions.
    """
    if panel.is_empty():
        return panel.with_columns(pl.lit(1.0).alias("split_factor"))

    ordered = panel if already_sorted else panel.sort(["symbol", "event_time"])
    step = ordered.select(
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).alias("step")
    )["step"].to_numpy()

    # The step across a boundary between two names is not a price move.
    symbols = ordered["symbol"].to_numpy()
    same_name = np.r_[False, symbols[1:] == symbols[:-1]]

    framed = ordered.with_columns(pl.Series("_f", _matched_ratios(step, same_name)))

    # A price is scaled by every split that happened *after* it. The reverse
    # cumulative product gives the product from each row to the end of the
    # name; dividing out the row's own factor leaves everything strictly after
    # it, which is the side of the event that needs adjusting.
    return framed.with_columns(
        (pl.col("_f").reverse().cum_prod().reverse().over("symbol") / pl.col("_f")).alias(
            "split_factor"
        )
    ).drop("_f")


def adjust_for_inferred_splits(panel: pl.DataFrame, already_sorted: bool = False) -> pl.DataFrame:
    """Panel with prices made continuous across inferred splits.

    Volume moves the other way — more shares outstanding, more traded — so it
    is divided by the same factor, keeping traded *value* invariant. A
    liquidity filter computed on unadjusted volume across a split would
    otherwise see a name's turnover jump for no reason.
    """
    framed = inferred_split_factors(panel, already_sorted=already_sorted)
    price_columns = [c for c in ("open", "high", "low", "close") if c in framed.columns]
    updates = [(pl.col(c) * pl.col("split_factor")).alias(c) for c in price_columns]
    if "volume" in framed.columns:
        updates.append((pl.col("volume") / pl.col("split_factor")).alias("volume"))
    return framed.with_columns(updates).drop("split_factor")
