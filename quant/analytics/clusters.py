"""Correlation groups for the risk engine's concentration limit — §8.

**The limit existed and nothing populated it.** `RiskEngine` checks
`max_cluster_pct` against `order.cluster`, and `cluster_assets` in
`quant.math.optim.allocation` carries a docstring saying it "feeds the risk
engine's cluster limit". Neither end was joined: every order was built with no
cluster, so the check was skipped on every order ever placed, and the console's
Cluster column rendered an em dash for every position.

**Correlation rather than sector.** A sector table would be the conventional
source and the repository holds none. It would also be a proxy: the limit exists
because correlated names are one bet however many tickers they carry, and
correlation is the thing itself rather than a label that usually tracks it. Two
banks in different sectors that move together are one bet; two banks in the same
sector that do not are two. The label would get both wrong.

**Estimated from a trailing window, never the full sample.** Clusters computed
over all history would use tomorrow's correlations to size today's order. The
window ends at the decision point like everything else here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from core.instruments import InstrumentId
from quant.math.optim.allocation import cluster_assets

__all__ = [
    "CLUSTER_THRESHOLD",
    "CLUSTER_WINDOW",
    "MIN_CLUSTER_BARS",
    "MIN_OBS_PER_NAME",
    "assign_clusters",
]

#: Sessions of returns behind each correlation estimate.
#:
#: A year rather than a quarter, because the matrix must be estimated from more
#: observations than it has names. Thirty names over 63 sessions is n/p ≈ 2 and
#: the sample correlations are mostly noise; over 252 it is n/p ≈ 8, and real
#: structure appears — 13 pairs above 0.5 against 5 on the shorter window.
CLUSTER_WINDOW = 252

#: Observations per name below which the correlation matrix is not estimated at
#: all. Clustering a matrix with fewer rows than columns groups estimation
#: error and calls the result a risk limit.
MIN_OBS_PER_NAME = 4.0

#: Correlation distance below which two names are one group. 0.5 in distance is
#: roughly 0.5 in correlation, which is where "these move together" starts being
#: true enough to matter for concentration.
CLUSTER_THRESHOLD = 0.5

#: Below this many overlapping bars a correlation is not worth estimating, and
#: the name is left unclustered rather than grouped on noise.
MIN_CLUSTER_BARS = 20


def assign_clusters(
    panel: pl.DataFrame,
    instruments: tuple[InstrumentId, ...],
    window: int = CLUSTER_WINDOW,
    threshold: float = CLUSTER_THRESHOLD,
) -> dict[InstrumentId, str]:
    """Map each instrument to a correlation-group label.

    Args:
        panel: Price history, filtered to what was observable at the decision
            point. Using unfiltered history would let tomorrow's correlations
            size today's order.
        instruments: The universe to group. Names with too little overlapping
            history are returned unclustered.

    Returns:
        `{instrument_id: "c0"}`. A name absent from the mapping has no cluster,
        which the risk engine treats as an unchecked concentration rather than
        a cleared one — the same distinction the liquidity check now makes.
    """
    if len(instruments) < 2:  # noqa: PLR2004 - a group of one is not a group
        return {}

    wanted = list(dict.fromkeys(instruments))
    recent = panel.filter(pl.col("instrument_id").is_in(wanted))
    if recent.is_empty():
        return {}

    sessions = recent["event_time"].unique().sort().tail(window)
    recent = recent.filter(pl.col("event_time").is_in(sessions.implode()))

    # One column per instrument, aligned on session. Inner alignment on purpose:
    # a correlation across non-overlapping dates is not a correlation.
    wide = (
        recent.select("event_time", "instrument_id", "close")
        .pivot(on="instrument_id", index="event_time", values="close")
        .sort("event_time")
    )
    columns = [c for c in wide.columns if c != "event_time"]
    if len(columns) < 2:  # noqa: PLR2004 - nothing to correlate against
        return {}

    prices = wide.select(columns).to_numpy().astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = prices[1:] / prices[:-1] - 1.0

    # Drop names that are mostly gaps before estimating anything.
    finite = np.isfinite(returns)
    usable = finite.sum(axis=0) >= MIN_CLUSTER_BARS
    if usable.sum() < 2:  # noqa: PLR2004
        return {}
    returns = np.nan_to_num(returns[:, usable], nan=0.0, posinf=0.0, neginf=0.0)
    names = [c for c, keep in zip(columns, usable, strict=True) if keep]

    if returns.shape[0] < MIN_CLUSTER_BARS:
        return {}

    # Enough rows to estimate a matrix this wide, or nothing is returned.
    if returns.shape[0] < MIN_OBS_PER_NAME * returns.shape[1]:
        return {}

    # Sample covariance, deliberately, not Ledoit-Wolf. Shrinkage exists to
    # stabilise an optimiser by pulling correlations toward a constant, and on
    # this panel it pulls hard: intensity came out at 0.902, flattening every
    # pairwise correlation to about 0.10 and leaving nothing to group. That is
    # shrinkage working as designed and being the wrong tool — clustering wants
    # the structure that shrinkage is built to suppress.
    covariance = np.cov(returns, rowvar=False)
    groups = cluster_assets(covariance, threshold=threshold)

    labels: dict[InstrumentId, str] = {}
    for index, members in enumerate(groups):
        for member in members:
            labels[InstrumentId(names[member])] = f"c{index}"
    return labels
