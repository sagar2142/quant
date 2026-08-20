"""Combining signals — MASTER_PLAN §6.

**Nobody trades one factor.** A desk z-scores several, removes the overlap
between them, weights what remains by how well each has actually predicted,
and trades the composite. The single-factor lab answers "does this predict
anything"; this answers "what should I actually hold".

**The overlap is not hypothetical.** Measured on this panel, `momentum_12_1`
and `high_52w_proximity` have a cross-sectional rank correlation of 0.65, and
the nine-factor library is worth about 5.9 independent bets. Stacking them
naively would count the same effect twice and call it diversification.

**Three steps, in this order, and the order matters:**

    z-score        put every factor on one scale, cross-sectionally per
                   session, so a factor measured in percent and one measured
                   in rupees can be added at all
    orthogonalize  remove from each factor the part already explained by the
                   ones before it, so overlap is counted once
    weight         by historical Information Coefficient, so a factor that has
                   predicted better carries more of the composite

**Weights are computed on the same sample they are applied to**, which is
in-sample by construction and the honest name for it is overfitting. That is
tolerable here and nowhere else: this module produces a *candidate*, and the
gauntlet's walk-forward and PBO checks exist precisely to find out whether the
candidate survives out of sample. It must never be read as evidence on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import polars as pl

from quant.research.factors import Factor, FactorSpec, build_factor
from quant.research.ic import information_coefficient

__all__ = [
    "WINSOR_LIMIT",
    "CompositeSpec",
    "FactorOverlap",
    "combine_factors",
    "factor_correlations",
    "orthogonalise",
    "zscore_by_session",
]

#: Cross-sectional z-scores are clipped here. A single name at 40 sigma — which
#: a ratio factor produces whenever its denominator approaches zero — would
#: otherwise dominate the composite for that session on its own.
WINSOR_LIMIT = 3.0

#: Below this a session has too few names for a cross-sectional z-score to
#: mean anything, and it is dropped rather than scaled against noise.
MIN_NAMES = 20

#: A composite is a combination; one factor is a factor.
MIN_FACTORS = 2


@dataclass(frozen=True)
class CompositeSpec:
    """How a set of factors is combined.

    Args:
        factors: Ordered. Orthogonalisation is sequential, so the first factor
            keeps all of its variance and later ones keep only what is new.
            Put the factor you believe in most first — that ordering is a
            research decision, not an implementation detail.
        orthogonalise: Remove overlap between factors. Off only to measure what
            the overlap was costing.
        ic_weight: Weight by historical IC rather than equally. Falls back to
            equal weights when no factor has a positive IC, because negative
            weights would silently invert a signal.
        horizon: Forward horizon the IC weights are fitted against.
    """

    factors: tuple[Factor, ...]
    min_adv: float = 1e7
    window: int = 0
    orthogonalise: bool = True
    ic_weight: bool = True
    horizon: int = 21

    def __post_init__(self) -> None:
        if len(self.factors) < MIN_FACTORS:
            raise ValueError("a composite needs at least two factors")
        if len(set(self.factors)) != len(self.factors):
            raise ValueError("a factor cannot appear twice in a composite")


@dataclass(frozen=True)
class FactorOverlap:
    """How independent a set of factors actually is."""

    names: list[str]
    correlation: npt.NDArray[np.float64]
    #: Participation ratio of the eigenvalues: how many independent bets the
    #: set is worth. Nine factors at 5.9 means three of them are duplicates.
    effective_factors: float
    #: Variance share of the largest principal component. A high number means
    #: the set is mostly one thing.
    first_component: float
    observations: int
    #: Pairs above the reporting threshold, worst first.
    redundant_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def format(self) -> str:
        lines = [
            f"  {len(self.names)} factors, {self.observations:,} observations",
            f"  effective independent factors {self.effective_factors:.2f}"
            f"   first component {self.first_component:.1%}",
        ]
        if self.redundant_pairs:
            lines.append("  overlapping pairs:")
            lines.extend(f"    {a:<22} {b:<22} {rho:+.2f}" for a, b, rho in self.redundant_pairs)
        return "\n".join(lines)


def zscore_by_session(column: str) -> pl.Expr:
    """Cross-sectional z-score, winsorised.

    Per session rather than pooled: a pooled z-score would rank a calm period
    against a volatile one and call the difference signal.
    """
    mean = pl.col(column).mean().over("event_time")
    deviation = pl.col(column).std().over("event_time")
    # A zero-dispersion session carries no cross-sectional information; scoring
    # it as zero is the honest answer rather than dividing by zero.
    return (
        pl.when(deviation > 0)
        .then((pl.col(column) - mean) / deviation)
        .otherwise(0.0)
        .clip(-WINSOR_LIMIT, WINSOR_LIMIT)
    )


def _residualise(
    target: npt.NDArray[np.float64], basis: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """`target` with the part explained by `basis` removed.

    Least squares rather than a correlation subtraction, so several existing
    factors are removed jointly instead of one at a time — sequential removal
    leaves the shared part of a correlated basis behind.
    """
    if basis.size == 0:
        return target
    coefficients, *_ = np.linalg.lstsq(basis, target, rcond=None)
    return target - basis @ coefficients


def orthogonalise(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Sequential Gram-Schmidt across factor columns, within each session.

    Each factor keeps only the variance the earlier ones do not explain, so an
    effect shared by two factors is counted once. Order therefore decides which
    factor keeps the shared part — the first one listed.

    Residuals are computed in one numpy pass and written back once. The obvious
    implementation — rebuilding the frame per session per column — is a
    fourteen-thousand-fold frame copy on this panel and takes minutes rather
    than seconds.
    """
    if len(columns) < MIN_FACTORS or frame.is_empty():
        return frame

    ordered = frame.sort("event_time")
    block = ordered.select(columns).to_numpy().astype(np.float64)
    sessions = ordered["event_time"].to_numpy()

    # Session boundaries in the sorted array, so each cross-section is a slice.
    starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    bounds = np.r_[starts, len(sessions)]

    for lo, hi in pairwise(bounds):
        if hi - lo < MIN_NAMES:
            continue
        section = block[lo:hi]
        residuals = np.empty_like(section)
        residuals[:, 0] = section[:, 0]
        for i in range(1, section.shape[1]):
            residuals[:, i] = _residualise(section[:, i], residuals[:, :i])

        # Rescale so an orthogonalised factor still contributes at unit size.
        # Residuals are smaller than the original by construction, and without
        # this the later factors would fade out of the composite.
        spreads = residuals.std(axis=0)
        np.divide(residuals, spreads, out=residuals, where=spreads > 0)
        block[lo:hi] = residuals

    return ordered.with_columns(
        [pl.Series(column, block[:, j]) for j, column in enumerate(columns)]
    )


def factor_correlations(
    history: pl.DataFrame,
    factors: tuple[Factor, ...],
    min_adv: float = 1e7,
    window: int = 750,
    threshold: float = 0.5,
) -> FactorOverlap:
    """How much the factors overlap, and which pairs are duplicates.

    The first thing to run on any factor set. A library of nine that is worth
    six independent bets is not a library of nine, and combining them as though
    it were double-counts whatever the duplicates share.
    """
    joined: pl.DataFrame | None = None
    names: list[str] = []
    # Deduplicated for the same reason as `aligned_returns`: each factor
    # becomes a column keyed by its own name, and a repeat would either
    # silently correlate a factor with itself or collide on the join.
    for factor in dict.fromkeys(factors):
        scored = build_factor(
            history, FactorSpec(factor, min_adv=min_adv, window=window), (21,)
        ).select("event_time", "symbol", pl.col("signal").alias(factor.value))
        names.append(factor.value)
        joined = scored if joined is None else joined.join(scored, on=["event_time", "symbol"])

    if joined is None or joined.is_empty():
        return FactorOverlap([], np.empty((0, 0)), 0.0, 0.0, 0)

    ranked = joined.with_columns([pl.col(n).rank().over("event_time") for n in names])
    matrix = ranked.select(names).drop_nulls().to_numpy().astype(np.float64)
    if matrix.shape[0] < MIN_NAMES:
        return FactorOverlap(names, np.empty((0, 0)), 0.0, 0.0, int(matrix.shape[0]))

    correlation = np.corrcoef(matrix, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    effective = float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())

    pairs = [
        (names[i], names[j], float(correlation[i, j]))
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(correlation[i, j]) >= threshold
    ]
    return FactorOverlap(
        names=names,
        correlation=correlation,
        effective_factors=effective,
        first_component=float(eigenvalues[0] / eigenvalues.sum()),
        observations=int(matrix.shape[0]),
        redundant_pairs=sorted(pairs, key=lambda p: -abs(p[2])),
    )


def combine_factors(
    history: pl.DataFrame, spec: CompositeSpec, horizons: tuple[int, ...] = (1, 5, 21, 63)
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Build the composite signal.

    Returns:
        The scored panel with a `signal` column holding the composite, in the
        same shape `build_factor` produces so every existing analysis works on
        it unchanged; and the weight each factor received, so a composite can
        be explained rather than merely used.
    """
    every = tuple(sorted({*horizons, spec.horizon}))
    joined: pl.DataFrame | None = None
    columns: list[str] = []

    for factor in spec.factors:
        scored = build_factor(
            history, FactorSpec(factor, min_adv=spec.min_adv, window=spec.window), every
        )
        columns.append(factor.value)
        renamed = scored.rename({"signal": factor.value})
        if joined is None:
            joined = renamed
        else:
            joined = joined.join(
                renamed.select("event_time", "symbol", factor.value),
                on=["event_time", "symbol"],
            )

    if joined is None or joined.is_empty():
        return pl.DataFrame(), {}

    # Weights come from the raw factors: the IC of an orthogonalised residual
    # measures the leftover, not the factor's own predictive power.
    weights = _ic_weights(joined, columns, spec)

    standardised = joined.with_columns(
        [zscore_by_session(column).alias(column) for column in columns]
    )
    if spec.orthogonalise:
        standardised = orthogonalise(standardised, columns)

    composite = standardised.with_columns(
        sum(
            (pl.col(column) * weights[column] for column in columns),
            start=pl.lit(0.0),
        ).alias("signal")
    )
    return (
        composite.select("event_time", "symbol", "signal", *[f"fwd_{h}" for h in every])
        .drop_nulls("signal")
        .filter(pl.col("signal").is_finite()),
        weights,
    )


def _ic_weights(joined: pl.DataFrame, columns: list[str], spec: CompositeSpec) -> dict[str, float]:
    """Weight per factor, normalised to sum to one."""
    if not spec.ic_weight:
        return dict.fromkeys(columns, 1.0 / len(columns))

    scores: dict[str, float] = {}
    for column in columns:
        renamed = joined.rename({column: "signal"})
        scores[column] = max(0.0, information_coefficient(renamed, spec.horizon).mean)

    total = sum(scores.values())
    if total <= 0:
        # Every factor has a non-positive IC. Equal weights make the composite
        # readable rather than inverting signals via negative weights, and the
        # IC of the result will say plainly that there is nothing here.
        return dict.fromkeys(columns, 1.0 / len(columns))
    return {name: value / total for name, value in scores.items()}
