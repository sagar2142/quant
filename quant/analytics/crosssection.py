"""Cross-sectional analytics — MASTER_PLAN §6, §8, §268.

One security in isolation tells you what it did. A cross-section tells you what
it did *relative to everything else*, which is the only form most equity edges
take: momentum is a ranking, value is a ranking, and a name that rose 20% in a
market that rose 25% underperformed.

Three questions, three sections:

    ranking       where does each name sit on return, risk and liquidity
    structure     what is actually correlated with what, and how many
                  independent bets are really here
    allocation    given that structure, what weights are defensible

**Covariance is shrunk, never raw.** With 30 names and 250 sessions the sample
covariance matrix is nearly singular, and mean-variance optimisation on it
produces enormous offsetting positions in the smallest-eigenvalue directions.
Ledoit-Wolf shrinkage is the difference between an optimiser that works and one
that blows up (§268), so the condition number is reported: it is the number
that tells you whether to trust the weights at all.

**Clusters matter more than the correlation matrix.** Ten positions in
correlated PSU banks is one bet with ten tickers. A gross-exposure limit sees
diversification that is not there; the cluster count sees the bet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.math.linalg.covariance import (
    condition_number,
    correlation_from_covariance,
    ledoit_wolf_shrinkage,
)
from quant.math.metrics.performance import TRADING_DAYS, sharpe_ratio, volatility
from quant.math.optim.allocation import (
    cluster_assets,
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_variance_weights,
)

__all__ = [
    "CrossSection",
    "NameStats",
    "analyse_cross_section",
    "diversification_ratio",
    "effective_bets",
]

#: Correlation above which two names are treated as one bet.
CLUSTER_THRESHOLD = 0.5

#: A condition number past this means the covariance matrix is near-singular
#: and any optimiser output should be treated as arbitrary.
ILL_CONDITIONED = 1e6

MIN_NAMES = 2
MIN_SESSIONS = 60

#: Effective bets below this fraction of the name count means the book is one
#: position wearing many tickers.
CONCENTRATED_FRACTION = 0.3

#: A (T, N) return matrix has exactly this many dimensions.
MATRIX_DIMS = 2


@dataclass(frozen=True)
class NameStats:
    """One instrument's place in the cross-section."""

    symbol: str
    total_return: float
    annual_volatility: float
    sharpe: float
    beta: float
    correlation_to_market: float
    weight_hrp: float
    weight_erc: float
    cluster: int


@dataclass(frozen=True)
class CrossSection:
    """The universe, ranked and decomposed."""

    names: list[NameStats]
    sessions: int
    correlation: npt.NDArray[np.float64]
    mean_correlation: float
    clusters: list[list[int]]
    condition_number: float
    shrinkage: float
    effective_bets: float
    diversification_ratio: float
    market_return: float
    market_volatility: float

    @property
    def is_ill_conditioned(self) -> bool:
        """Whether the covariance matrix can support an optimiser at all."""
        return self.condition_number > ILL_CONDITIONED

    @property
    def concentration_warning(self) -> str | None:
        """Says so when the book is one bet wearing many tickers."""
        if len(self.names) < MIN_NAMES:
            return None
        independent = self.effective_bets / len(self.names)
        if independent < CONCENTRATED_FRACTION:
            return (
                f"{len(self.names)} names but only {self.effective_bets:.1f} "
                "effective bets — this is a concentrated position, not a "
                "diversified book"
            )
        return None

    def ranked_by(self, field: str, descending: bool = True) -> list[NameStats]:
        return sorted(self.names, key=lambda n: getattr(n, field), reverse=descending)


def effective_bets(correlation: npt.NDArray[np.float64]) -> float:
    """How many independent positions a correlated book really holds.

    The inverse participation ratio of the correlation matrix eigenvalues. Ten
    uncorrelated names score 10; ten copies of the same name score 1. This is
    the honest denominator for any diversification claim.
    """
    eigenvalues = np.linalg.eigvalsh(correlation)
    positive = eigenvalues[eigenvalues > 0]
    if positive.size == 0:
        return 0.0
    normalised = positive / positive.sum()
    return float(1.0 / np.sum(normalised**2))


def diversification_ratio(
    weights: npt.NDArray[np.float64], covariance: npt.NDArray[np.float64]
) -> float:
    """Weighted average volatility over portfolio volatility.

    1.0 means diversification bought nothing — the names move together. Higher
    is better, and the gap between this and 1.0 is what correlation is
    actually paying you.
    """
    variances = np.diag(covariance)
    weighted_vol = float(np.sum(weights * np.sqrt(np.maximum(variances, 0.0))))
    variance = float(weights @ covariance @ weights)
    portfolio_vol = float(np.sqrt(max(variance, 0.0)))
    return weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0


def analyse_cross_section(
    symbols: list[str],
    returns: npt.ArrayLike,
    periods_per_year: int = TRADING_DAYS,
) -> CrossSection:
    """Full cross-sectional decomposition.

    Args:
        symbols: Names, in column order.
        returns: (T, N) matrix of per-period returns.

    Raises:
        ValueError: on too few names or too little history. A correlation
            matrix from 20 sessions is noise, and weights derived from it are
            noise with decimal places.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != MATRIX_DIMS:
        raise ValueError(f"returns must be a (T, N) matrix, got shape {matrix.shape}")
    if matrix.shape[1] < MIN_NAMES:
        raise ValueError(f"need at least {MIN_NAMES} names, got {matrix.shape[1]}")
    if matrix.shape[0] < MIN_SESSIONS:
        raise ValueError(
            f"{matrix.shape[0]} sessions is below the {MIN_SESSIONS} needed for "
            "a correlation matrix to be anything but noise"
        )
    if len(symbols) != matrix.shape[1]:
        raise ValueError(f"{len(symbols)} symbols for {matrix.shape[1]} columns")

    covariance, shrinkage = ledoit_wolf_shrinkage(matrix)
    correlation = correlation_from_covariance(covariance)
    clusters = cluster_assets(covariance, threshold=CLUSTER_THRESHOLD)

    # Equal-weight proxy for "the market these names face" (§ generators).
    market = matrix.mean(axis=1)
    market_vol = float(np.std(market, ddof=1))

    hrp = hierarchical_risk_parity(covariance)
    try:
        erc = equal_risk_contribution(covariance)
    except (ValueError, np.linalg.LinAlgError):
        # A near-singular matrix can defeat the solver. Inverse variance needs
        # only the diagonal, so it degrades rather than failing.
        erc = inverse_variance_weights(covariance)

    cluster_of = {member: i for i, group in enumerate(clusters) for member in group}

    names = []
    for i, symbol in enumerate(symbols):
        column = matrix[:, i]
        column_vol = float(np.std(column, ddof=1))
        beta = (
            float(np.cov(column, market, ddof=1)[0, 1] / (market_vol**2)) if market_vol > 0 else 0.0
        )
        corr = (
            float(np.corrcoef(column, market)[0, 1]) if column_vol > 0 and market_vol > 0 else 0.0
        )
        names.append(
            NameStats(
                symbol=symbol,
                total_return=float(np.prod(1.0 + column) - 1.0),
                annual_volatility=volatility(column, periods_per_year=periods_per_year),
                sharpe=sharpe_ratio(column, periods_per_year=periods_per_year),
                beta=beta,
                correlation_to_market=corr,
                weight_hrp=float(hrp[i]),
                weight_erc=float(erc[i]),
                cluster=cluster_of.get(i, -1),
            )
        )

    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]

    return CrossSection(
        names=names,
        sessions=int(matrix.shape[0]),
        correlation=correlation,
        mean_correlation=float(np.mean(off_diagonal)) if off_diagonal.size else 0.0,
        clusters=clusters,
        condition_number=condition_number(covariance),
        shrinkage=shrinkage,
        effective_bets=effective_bets(correlation),
        diversification_ratio=diversification_ratio(hrp, covariance),
        market_return=float(np.prod(1.0 + market) - 1.0),
        market_volatility=market_vol * float(np.sqrt(periods_per_year)),
    )
