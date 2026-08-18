"""Portfolio allocation — MASTER_PLAN §17, Part 4.

Three allocators, deliberately ordered from most robust to least:

**Hierarchical Risk Parity** (Lopez de Prado). Clusters assets by correlation,
then allocates down the tree by inverse variance. Its decisive property is that
**it never inverts the covariance matrix**, which is precisely where
mean-variance optimisation goes wrong when assets outnumber observations. Slightly
worse in theory, dramatically better in practice.

**Equal risk contribution.** Each asset contributes the same share of portfolio
variance. Sound, simple, and needs no return forecast — which matters because
return forecasts are the least reliable input any optimiser receives.

**Inverse variance.** The naive baseline. Correct only when assets are
uncorrelated, which they never are, but a useful sanity floor.

Notably absent: unconstrained mean-variance. The plan's position (§17) is
"start simple and add optimisation only when evidence supports it", and
mean-variance on an ill-conditioned covariance matrix produces confident
nonsense. `covariance.condition_number` exists to tell you when you are in that
regime.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster, linkage, to_tree
from scipy.spatial.distance import squareform

from quant.math.linalg.covariance import correlation_from_covariance

__all__ = [
    "cluster_assets",
    "correlation_distance",
    "equal_risk_contribution",
    "hierarchical_risk_parity",
    "inverse_variance_weights",
]

FloatArray = npt.NDArray[np.float64]

#: A covariance matrix is square and two-dimensional.
MATRIX_DIMENSIONS = 2


def _validate(covariance: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != MATRIX_DIMENSIONS or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"covariance must be square, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("covariance matrix is empty")
    return matrix


def inverse_variance_weights(covariance: npt.ArrayLike) -> FloatArray:
    """Weights proportional to 1/variance. Correct only if uncorrelated."""
    matrix = _validate(covariance)
    variances = np.maximum(np.diag(matrix), 1e-300)
    weights = 1.0 / variances
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def equal_risk_contribution(
    covariance: npt.ArrayLike,
    iterations: int = 500,
    tolerance: float = 1e-10,
) -> FloatArray:
    """Weights where every asset contributes equally to portfolio variance.

    Solved by *damped* fixed-point iteration. The damping is not a refinement,
    it is required for convergence: the undamped update
    ``w <- normalise(1 / marginal)`` overshoots and settles into a period-2
    cycle, so an even iteration count returns the starting point. That failure
    is quiet and convincing — it hands back equal weights, which look like a
    reasonable answer.

    The square root halves each step and converges monotonically. For a
    diagonal covariance it reaches the exact inverse-volatility solution in one
    iteration.
    """
    matrix = _validate(covariance)
    n = matrix.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)

    weights = np.ones(n, dtype=np.float64) / n
    for _ in range(iterations):
        marginal = np.maximum(matrix @ weights, 1e-300)
        # Each asset's risk contribution; equalising these is the whole goal.
        target = float((weights * marginal).sum()) / n
        updated = np.sqrt(np.maximum(weights * target / marginal, 0.0))
        total = float(updated.sum())
        if total <= 0:
            return np.asarray(weights, dtype=np.float64)
        updated = updated / total
        if float(np.max(np.abs(updated - weights))) < tolerance:
            return np.asarray(updated, dtype=np.float64)
        weights = updated
    return np.asarray(weights, dtype=np.float64)


def correlation_distance(covariance: npt.ArrayLike) -> FloatArray:
    """Distance metric on correlations: ``sqrt((1 - rho) / 2)``.

    Maps perfect correlation to 0 and perfect anticorrelation to 1, and is a
    proper metric, which is what makes hierarchical clustering meaningful here.
    """
    correlations = correlation_from_covariance(covariance)
    distance = np.sqrt(np.maximum((1.0 - correlations) / 2.0, 0.0))
    np.fill_diagonal(distance, 0.0)
    # Enforce exact symmetry; squareform rejects even tiny asymmetry.
    return np.asarray((distance + distance.T) / 2.0, dtype=np.float64)


def cluster_assets(covariance: npt.ArrayLike, threshold: float = 0.5) -> list[list[int]]:
    """Group assets whose correlation distance falls below `threshold`.

    Feeds the risk engine's cluster limit (§8): names in one group count as a
    single bet, however many tickers they carry.
    """
    matrix = _validate(covariance)
    if matrix.shape[0] == 1:
        return [[0]]

    distance = correlation_distance(matrix)
    tree = linkage(squareform(distance, checks=False), method="single")

    labels = fcluster(tree, t=threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return [groups[k] for k in sorted(groups)]


def _quasi_diagonal_order(covariance: FloatArray) -> list[int]:
    """Order assets so correlated ones sit adjacent, via the cluster tree."""
    n = covariance.shape[0]
    if n == 1:
        return [0]
    distance = correlation_distance(covariance)
    tree = to_tree(linkage(squareform(distance, checks=False), method="single"))

    order: list[int] = []

    def walk(node: object) -> None:
        if node.is_leaf():  # type: ignore[attr-defined]
            order.append(int(node.id))  # type: ignore[attr-defined]
            return
        walk(node.get_left())  # type: ignore[attr-defined]
        walk(node.get_right())  # type: ignore[attr-defined]

    walk(tree)
    return order


def _cluster_variance(covariance: FloatArray, members: list[int]) -> float:
    """Variance of an inverse-variance-weighted sub-portfolio."""
    block = covariance[np.ix_(members, members)]
    weights = inverse_variance_weights(block)
    return float(weights @ block @ weights)


def hierarchical_risk_parity(covariance: npt.ArrayLike) -> FloatArray:
    """HRP weights — the default allocator.

    Recursively bisects the correlation-ordered asset list and splits capital
    between halves in inverse proportion to their variance. No matrix
    inversion anywhere, so an ill-conditioned covariance degrades the answer
    gently instead of producing a confident, enormous, wrong one.
    """
    matrix = _validate(covariance)
    n = matrix.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)

    order = _quasi_diagonal_order(matrix)
    weights = np.ones(n, dtype=np.float64)
    clusters = [order]

    while clusters:
        # Bisect every cluster with more than one member.
        clusters = [
            half
            for cluster in clusters
            for half in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            left_var = _cluster_variance(matrix, left)
            right_var = _cluster_variance(matrix, right)
            total = left_var + right_var
            # Allocate away from the riskier half.
            alpha = 1.0 - left_var / total if total > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
        clusters = [c for c in clusters if len(c) > 1]

    return np.asarray(weights / weights.sum(), dtype=np.float64)
