"""Covariance estimation — MASTER_PLAN Part 4.

**The thing nobody tells you**, and the reason this module exists: with 50
instruments and 250 days of data, the sample covariance matrix is nearly
singular. Mean-variance optimisation on it produces enormous leveraged
positions along the smallest-eigenvalue directions — the ones estimated
worst — because the optimiser reads low estimated variance as low risk.

Ledoit-Wolf shrinkage pulls the sample estimate toward a well-conditioned
target, trading a little bias for an enormous reduction in variance. It is the
difference between a portfolio optimiser that works and one that blows up, and
it costs about fifteen lines.

float64 here: these are statistics, not money (§14.1.2).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "condition_number",
    "correlation_from_covariance",
    "ledoit_wolf_shrinkage",
    "sample_covariance",
]

FloatArray = npt.NDArray[np.float64]

#: Below this many observations per asset, a sample covariance is not an
#: estimate, it is an accident.
MIN_OBS_PER_ASSET = 2
#: A covariance input is (observations, assets).
MATRIX_DIMENSIONS = 2


def _clean(returns: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != MATRIX_DIMENSIONS:
        raise ValueError(f"expected a 2-D (observations, assets) matrix, got {matrix.shape}")
    if matrix.shape[0] < MIN_OBS_PER_ASSET:
        raise ValueError(f"need at least {MIN_OBS_PER_ASSET} observations")
    return np.asarray(np.nan_to_num(matrix, nan=0.0), dtype=np.float64)


def sample_covariance(returns: npt.ArrayLike) -> FloatArray:
    """Plain sample covariance.

    Provided for comparison and for the cases where observations comfortably
    exceed assets. Prefer `ledoit_wolf_shrinkage` whenever they do not.
    """
    matrix = _clean(returns)
    return np.asarray(np.cov(matrix, rowvar=False, ddof=1), dtype=np.float64)


def ledoit_wolf_shrinkage(returns: npt.ArrayLike) -> tuple[FloatArray, float]:
    """Shrink the sample covariance toward a constant-correlation target.

    Returns:
        The shrunk covariance matrix and the shrinkage intensity in [0, 1].
        An intensity near 1 means the sample estimate was almost pure noise —
        which is itself worth knowing before trusting any optimiser output.

    The target preserves each asset's own variance (which is estimated
    reasonably well) and replaces the correlation structure (which is not) with
    a single average correlation.
    """
    matrix = _clean(returns)
    observations, assets = matrix.shape

    centred = matrix - matrix.mean(axis=0)
    sample = np.asarray(centred.T @ centred / observations, dtype=np.float64)

    variances = np.diag(sample)
    std = np.sqrt(np.maximum(variances, 1e-300))
    outer_std = np.outer(std, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.where(outer_std > 0, sample / outer_std, 0.0)

    off_diagonal = ~np.eye(assets, dtype=bool)
    mean_correlation = float(correlations[off_diagonal].mean()) if assets > 1 else 0.0

    target = mean_correlation * outer_std
    np.fill_diagonal(target, variances)

    # Shrinkage intensity: estimation error over the distance to the target.
    squared = np.asarray(centred**2, dtype=np.float64)
    phi = float(np.sum(squared.T @ squared / observations - sample**2))
    gamma = float(np.sum((target - sample) ** 2))

    if gamma <= 0:
        return np.asarray(sample, dtype=np.float64), 0.0

    intensity = max(0.0, min(1.0, phi / (gamma * observations)))
    shrunk = intensity * target + (1.0 - intensity) * sample
    return np.asarray(shrunk, dtype=np.float64), intensity


def correlation_from_covariance(covariance: npt.ArrayLike) -> FloatArray:
    """Correlation matrix implied by a covariance matrix."""
    matrix = np.asarray(covariance, dtype=np.float64)
    std = np.sqrt(np.maximum(np.diag(matrix), 1e-300))
    outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlations = np.where(outer > 0, matrix / outer, 0.0)
    np.fill_diagonal(correlations, 1.0)
    return np.asarray(np.clip(correlations, -1.0, 1.0), dtype=np.float64)


def condition_number(matrix: npt.ArrayLike) -> float:
    """Ratio of largest to smallest eigenvalue.

    A practical diagnostic: above roughly 1e6 the matrix is numerically
    singular and any optimiser output derived from it should be treated as
    noise regardless of how confident it looks.
    """
    eigenvalues = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    smallest = float(np.min(np.abs(eigenvalues)))
    largest = float(np.max(np.abs(eigenvalues)))
    if smallest <= 0:
        return float("inf")
    return largest / smallest
