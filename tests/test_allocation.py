"""Covariance estimation and portfolio allocation (§17, Part 4).

`TestShrinkageMatters` is the point of the covariance module: it demonstrates
the ill-conditioning that destroys naive mean-variance optimisation, and that
shrinkage fixes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.math.linalg.covariance import (
    condition_number,
    correlation_from_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from quant.math.optim.allocation import (
    cluster_assets,
    correlation_distance,
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_variance_weights,
)

SEED = 20260816


def uncorrelated(n_obs: int = 500, n_assets: int = 5) -> np.ndarray:
    return np.random.default_rng(SEED).normal(0, 0.01, (n_obs, n_assets))


def two_blocks(n_obs: int = 500) -> np.ndarray:
    """Six assets in two tightly-correlated groups of three."""
    rng = np.random.default_rng(SEED)
    factor_a = rng.normal(0, 0.01, n_obs)
    factor_b = rng.normal(0, 0.01, n_obs)
    noise = lambda: rng.normal(0, 0.002, n_obs)  # noqa: E731
    return np.column_stack(
        [
            factor_a + noise(),
            factor_a + noise(),
            factor_a + noise(),
            factor_b + noise(),
            factor_b + noise(),
            factor_b + noise(),
        ]
    )


class TestShrinkageMatters:
    """Why this module exists (§Part 4)."""

    def test_more_assets_than_observations_is_ill_conditioned(self):
        # 40 assets, 30 observations: the sample estimate is singular.
        returns = np.random.default_rng(SEED).normal(0, 0.01, (30, 40))
        assert condition_number(sample_covariance(returns)) > 1e6

    def test_shrinkage_conditions_the_matrix(self):
        returns = np.random.default_rng(SEED).normal(0, 0.01, (30, 40))
        shrunk, intensity = ledoit_wolf_shrinkage(returns)
        assert intensity > 0
        assert condition_number(shrunk) < condition_number(sample_covariance(returns))

    def test_intensity_is_high_when_data_is_noise(self):
        # Pure noise: the sample correlation structure is entirely estimation
        # error, so the estimator should lean almost wholly on the target.
        _, intensity = ledoit_wolf_shrinkage(uncorrelated(60, 40))
        assert intensity > 0.5

    def test_intensity_is_bounded(self):
        for returns in (uncorrelated(), two_blocks(), uncorrelated(50, 45)):
            _, intensity = ledoit_wolf_shrinkage(returns)
            assert 0.0 <= intensity <= 1.0

    def test_variances_are_preserved(self):
        """The target keeps each asset's own variance, which is well estimated."""
        returns = two_blocks()
        shrunk, _ = ledoit_wolf_shrinkage(returns)
        sample = sample_covariance(returns)
        assert np.allclose(np.diag(shrunk), np.diag(sample), rtol=0.05)

    def test_result_is_symmetric_positive_semidefinite(self):
        shrunk, _ = ledoit_wolf_shrinkage(uncorrelated(60, 30))
        assert np.allclose(shrunk, shrunk.T)
        assert np.min(np.linalg.eigvalsh(shrunk)) > -1e-12


class TestCovarianceValidation:
    def test_non_square_rejected(self):
        with pytest.raises(ValueError, match="2-D"):
            sample_covariance(np.zeros(10))

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least"):
            sample_covariance(np.zeros((1, 5)))

    def test_correlation_diagonal_is_one(self):
        correlations = correlation_from_covariance(sample_covariance(two_blocks()))
        assert np.allclose(np.diag(correlations), 1.0)

    def test_correlation_is_bounded(self):
        correlations = correlation_from_covariance(sample_covariance(two_blocks()))
        assert correlations.min() >= -1.0
        assert correlations.max() <= 1.0

    def test_singular_matrix_has_infinite_condition_number(self):
        assert condition_number(np.zeros((3, 3))) == float("inf")


class TestInverseVariance:
    def test_weights_sum_to_one(self):
        assert inverse_variance_weights(sample_covariance(uncorrelated())).sum() == pytest.approx(
            1.0
        )

    def test_lower_variance_gets_more_weight(self):
        covariance = np.diag([0.01, 0.04])
        weights = inverse_variance_weights(covariance)
        assert weights[0] > weights[1]
        # 4x the variance gets 1/4 the weight.
        assert weights[0] / weights[1] == pytest.approx(4.0)

    def test_single_asset(self):
        assert inverse_variance_weights(np.array([[0.01]])) == pytest.approx([1.0])

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            inverse_variance_weights(np.zeros((0, 0)))


class TestEqualRiskContribution:
    def test_weights_sum_to_one(self):
        weights = equal_risk_contribution(sample_covariance(two_blocks()))
        assert weights.sum() == pytest.approx(1.0)

    def test_risk_contributions_are_equal(self):
        covariance = sample_covariance(two_blocks())
        weights = equal_risk_contribution(covariance)
        contributions = weights * (covariance @ weights)
        # Every asset contributes the same share of portfolio variance.
        assert np.std(contributions) / np.mean(contributions) < 0.05

    def test_matches_inverse_vol_when_uncorrelated(self):
        covariance = np.diag([0.01, 0.04, 0.09])
        weights = equal_risk_contribution(covariance)
        # For a diagonal matrix, ERC reduces to inverse volatility.
        expected = 1.0 / np.sqrt(np.diag(covariance))
        assert np.allclose(weights, expected / expected.sum(), atol=0.01)

    def test_weights_are_non_negative(self):
        assert equal_risk_contribution(sample_covariance(two_blocks())).min() >= 0

    def test_single_asset(self):
        assert equal_risk_contribution(np.array([[0.01]])) == pytest.approx([1.0])


class TestHierarchicalRiskParity:
    def test_weights_sum_to_one(self):
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        assert hierarchical_risk_parity(covariance).sum() == pytest.approx(1.0)

    def test_weights_are_non_negative(self):
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        assert hierarchical_risk_parity(covariance).min() >= 0

    def test_survives_ill_conditioned_covariance(self):
        """The decisive property: no matrix inversion anywhere.

        Mean-variance on this input would produce enormous offsetting
        positions along the smallest-eigenvalue directions.
        """
        returns = np.random.default_rng(SEED).normal(0, 0.01, (25, 40))
        weights = hierarchical_risk_parity(sample_covariance(returns))
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)
        assert weights.max() < 0.5  # nothing enormous

    def test_correlated_block_shares_one_allocation(self):
        """Three near-identical assets should not get triple the capital."""
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        weights = hierarchical_risk_parity(covariance)
        # Two blocks of three; each block should get roughly half.
        assert weights[:3].sum() == pytest.approx(0.5, abs=0.15)

    def test_single_asset(self):
        assert hierarchical_risk_parity(np.array([[0.01]])) == pytest.approx([1.0])

    def test_deterministic(self):
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        runs = {tuple(np.round(hierarchical_risk_parity(covariance), 12)) for _ in range(5)}
        assert len(runs) == 1


class TestClustering:
    """Feeds the risk engine's cluster limit (§8)."""

    def test_distance_metric_bounds(self):
        distance = correlation_distance(sample_covariance(two_blocks()))
        assert distance.min() >= 0
        assert distance.max() <= 1
        assert np.allclose(np.diag(distance), 0.0)

    def test_distance_is_symmetric(self):
        distance = correlation_distance(sample_covariance(two_blocks()))
        assert np.allclose(distance, distance.T)

    def test_finds_the_two_blocks(self):
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        clusters = cluster_assets(covariance, threshold=0.4)
        # Members 0-2 belong together, and 3-5 belong together.
        membership = {i: c for c, group in enumerate(clusters) for i in group}
        assert membership[0] == membership[1] == membership[2]
        assert membership[3] == membership[4] == membership[5]
        assert membership[0] != membership[3]

    def test_uncorrelated_assets_stay_separate(self):
        covariance = sample_covariance(uncorrelated(500, 5))
        clusters = cluster_assets(covariance, threshold=0.1)
        assert len(clusters) == 5

    def test_single_asset(self):
        assert cluster_assets(np.array([[0.01]])) == [[0]]

    def test_every_asset_appears_exactly_once(self):
        covariance, _ = ledoit_wolf_shrinkage(two_blocks())
        members = [i for group in cluster_assets(covariance) for i in group]
        assert sorted(members) == list(range(6))
