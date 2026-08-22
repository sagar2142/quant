"""Performance metrics — MASTER_PLAN Part 4, §35.

**These had no direct tests, which is how two of them shipped wrong.** Every
verdict the system reaches is computed from this module: the gauntlet's
drawdown gate, the Calmar in every report, the Sharpe that DSR deflates. A
metric that is quietly optimistic does not raise; it approves a strategy.

The tests are known-answer wherever an analytic result exists, and property
tests otherwise. Both bugs found here failed in the *flattering* direction,
which is the direction nobody investigates.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.math.metrics.performance import (
    TRADING_DAYS,
    cagr,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    volatility,
)


def equity_drawdown(returns: list[float]) -> float:
    """Independent reference: build the curve explicitly and measure it."""
    equity = np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns, dtype=float))])
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


class TestMaxDrawdown:
    def test_a_loss_on_the_first_bar_is_a_drawdown(self):
        """The bug: without a leading 1.0 the first bar is its own peak, so a
        decline starting at inception was invisible. This reported 0.00%."""
        assert max_drawdown([-0.5, 0.5]) == pytest.approx(-0.5)

    def test_a_strategy_that_only_loses_reports_its_full_decline(self):
        assert max_drawdown([-0.1] * 10) == pytest.approx(equity_drawdown([-0.1] * 10))

    def test_the_worst_day_being_day_one_is_not_special(self):
        assert max_drawdown([-0.3, 0.01, 0.01]) == pytest.approx(-0.3)

    def test_a_peak_after_inception_still_works(self):
        """The case that always worked; it must not regress."""
        assert max_drawdown([0.5, -0.5]) == pytest.approx(-0.5)

    def test_a_rising_series_has_no_drawdown(self):
        assert max_drawdown([0.01] * 10) == 0.0

    def test_a_flat_series_has_no_drawdown(self):
        assert max_drawdown([0.0] * 5) == 0.0

    def test_it_is_never_positive(self):
        rng = np.random.default_rng(0)
        for seed in range(20):
            r = np.random.default_rng(seed).normal(0.001, 0.02, 200)
            assert max_drawdown(r) <= 0.0
        assert max_drawdown(rng.normal(0.05, 0.001, 50)) <= 0.0

    def test_it_agrees_with_an_explicit_curve(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            r = rng.normal(-0.001, 0.02, 300).tolist()
            assert max_drawdown(r) == pytest.approx(equity_drawdown(r))

    def test_an_empty_series_is_zero(self):
        assert max_drawdown([]) == 0.0

    def test_total_loss_is_minus_one(self):
        assert max_drawdown([-1.0]) == pytest.approx(-1.0)


class TestSharpeZeroVariance:
    def test_a_constant_series_scores_zero(self):
        """The docstring promised 0.0 and returned 7.3e16. `np.std` of a
        constant array is ~1e-19, never exactly 0.0, so an `== 0.0` guard never
        fired and the mean divided by floating-point residue."""
        assert sharpe_ratio(np.full(300, 0.001)) == 0.0

    def test_exact_zeros_still_score_zero(self):
        assert sharpe_ratio(np.zeros(300)) == 0.0

    @pytest.mark.parametrize("level", [1e-9, 1e-3, 1.0, 100.0])
    def test_no_constant_series_scores_anything(self, level):
        """Scaled against the series, so a constant of any magnitude is caught
        rather than only the ones near 1.0."""
        assert sharpe_ratio(np.full(300, level)) == 0.0

    def test_a_real_series_is_unaffected(self):
        r = np.random.default_rng(0).normal(0.001, 0.01, 5000)
        expected = 0.001 / 0.01 * np.sqrt(TRADING_DAYS)
        assert sharpe_ratio(r) == pytest.approx(expected, rel=0.1)

    def test_a_genuinely_tiny_but_varying_series_is_not_suppressed(self):
        """The floor must not swallow a real signal that happens to be small."""
        r = np.random.default_rng(1).normal(1e-6, 1e-6, 1000)
        assert sharpe_ratio(r) != 0.0

    def test_sortino_of_a_constant_loser_is_defined_not_zero(self):
        """Sortino's denominator is the RMS of the *negative* returns, not
        their spread, so a constant -0.1%/day has a real downside deviation of
        0.001. -15.9 is the right answer; suppressing it would hide a strategy
        that loses every single day.
        """
        assert sortino_ratio(np.full(300, -0.001)) == pytest.approx(
            -0.001 / 0.001 * np.sqrt(TRADING_DAYS)
        )

    def test_sortino_suppresses_only_numerically_empty_downside(self):
        assert sortino_ratio(np.full(300, -1e-15)) == 0.0

    def test_sortino_with_no_losing_period_is_zero_not_infinite(self):
        assert sortino_ratio(np.full(300, 0.001)) == 0.0

    def test_sharpe_is_never_absurd_on_constant_input(self):
        """§2.1's smell test, applied to the metric itself."""
        for level in (1e-12, 1e-6, 0.001, 5.0):
            assert abs(sharpe_ratio(np.full(500, level))) < 1e6


class TestCalmar:
    def test_it_uses_the_corrected_drawdown(self):
        """Calmar was 0.0 for a series whose drawdown was misreported as 0."""
        assert calmar_ratio([-0.5, 0.5]) == pytest.approx(cagr([-0.5, 0.5]) / 0.5)

    def test_no_drawdown_yields_zero_rather_than_infinity(self):
        assert calmar_ratio([0.01] * 10) == 0.0


class TestKnownAnswers:
    def test_cagr_doubling_in_one_year(self):
        daily = 2.0 ** (1 / TRADING_DAYS) - 1
        assert cagr([daily] * TRADING_DAYS) == pytest.approx(1.0, abs=1e-9)

    def test_cagr_of_a_wipeout_is_total_loss(self):
        assert cagr([-1.0, 0.5]) == -1.0

    def test_volatility_annualises(self):
        r = np.random.default_rng(0).normal(0, 0.01, 100_000)
        assert volatility(r) == pytest.approx(0.01 * np.sqrt(TRADING_DAYS), rel=0.02)

    def test_hit_rate_counts_positive_periods(self):
        assert hit_rate([1.0, -1.0, 1.0, -1.0]) == 0.5

    def test_a_flat_series_never_wins(self):
        """Zero is not a positive return."""
        assert hit_rate(np.zeros(10)) == 0.0

    def test_returns_from_equity_round_trips(self):
        assert returns_from_equity([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])

    def test_one_equity_point_yields_no_returns(self):
        assert returns_from_equity([100.0]).size == 0


class TestNonFiniteHandling:
    def test_nans_are_dropped_not_zero_filled(self):
        """Treating a missing observation as a flat day understates volatility."""
        clean = np.array([0.01, -0.02, 0.03])
        dirty = np.array([0.01, np.nan, -0.02, 0.03])
        assert volatility(dirty) == pytest.approx(volatility(clean))

    def test_infinities_are_dropped(self):
        assert np.isfinite(sharpe_ratio([0.01, np.inf, -0.01, 0.02]))
