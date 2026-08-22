"""Composed analytics (§2, §6, §8, §253).

`quant/math` is tested on its estimators. This tests the *composition*: that a
profile reports what the series actually did, and that a cross-section sees
structure that is there and refuses to invent structure that is not.

Fixtures are built from series of known construction, so a wrong answer is
distinguishable from a surprising market.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.analytics.crosssection import (
    CONCENTRATED_FRACTION,
    analyse_cross_section,
    diversification_ratio,
    effective_bets,
)
from quant.analytics.security import (
    HORIZONS,
    MIN_HISTORY,
    autocorrelation,
    conditional_var,
    current_drawdown,
    horizon_return,
    profile_security,
    tail_ratio,
    value_at_risk,
)

SEED = 20260818


def prices(n: int = 800, drift: float = 0.0004, sigma: float = 0.012, seed: int = SEED):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(drift, sigma, n)))


class TestSecurityProfile:
    def test_a_short_series_is_refused(self):
        """A profile of zeros reads as a calm instrument. Refuse instead."""
        with pytest.raises(ValueError, match="below the"):
            profile_security("X", prices(MIN_HISTORY - 1))

    def test_last_close_matches_the_input(self):
        series = prices()
        assert profile_security("X", series).last_close == pytest.approx(series[-1])

    def test_horizons_are_none_when_history_is_short(self):
        """A '3y return' over 200 sessions is a different statistic under the
        same label."""
        profile = profile_security("X", prices(200))
        assert profile.horizon_returns["3y"] is None
        assert profile.horizon_returns["1m"] is not None

    def test_a_rising_series_has_positive_cagr(self):
        assert profile_security("X", prices(drift=0.0008)).cagr > 0

    def test_a_falling_series_has_negative_cagr(self):
        assert profile_security("X", prices(drift=-0.0008)).cagr < 0

    def test_drawdown_is_never_positive(self):
        profile = profile_security("X", prices())
        assert profile.max_drawdown <= 0
        assert profile.current_drawdown <= 0

    def test_off_high_is_zero_at_a_new_high(self):
        rising = np.cumsum(np.full(300, 1.0)) + 100.0
        assert profile_security("X", rising).off_high == pytest.approx(0.0)

    def test_volatility_recovers_the_input(self):
        profile = profile_security("X", prices(sigma=0.02))
        assert profile.annual_volatility == pytest.approx(0.02 * np.sqrt(252), rel=0.2)

    def test_adv_is_computed_when_volume_is_given(self):
        series = prices()
        profile = profile_security("X", series, np.full(series.size, 1000.0))
        assert profile.adv_value is not None
        assert profile.adv_value > 0

    def test_adv_is_absent_without_volume(self):
        assert profile_security("X", prices()).adv_value is None

    def test_a_random_walk_is_not_fadeable(self):
        """The check that stops a z-score strategy on a trending name."""
        profile = profile_security("X", prices())
        assert not profile.stationarity.tradable_as_mean_reversion

    def test_implausible_sharpe_is_flagged(self):
        """A single name above 2.5 usually means a missed corporate action."""
        rng = np.random.default_rng(SEED)
        strong = 100.0 * np.exp(np.cumsum(rng.normal(0.004, 0.005, 500)))
        assert profile_security("X", strong).is_implausible

    def test_a_perfectly_smooth_series_is_flagged(self):
        """Constant daily returns are not a market. This used to be caught only
        because `np.std` of a constant array is ~1e-19 rather than 0, so the
        Sharpe overflowed to 7e16 and tripped the gate by accident."""
        smooth = 100.0 * np.exp(np.cumsum(np.full(500, 0.001)))
        profile = profile_security("X", smooth)
        assert profile.has_no_variance
        assert profile.is_implausible

    def test_a_perfectly_flat_series_is_flagged_too(self):
        """The case the accident missed: dividing by exactly 0.0 scored 0.0 and
        sailed through. Two identical pathologies, opposite verdicts."""
        profile = profile_security("X", np.full(500, 100.0))
        assert profile.has_no_variance
        assert profile.is_implausible

    def test_a_normal_name_is_not_flagged(self):
        rng = np.random.default_rng(SEED)
        ordinary = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, 500)))
        profile = profile_security("X", ordinary)
        assert not profile.has_no_variance
        assert not profile.is_implausible

    def test_a_fat_left_tail_is_flagged(self):
        """Small gains until ruin — the distribution a Sharpe ratio hides."""
        rng = np.random.default_rng(SEED)
        returns = rng.normal(0.001, 0.005, 500)
        returns[100] = -0.30
        returns[300] = -0.25
        series = 100.0 * np.exp(np.cumsum(returns))
        assert profile_security("X", series).fat_left_tail

    def test_every_horizon_is_reported(self):
        profile = profile_security("X", prices(1000))
        assert set(profile.horizon_returns) == set(HORIZONS)


class TestRiskStatistics:
    def test_var_is_the_percentile_loss(self):
        returns = np.linspace(-0.10, 0.10, 201)
        assert value_at_risk(returns, 5.0) == pytest.approx(-0.09, abs=0.005)

    def test_cvar_is_worse_than_var(self):
        """CVaR says how bad; VaR only says how often. A strategy dies of the
        first."""
        rng = np.random.default_rng(SEED)
        returns = rng.normal(0, 0.02, 1000)
        assert conditional_var(returns) < value_at_risk(returns)

    def test_tail_ratio_above_one_when_gains_outrun_losses(self):
        rng = np.random.default_rng(SEED)
        returns = rng.gumbel(0, 0.01, 2000)  # right-skewed
        assert tail_ratio(returns) > 1.0

    def test_empty_input_is_safe(self):
        empty = np.array([])
        assert value_at_risk(empty) == 0.0
        assert conditional_var(empty) == 0.0
        assert tail_ratio(empty) == 1.0
        assert current_drawdown(empty) == 0.0

    def test_drawdown_is_zero_at_the_peak(self):
        assert current_drawdown(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)

    def test_drawdown_measures_from_the_peak(self):
        assert current_drawdown(np.array([100.0, 50.0])) == pytest.approx(-0.5)


class TestAutocorrelation:
    def test_positive_for_a_persistent_series(self):
        rng = np.random.default_rng(SEED)
        values = np.zeros(500)
        for i in range(1, 500):
            values[i] = 0.6 * values[i - 1] + rng.normal(0, 1)
        assert autocorrelation(values, 1) > 0.4

    def test_negative_for_an_alternating_series(self):
        values = np.array([1.0, -1.0] * 250)
        assert autocorrelation(values, 1) < -0.9

    def test_near_zero_for_noise(self):
        rng = np.random.default_rng(SEED)
        assert abs(autocorrelation(rng.normal(0, 1, 2000), 1)) < 0.1

    def test_a_short_series_returns_zero(self):
        assert autocorrelation(np.array([1.0, 2.0]), 5) == 0.0

    def test_a_constant_series_returns_zero(self):
        assert autocorrelation(np.full(100, 3.0), 1) == 0.0


class TestHorizonReturn:
    def test_computes_the_simple_return(self):
        series = np.array([100.0, 110.0, 121.0])
        assert horizon_return(series, 2) == pytest.approx(0.21)

    def test_none_when_the_window_exceeds_history(self):
        assert horizon_return(np.array([100.0, 110.0]), 10) is None

    def test_none_on_a_non_positive_base(self):
        assert horizon_return(np.array([0.0, 110.0]), 1) is None


class TestCrossSection:
    def correlated_block(self, n: int = 300, seed: int = SEED):
        """Two names that move together, two that do not."""
        rng = np.random.default_rng(seed)
        common = rng.normal(0, 0.01, n)
        return np.column_stack(
            [
                common + rng.normal(0, 0.002, n),
                common + rng.normal(0, 0.002, n),
                rng.normal(0, 0.01, n),
                rng.normal(0, 0.01, n),
            ]
        )

    def names(self) -> list[str]:
        return ["TWIN_A", "TWIN_B", "SOLO_C", "SOLO_D"]

    def test_correlated_names_land_in_one_cluster(self):
        """Ten positions in correlated banks is one bet with ten tickers."""
        section = analyse_cross_section(self.names(), self.correlated_block())
        by_symbol = {n.symbol: n.cluster for n in section.names}
        assert by_symbol["TWIN_A"] == by_symbol["TWIN_B"]

    def test_effective_bets_is_below_the_name_count(self):
        section = analyse_cross_section(self.names(), self.correlated_block())
        assert section.effective_bets < len(section.names)

    def test_independent_names_score_near_the_full_count(self):
        rng = np.random.default_rng(SEED)
        independent = rng.normal(0, 0.01, (300, 4))
        section = analyse_cross_section(self.names(), independent)
        assert section.effective_bets > 3.5

    def test_weights_sum_to_one(self):
        section = analyse_cross_section(self.names(), self.correlated_block())
        assert sum(n.weight_hrp for n in section.names) == pytest.approx(1.0)
        assert sum(n.weight_erc for n in section.names) == pytest.approx(1.0)

    def test_shrinkage_is_applied(self):
        """Raw sample covariance is near-singular and blows up an optimiser."""
        section = analyse_cross_section(self.names(), self.correlated_block())
        assert 0.0 <= section.shrinkage <= 1.0

    def test_beta_to_the_equal_weight_market(self):
        section = analyse_cross_section(self.names(), self.correlated_block())
        assert all(-3.0 < n.beta < 3.0 for n in section.names)

    def test_ranking_is_ordered(self):
        section = analyse_cross_section(self.names(), self.correlated_block())
        returns = [n.total_return for n in section.ranked_by("total_return")]
        assert returns == sorted(returns, reverse=True)

    def test_a_concentrated_book_is_warned_about(self):
        """Four copies of one series is one bet, and must say so."""
        rng = np.random.default_rng(SEED)
        common = rng.normal(0, 0.01, 300)
        identical = np.column_stack([common + rng.normal(0, 1e-6, 300) for _ in range(4)])
        section = analyse_cross_section(self.names(), identical)
        assert section.effective_bets / len(section.names) < CONCENTRATED_FRACTION
        assert section.concentration_warning is not None

    def test_a_diversified_book_is_not_warned_about(self):
        rng = np.random.default_rng(SEED)
        section = analyse_cross_section(self.names(), rng.normal(0, 0.01, (300, 4)))
        assert section.concentration_warning is None

    def test_too_few_names_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            analyse_cross_section(["ONLY"], np.random.default_rng(1).normal(0, 0.01, (300, 1)))

    def test_too_little_history_is_refused(self):
        """A correlation matrix from 20 sessions is noise, and weights from it
        are noise with decimal places."""
        with pytest.raises(ValueError, match="below the"):
            analyse_cross_section(["A", "B"], np.random.default_rng(1).normal(0, 0.01, (20, 2)))

    def test_mismatched_symbols_are_refused(self):
        with pytest.raises(ValueError, match="symbols for"):
            analyse_cross_section(
                ["A", "B", "C"], np.random.default_rng(1).normal(0, 0.01, (300, 2))
            )

    def test_a_non_matrix_is_refused(self):
        with pytest.raises(ValueError, match="must be a"):
            analyse_cross_section(["A"], np.array([1.0, 2.0, 3.0]))


class TestStructureMetrics:
    def test_effective_bets_of_an_identity_matrix_is_the_dimension(self):
        assert effective_bets(np.eye(5)) == pytest.approx(5.0)

    def test_effective_bets_of_a_fully_correlated_matrix_is_one(self):
        assert effective_bets(np.ones((5, 5))) == pytest.approx(1.0)

    def test_diversification_ratio_is_one_when_perfectly_correlated(self):
        covariance = np.full((3, 3), 0.04)
        weights = np.full(3, 1 / 3)
        assert diversification_ratio(weights, covariance) == pytest.approx(1.0)

    def test_diversification_ratio_exceeds_one_when_uncorrelated(self):
        covariance = np.eye(3) * 0.04
        weights = np.full(3, 1 / 3)
        assert diversification_ratio(weights, covariance) > 1.0


class TestDuplicateSymbols:
    """A cross-section of a name against itself is not a cross-section.

    Each name becomes a column keyed by its own symbol, so a repeat used to
    produce a second auto-suffixed column: the same instrument twice,
    correlating 1.0 with itself and understating the effective-bet count. That
    returned a plausible 200 rather than an error, which is the worse failure.
    Above two repeats the join collided outright.
    """

    def panel(self, symbols, sessions: int = 300):
        from datetime import datetime, timedelta

        import polars as pl

        from core.clock import UTC

        rng = np.random.default_rng(7)
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(sessions)]
        return pl.concat(
            [
                pl.DataFrame(
                    {
                        "event_time": times,
                        "symbol": [s] * sessions,
                        "instrument_id": [f"NSE:{s}"] * sessions,
                        "open": (
                            c := list(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, sessions))))
                        ),
                        "high": c,
                        "low": c,
                        "close": c,
                        "volume": [1e6] * sessions,
                    },
                    schema_overrides={"event_time": pl.Datetime("us", "UTC")},
                )
                for s in symbols
            ]
        )

    def test_a_repeated_symbol_is_collapsed(self):
        from apps.cli.terminal import aligned_returns

        frame = self.panel(["AAA", "BBB"])
        kept, matrix = aligned_returns(frame, ["AAA", "AAA", "BBB"])
        assert kept == ["AAA", "BBB"]
        assert matrix.shape[1] == 2

    def test_many_repeats_do_not_crash(self):
        """Three or more collided on the join and raised DuplicateError."""
        from apps.cli.terminal import aligned_returns

        frame = self.panel(["AAA"])
        kept, matrix = aligned_returns(frame, ["AAA"] * 40)
        assert kept == ["AAA"]
        assert matrix.shape[1] == 1

    def test_first_occurrence_order_is_kept(self):
        from apps.cli.terminal import aligned_returns

        frame = self.panel(["AAA", "BBB"])
        kept, _ = aligned_returns(frame, ["BBB", "AAA", "BBB"])
        assert kept == ["BBB", "AAA"]
