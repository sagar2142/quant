"""Time-series statistics (§253-256).

Every test builds a series whose true nature is known by construction — a
random walk, an OU process, a cointegrated pair — and asserts the estimator
recovers it. That is the only honest way to test a statistical routine: on
real data you cannot tell a wrong answer from a surprising market.

The gates these functions feed are the ones the plan names as the risks of
strategy families 4 and 5: non-stationarity and spurious correlation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant.math.timeseries.cointegration import (
    engle_granger,
    hedge_ratio,
    spread_series,
)
from quant.math.timeseries.stationarity import (
    MIN_OBSERVATIONS,
    Stationarity,
    adf_test,
    assess_stationarity,
    hurst_exponent,
    kpss_test,
)
from quant.math.timeseries.volatility import (
    MAX_VOL_SCALE,
    ewma_volatility,
    forecast_volatility,
    garch_volatility,
    realised_volatility,
)

SEED = 20260818
N = 600


def random_walk(n: int = N, seed: int = SEED, drift: float = 0.0) -> np.ndarray:
    """Non-stationary by construction: a unit root."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(drift, 1.0, size=n)) + 100.0


def persistent_walk(n: int = N, seed: int = SEED, phi: float = 0.7) -> np.ndarray:
    """Trending in the Hurst sense: increments are positively autocorrelated.

    Note this is *not* a straight line plus noise. A deterministic trend has no
    variance growth across lags, so it scores near zero — correctly, because
    Hurst measures stochastic persistence, not slope.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0, 1.0, n)
    increments = np.zeros(n)
    for i in range(1, n):
        increments[i] = phi * increments[i - 1] + shocks[i]
    return np.cumsum(increments) + 100.0


def ou_process(n: int = N, seed: int = SEED, theta: float = 0.15) -> np.ndarray:
    """Stationary by construction: Ornstein-Uhlenbeck about zero."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = out[i - 1] + theta * (0.0 - out[i - 1]) + rng.normal(0, 1.0)
    return out


class TestStationarityOnKnownSeries:
    def test_a_random_walk_is_not_stationary(self):
        """The exact case that breaks a z-score strategy: no level to revert to."""
        report = assess_stationarity(random_walk())
        assert report.verdict is not Stationarity.STATIONARY
        assert not report.tradable_as_mean_reversion

    def test_an_ou_process_is_stationary(self):
        report = assess_stationarity(ou_process())
        assert report.verdict is Stationarity.STATIONARY
        assert report.tradable_as_mean_reversion

    def test_adf_rejects_the_unit_root_on_a_stationary_series(self):
        assert adf_test(ou_process()) < 0.05

    def test_adf_cannot_reject_on_a_random_walk(self):
        assert adf_test(random_walk()) > 0.05

    def test_kpss_inverts_adf(self):
        """KPSS's null is stationarity. Getting the inversion wrong flips every
        verdict, so it is asserted directly."""
        assert kpss_test(ou_process()) > 0.05  # cannot reject stationarity
        assert kpss_test(random_walk()) < 0.05  # rejects it

    def test_inconclusive_is_not_tradable(self):
        """Disagreement between the two tests must not read as permission."""
        assert not Stationarity.INCONCLUSIVE.supports_mean_reversion
        assert not Stationarity.UNIT_ROOT.supports_mean_reversion
        assert Stationarity.STATIONARY.supports_mean_reversion


class TestStationarityDegenerateInput:
    """Every guard fails closed — toward "do not trade this"."""

    def test_a_short_series_cannot_reject_a_unit_root(self):
        assert adf_test(np.arange(10, dtype=float)) == 1.0

    def test_a_short_series_fails_kpss(self):
        assert kpss_test(np.arange(10, dtype=float)) == 0.0

    def test_a_constant_series_is_not_declared_tradable(self):
        report = assess_stationarity(np.full(200, 42.0))
        assert not report.tradable_as_mean_reversion

    def test_non_finite_values_are_dropped(self):
        series = ou_process()
        polluted = np.concatenate([series, [np.nan, np.inf, -np.inf]])
        assert assess_stationarity(polluted).observations == series.size

    def test_an_empty_series_is_safe(self):
        assert assess_stationarity(np.array([])).observations == 0

    def test_report_formats(self):
        assert "ADF" in assess_stationarity(ou_process()).format()


class TestHurst:
    def test_a_random_walk_sits_near_one_half(self):
        assert hurst_exponent(random_walk()) == pytest.approx(0.5, abs=0.15)

    def test_a_reverting_series_is_below_one_half(self):
        assert hurst_exponent(ou_process()) < 0.5

    def test_a_persistent_series_is_above_one_half(self):
        assert hurst_exponent(persistent_walk()) > 0.5

    def test_a_deterministic_line_is_not_mistaken_for_persistence(self):
        """The estimator measures stochastic persistence, not slope.

        A straight line has identical variance at every lag, so it scores near
        zero. Reporting a deterministic ramp as strongly trending would be the
        wrong answer to the question the strategy is asking.
        """
        line = np.arange(N, dtype=float) + np.random.default_rng(SEED).normal(0, 0.5, N)
        assert hurst_exponent(line) < 0.5

    def test_a_short_series_returns_a_random_walk(self):
        """No edge either way is the honest answer when there is no data."""
        assert hurst_exponent(np.arange(MIN_OBSERVATIONS - 1, dtype=float)) == 0.5

    def test_the_exponent_stays_in_range(self):
        for series in (random_walk(), ou_process(), persistent_walk()):
            assert 0.0 <= hurst_exponent(series) <= 1.0


class TestCointegration:
    def cointegrated_pair(self, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
        """y = 2x + stationary noise. Cointegrated by construction."""
        rng = np.random.default_rng(seed)
        x = np.cumsum(rng.normal(0, 1.0, N)) + 100.0
        return 2.0 * x + ou_process(N, seed + 1), x

    def independent_walks(self) -> tuple[np.ndarray, np.ndarray]:
        """Two unrelated random walks. Not cointegrated, whatever they correlate."""
        return random_walk(seed=SEED), random_walk(seed=SEED + 99)

    def test_a_cointegrated_pair_is_detected(self):
        y, x = self.cointegrated_pair()
        report = engle_granger(y, x)
        assert report.cointegrated
        assert report.hedge_ratio == pytest.approx(2.0, abs=0.1)

    def test_independent_walks_are_rejected(self):
        """The spurious-pair case the plan flags for strategy family 5."""
        y, x = self.independent_walks()
        assert not engle_granger(y, x).cointegrated

    def test_correlation_is_reported_even_when_the_test_fails(self):
        """A high correlation with a failed test is exactly what a spurious
        pair looks like, so it must be visible rather than hidden."""
        y, x = self.independent_walks()
        report = engle_granger(y, x)
        assert not report.cointegrated
        assert -1.0 <= report.correlation <= 1.0

    def test_hedge_ratio_recovers_the_slope(self):
        y, x = self.cointegrated_pair()
        beta, _ = hedge_ratio(y, x)
        assert beta == pytest.approx(2.0, abs=0.1)

    def test_the_spread_is_the_residual(self):
        y, x = self.cointegrated_pair()
        beta, intercept = hedge_ratio(y, x)
        spread = spread_series(y, x, beta, intercept)
        assert abs(float(np.mean(spread))) < 1.0

    def test_a_slow_pair_is_not_tradable(self):
        """Cointegrated but reverting over a year is a directional position
        wearing a pairs-trade label — its costs exceed its reversion."""
        y, x = self.cointegrated_pair()
        report = engle_granger(y, x)
        assert report.tradable == (
            report.cointegrated
            and np.isfinite(report.half_life_bars)
            and 0 < report.half_life_bars < report.observations / 4
        )

    def test_misaligned_lengths_are_trimmed_together(self):
        y, x = self.cointegrated_pair()
        assert engle_granger(y[:-50], x).observations == y.size - 50

    def test_a_short_pair_is_refused(self):
        report = engle_granger(np.arange(10, dtype=float), np.arange(10, dtype=float))
        assert not report.cointegrated
        assert report.spread_verdict is Stationarity.INCONCLUSIVE

    def test_a_constant_leg_is_handled(self):
        report = engle_granger(ou_process(), np.full(N, 5.0))
        assert report.hedge_ratio == 0.0

    def test_report_formats(self):
        y, x = self.cointegrated_pair()
        assert "beta=" in engle_granger(y, x).format()


class TestVolatility:
    def returns(self, sigma: float = 0.01, n: int = 500) -> np.ndarray:
        return np.random.default_rng(SEED).normal(0, sigma, n)

    def test_realised_recovers_the_true_sigma(self):
        forecast = realised_volatility(self.returns(sigma=0.02))
        assert forecast.daily == pytest.approx(0.02, rel=0.15)
        assert forecast.method == "realised"

    def test_annualisation_uses_the_stated_periods(self):
        forecast = realised_volatility(self.returns(), periods_per_year=252)
        assert forecast.annualised == pytest.approx(forecast.daily * np.sqrt(252))

    def test_ewma_reacts_faster_than_realised(self):
        """A calm history followed by a shock: EWMA should read higher, since
        the equal-weighted estimate still averages in the calm period."""
        calm = np.random.default_rng(SEED).normal(0, 0.005, 400)
        shock = np.random.default_rng(SEED + 1).normal(0, 0.05, 60)
        series = np.concatenate([calm, shock])
        assert ewma_volatility(series).daily > realised_volatility(series).daily

    def test_ewma_rejects_a_nonsense_decay(self):
        for decay in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="decay"):
                ewma_volatility(self.returns(), decay=decay)

    def test_garch_falls_back_on_a_short_sample_and_says_so(self):
        """A risk control that disappears when its estimator fails is not a
        risk control — but it must not lie about which estimator ran."""
        forecast = garch_volatility(self.returns(n=100))
        assert forecast.method == "ewma"

    def test_garch_runs_on_a_long_sample(self):
        forecast = garch_volatility(self.returns(n=800))
        assert forecast.method in {"garch", "ewma"}
        assert forecast.daily > 0

    def test_insufficient_data_reports_zero_not_a_guess(self):
        forecast = realised_volatility(np.array([0.01, 0.02]))
        assert forecast.method == "insufficient"
        assert forecast.annualised == 0.0

    def test_scale_to_target_is_capped(self):
        """Leverage justified by a calm tape is how a calm market becomes a
        large loss."""
        tiny = realised_volatility(self.returns(sigma=0.0001))
        assert tiny.scale_to(0.15) == MAX_VOL_SCALE

    def test_scale_to_reduces_size_when_vol_is_high(self):
        wild = realised_volatility(self.returns(sigma=0.05))
        assert wild.scale_to(0.15) < 1.0

    def test_zero_volatility_scales_to_zero_not_infinity(self):
        assert realised_volatility(np.array([0.01, 0.02])).scale_to(0.15) == 0.0

    def test_dispatch_selects_the_named_estimator(self):
        assert forecast_volatility(self.returns(), "realised").method == "realised"
        assert forecast_volatility(self.returns(), "ewma").method == "ewma"

    def test_an_unknown_method_is_refused(self):
        """A typo in a config must not silently change every position size."""
        with pytest.raises(ValueError, match="unknown volatility method"):
            forecast_volatility(self.returns(), "exponential")
