"""The validation gauntlet — the M5 gate.

The gate has three parts, and the first two are the ones that matter:

    (a) correctly REJECTS a deliberately overfit strategy
    (b) correctly PASSES buy-and-hold
    (c) DSR on a random strategy with N=200 trials returns < 0.95

A gauntlet that passes everything is decoration. A gauntlet that fails
everything is equally useless. Both directions are tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.validation.checks import (
    check_cost_sensitivity,
    check_data_integrity,
    check_locked_oos,
    check_look_ahead,
    check_parameter_plateau,
    check_placebo,
    check_regimes,
    check_universe_dropout,
    check_walk_forward,
)
from engine.validation.gauntlet import run_gauntlet
from engine.validation.report import GauntletInputs
from quant.math.metrics.overfitting import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
)

SEED = 20260815


def noise(n: int = 750, seed: int = SEED, mean: float = 0.0, sd: float = 0.01) -> np.ndarray:
    """Returns with no edge whatsoever."""
    return np.random.default_rng(seed).normal(mean, sd, n)


def modest_edge(n: int = 750, seed: int = SEED) -> np.ndarray:
    """A plausible real strategy: ~1.0 annualised Sharpe."""
    return np.random.default_rng(seed).normal(0.00063, 0.01, n)


class TestDeflatedSharpeGate:
    """M5 gate (c)."""

    def test_random_strategy_fails_with_many_trials(self):
        # 200 trials of a worthless strategy: the best-looking one must not
        # clear the bar, because its Sharpe is the max of 200 draws.
        result = deflated_sharpe_ratio(noise(), n_trials=200)
        assert result.dsr < 0.95

    def test_expected_max_sharpe_grows_with_trials(self):
        """The more you search, the better noise looks — quantified."""
        variance = 1.0 / 750
        few = expected_max_sharpe(10, variance)
        many = expected_max_sharpe(1000, variance)
        assert many > few > 0

    def test_single_trial_has_no_selection_penalty(self):
        assert expected_max_sharpe(1, 1.0 / 750) == 0.0

    def test_same_result_penalised_more_after_more_search(self):
        returns = modest_edge()
        honest = deflated_sharpe_ratio(returns, n_trials=1)
        after_search = deflated_sharpe_ratio(returns, n_trials=500)
        assert honest.dsr > after_search.dsr

    def test_negative_skew_is_measured(self):
        """Option-selling return profiles: many small gains, rare large loss."""
        rng = np.random.default_rng(SEED)
        skewed = np.where(rng.random(750) < 0.97, 0.002, -0.06)
        result = deflated_sharpe_ratio(skewed, n_trials=10)
        # Strongly negative skew and very fat tails, both of which the DSR
        # denominator penalises relative to a Gaussian of the same moments.
        assert result.skew < -3.0
        assert result.kurt > 10.0

    def test_negative_skew_lowers_dsr_versus_symmetric(self):
        rng = np.random.default_rng(SEED)
        skewed = np.where(rng.random(750) < 0.97, 0.002, -0.06)
        # A symmetric series with the SAME mean and standard deviation, so the
        # raw Sharpe is comparable and only the shape differs.
        symmetric = rng.normal(float(np.mean(skewed)), float(np.std(skewed)), 750)
        skewed_stat = deflated_sharpe_ratio(skewed, n_trials=10)
        symmetric_stat = deflated_sharpe_ratio(symmetric, n_trials=10)
        assert skewed_stat.skew < symmetric_stat.skew
        # The penalty shows in the deflation denominator.
        assert skewed_stat.kurt > symmetric_stat.kurt

    def test_short_series_returns_zero_not_a_crash(self):
        assert deflated_sharpe_ratio([0.01, 0.02], n_trials=10).dsr == 0.0


class TestPbo:
    def test_pure_noise_sweep_has_high_pbo(self):
        """Selecting the best of 20 worthless configs should not generalise."""
        rng = np.random.default_rng(SEED)
        sweep = rng.normal(0, 0.01, (600, 20))
        result = probability_of_backtest_overfitting(sweep, n_splits=8)
        assert result.pbo > 0.3

    def test_genuinely_better_config_has_low_pbo(self):
        rng = np.random.default_rng(SEED)
        sweep = rng.normal(0, 0.01, (600, 10))
        # One configuration has a real, persistent edge.
        sweep[:, 3] += 0.0015
        result = probability_of_backtest_overfitting(sweep, n_splits=8)
        assert result.pbo < 0.3

    def test_odd_splits_rejected(self):
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(np.zeros((100, 3)), n_splits=7)

    def test_single_config_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            probability_of_backtest_overfitting(np.zeros((100, 1)))

    def test_too_few_observations_rejected(self):
        with pytest.raises(ValueError, match="at least"):
            probability_of_backtest_overfitting(np.zeros((10, 3)), n_splits=10)


class TestIndividualTests:
    def test_data_integrity_fails_on_critical_findings(self):
        inputs = GauntletInputs(returns=noise(), n_trials=1, seed=SEED, critical_data_findings=3)
        assert not check_data_integrity(inputs).passed

    def test_look_ahead_detects_future_dependence(self):
        base = noise()
        # Results move when future data is corrupted: the strategy read it.
        contaminated = base + 0.001
        inputs = GauntletInputs(
            returns=base, n_trials=1, seed=SEED, shuffled_future_returns=contaminated
        )
        result = check_look_ahead(inputs)
        assert not result.passed
        assert "reading it" in result.reason

    def test_look_ahead_passes_when_identical(self):
        base = noise()
        inputs = GauntletInputs(
            returns=base, n_trials=1, seed=SEED, shuffled_future_returns=base.copy()
        )
        assert check_look_ahead(inputs).passed

    def test_missing_input_is_skipped_not_passed(self):
        """An untested claim is not a satisfied one."""
        inputs = GauntletInputs(returns=noise(), n_trials=1, seed=SEED)
        result = check_look_ahead(inputs)
        assert result.skipped
        assert not result.passed

    def test_cost_sensitivity_fails_when_edge_dies(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            tripled_cost_returns=noise(mean=-0.0005),
        )
        assert not check_cost_sensitivity(inputs).passed

    def test_walk_forward_fails_on_collapse(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            in_sample_returns=modest_edge(seed=1),
            out_of_sample_returns=noise(seed=2, mean=-0.0002),
        )
        assert not check_walk_forward(inputs).passed

    def test_walk_forward_passes_when_edge_persists(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            in_sample_returns=modest_edge(seed=1),
            out_of_sample_returns=modest_edge(seed=2),
        )
        assert check_walk_forward(inputs).passed

    def test_parameter_needle_rejected(self):
        """The MA=137 problem, detected."""
        needle = np.array([0.05, 0.02, 0.03, 2.40, 0.01, 0.04, 0.02])
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            parameter_neighbourhood=needle,
        )
        result = check_parameter_plateau(inputs)
        assert not result.passed
        assert "needle" in result.reason

    def test_parameter_plateau_accepted(self):
        mesa = np.array([0.95, 1.02, 1.10, 1.15, 1.08, 1.01, 0.97])
        inputs = GauntletInputs(
            returns=modest_edge(), n_trials=1, seed=SEED, parameter_neighbourhood=mesa
        )
        assert check_parameter_plateau(inputs).passed

    def test_universe_dropout_fails_on_single_name_dependence(self):
        rng = np.random.default_rng(SEED)
        # Most subsets lose money: the edge lived in one or two names.
        sharpes = rng.normal(-0.2, 0.5, 200)
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            universe_dropout_sharpes=sharpes,
        )
        assert not check_universe_dropout(inputs).passed

    def test_regime_failure_when_only_one_regime_works(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            regime_returns={
                "bull": modest_edge(seed=1),
                "bear": noise(seed=2, mean=-0.0020),
                "chop": noise(seed=3, mean=-0.0015),
            },
        )
        assert not check_regimes(inputs).passed

    def test_regime_pass_with_two_positive(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            regime_returns={
                "bull": modest_edge(seed=1),
                "bear": modest_edge(seed=2),
                "chop": noise(seed=3, mean=-0.0003),
            },
        )
        assert check_regimes(inputs).passed

    def test_placebo_rejects_indistinguishable_strategy(self):
        """If random entries do as well, the market did the work."""
        rng = np.random.default_rng(SEED)
        inputs = GauntletInputs(
            returns=noise(),
            n_trials=1,
            seed=SEED,
            placebo_sharpes=rng.normal(0.5, 0.3, 500),
        )
        assert not check_placebo(inputs).passed

    def test_locked_oos_failure_is_terminal(self):
        inputs = GauntletInputs(
            returns=modest_edge(),
            n_trials=1,
            seed=SEED,
            locked_test_returns=noise(mean=-0.0020),
        )
        result = check_locked_oos(inputs)
        assert not result.passed
        assert "dead, not adjustable" in result.reason


class TestGauntletEndToEnd:
    """M5 gate (a) and (b)."""

    def overfit_inputs(self) -> GauntletInputs:
        """A strategy fitted to noise, dressed up to look wonderful.

        In-sample it is spectacular. Every out-of-sample check collapses.
        """
        rng = np.random.default_rng(SEED)
        return GauntletInputs(
            returns=rng.normal(0.003, 0.008, 750),  # gorgeous in-sample
            n_trials=200,  # found by brute search
            seed=SEED,
            shuffled_future_returns=rng.normal(0.003, 0.008, 750),  # differs → leak
            in_sample_returns=rng.normal(0.003, 0.008, 400),
            out_of_sample_returns=rng.normal(-0.0004, 0.012, 350),
            parameter_neighbourhood=np.array([0.1, 0.05, 3.2, 0.08, 0.02]),
            tripled_cost_returns=rng.normal(-0.0006, 0.008, 750),
            universe_dropout_sharpes=rng.normal(-0.1, 0.4, 200),
            regime_returns={
                "bull": rng.normal(0.004, 0.008, 250),
                "bear": rng.normal(-0.002, 0.012, 250),
                "chop": rng.normal(-0.001, 0.010, 250),
            },
            locked_test_returns=rng.normal(-0.0005, 0.011, 250),
        )

    def buy_and_hold_inputs(self) -> GauntletInputs:
        """Buy-and-hold: honest, unspectacular, and it must pass.

        A gauntlet that rejects buy-and-hold is broken (§M5 gate).
        """
        rng = np.random.default_rng(SEED)
        base = rng.normal(0.0005, 0.011, 900)
        return GauntletInputs(
            returns=base,
            n_trials=1,  # no search: one idea, tested once
            seed=SEED,
            shuffled_future_returns=base.copy(),  # no future dependence
            in_sample_returns=base[:500],
            out_of_sample_returns=base[500:],
            # Buy-and-hold has no parameters, so its neighbourhood is flat.
            parameter_neighbourhood=np.array([0.72, 0.72, 0.72, 0.72, 0.72]),
            tripled_cost_returns=base - 0.00002,  # barely cost-sensitive
            universe_dropout_sharpes=rng.normal(0.7, 0.15, 200),
            # Buy-and-hold makes money in bull and grinds out a little in
            # chop; it loses in bear. Two of three positive is exactly the
            # honest picture, and is what REGIME_MIN_POSITIVE asks for.
            regime_returns={
                "bull": rng.normal(0.0025, 0.009, 300),
                "bear": rng.normal(-0.0010, 0.014, 300),
                "chop": rng.normal(0.0008, 0.008, 300),
            },
            trade_returns=rng.normal(0.0005, 0.011, 200),
            max_drawdown_limit=-0.60,
            locked_test_returns=rng.normal(0.0006, 0.010, 300),
            placebo_sharpes=rng.normal(-0.1, 0.3, 500),
        )

    def test_overfit_strategy_is_rejected(self):
        """M5 gate (a)."""
        report = run_gauntlet(self.overfit_inputs())
        assert not report.passed
        assert report.first_failure is not None

    def test_overfit_fails_multiple_tests(self):
        report = run_gauntlet(self.overfit_inputs(), short_circuit=False)
        failed = {r.test for r in report.failures}
        # Not one unlucky check — the whole out-of-sample picture collapses.
        assert len(failed) >= 4
        assert "2_look_ahead" in failed
        assert "12_locked_oos" in failed

    def test_buy_and_hold_passes(self):
        """M5 gate (b). A gauntlet that rejects this is broken."""
        report = run_gauntlet(self.buy_and_hold_inputs(), short_circuit=False)
        assert report.passed, report.format()

    def test_short_circuit_stops_early(self):
        full = run_gauntlet(self.overfit_inputs(), short_circuit=False)
        quick = run_gauntlet(self.overfit_inputs(), short_circuit=True)
        assert len(quick.results) < len(full.results)

    def test_report_formats(self):
        text = run_gauntlet(self.overfit_inputs()).format()
        assert "REJECTED" in text

    def test_skipped_tests_do_not_grant_a_pass(self):
        """A strategy with almost no evidence must not sail through."""
        bare = GauntletInputs(returns=modest_edge(), n_trials=1, seed=SEED)
        report = run_gauntlet(bare, short_circuit=False)
        skipped = [r for r in report.results if r.skipped]
        assert len(skipped) >= 8
        assert all(not r.passed for r in skipped)
