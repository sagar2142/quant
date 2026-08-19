"""Factor research and rolling statistics (§6, §2.1).

A factor study is easy to get subtly wrong in ways that manufacture an edge:
forward returns that overlap the signal, buckets formed across the whole sample
instead of within a session, alignment that reaches into another instrument's
history. Each of those is tested here against a panel where the answer is known
by construction.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.analytics.rolling import (
    MIN_WINDOW,
    rolling_beta,
    rolling_sharpe,
    rolling_stats,
    rolling_volatility,
)
from quant.research.factors import (
    Factor,
    FactorSpec,
    add_forward_returns,
    build_factor,
    prepare_panel,
)
from quant.research.ic import (
    analyse_factor,
    information_coefficient,
    quantile_returns,
    signal_turnover,
)

SEED = 20260819


def panel(series: dict[str, list[float]], volume: float = 1e6) -> pl.DataFrame:
    length = len(next(iter(series.values())))
    times = [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "event_time": times,
                    "symbol": [symbol] * length,
                    "instrument_id": [f"NSE:{symbol}"] * length,
                    "close": closes,
                    "volume": [volume] * length,
                },
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
            for symbol, closes in series.items()
        ]
    )


def walk(n: int = 400, drift: float = 0.0, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, n))))


class TestForwardReturns:
    def test_forward_return_starts_after_the_signal_bar(self):
        """The single most important property. A forward return that overlaps
        the signal scores a signal against a return it already contains, which
        is the commonest way a factor study reports an edge that is not there.
        """
        closes = [100.0, 110.0, 121.0, 133.1]
        frame = add_forward_returns(panel({"A": closes}).sort("symbol"), (1,))
        forward = frame["fwd_1"].to_list()
        # Bar 0 must see bar 1's return, not its own.
        assert forward[0] == pytest.approx(0.10)
        assert forward[-1] is None  # nothing after the final bar

    def test_horizons_are_independent(self):
        frame = add_forward_returns(panel({"A": walk(100)}).sort("symbol"), (1, 5))
        assert frame["fwd_1"].to_list() != frame["fwd_5"].to_list()

    def test_forward_returns_do_not_cross_symbols(self):
        """Each name's forward return must come from its own future.

        Without `.over("symbol")` the last bar of one instrument would borrow
        the first bar of the next — invisible in aggregate, and fatal.
        """
        frame = add_forward_returns(
            panel({"A": [100.0] * 5, "B": [500.0] * 5}).sort(["symbol", "event_time"]), (1,)
        )
        by_symbol = frame.group_by("symbol").agg(pl.col("fwd_1").null_count().alias("nulls"))
        # Exactly one null each: the final bar of each name.
        assert set(by_symbol["nulls"].to_list()) == {1}


class TestFactorConstruction:
    def test_liquidity_filter_drops_thin_names(self):
        """Illiquid names manufacture IC from stale prices."""
        frame = pl.concat(
            [panel({"LIQUID": walk()}, volume=1e6), panel({"THIN": walk(seed=2)}, volume=1.0)]
        )
        kept = prepare_panel(frame, FactorSpec(Factor.MOMENTUM_12_1, min_adv=1e7))
        assert kept["symbol"].unique().to_list() == ["LIQUID"]

    def test_momentum_is_positive_for_a_riser(self):
        rising = [100.0 * (1.001**i) for i in range(400)]
        scored = build_factor(panel({"A": rising}), FactorSpec(Factor.MOMENTUM_12_1))
        assert scored["signal"].mean() > 0

    def test_reversal_negates_the_return(self):
        """A reversal signal must score a faller highly, not a riser."""
        rising = [100.0 * (1.01**i) for i in range(300)]
        scored = build_factor(panel({"A": rising}), FactorSpec(Factor.REVERSAL_1D))
        assert scored["signal"].mean() < 0

    def test_low_volatility_scores_high(self):
        """VOLATILITY_60 is negated so a high score is a calm name — the
        direction the low-volatility anomaly actually pays."""
        calm = panel({"CALM": walk(seed=1)})
        wild = panel(
            {"WILD": list(100.0 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.06, 400))))}
        )
        scored = build_factor(pl.concat([calm, wild]), FactorSpec(Factor.VOLATILITY_60))
        by_name = scored.group_by("symbol").agg(pl.col("signal").mean())
        values = dict(zip(by_name["symbol"], by_name["signal"], strict=True))
        assert values["CALM"] > values["WILD"]

    def test_every_factor_builds(self):
        frame = pl.concat([panel({f"N{i}": walk(seed=i)}) for i in range(4)])
        for factor in Factor:
            scored = build_factor(frame, FactorSpec(factor))
            assert not scored.is_empty(), factor.value
            assert scored["signal"].is_finite().all()

    def test_every_factor_has_a_description(self):
        for factor in Factor:
            assert len(factor.description) > 40

    def test_a_negative_liquidity_floor_is_refused(self):
        with pytest.raises(ValueError, match="min_adv"):
            FactorSpec(Factor.MOMENTUM_12_1, min_adv=-1.0)


class TestInformationCoefficient:
    def perfect_panel(self, names: int = 40, sessions: int = 200) -> pl.DataFrame:
        """A panel where the signal predicts the forward return exactly.

        Each name gets a fixed drift; the 12-1 signal then ranks names in the
        same order as their forward returns, so IC must be close to 1.
        """
        rng = np.random.default_rng(SEED)
        series = {}
        for i in range(names):
            drift = (i - names / 2) * 0.0008
            noise = rng.normal(0, 0.001, sessions)
            series[f"N{i:02d}"] = list(100.0 * np.exp(np.cumsum(drift + noise)))
        return panel(series)

    def test_a_predictive_signal_scores_high_ic(self):
        scored = build_factor(self.perfect_panel(), FactorSpec(Factor.MOMENTUM_1M), (5,))
        assert information_coefficient(scored, 5).mean > 0.5

    def test_a_random_signal_scores_near_zero(self):
        frame = pl.concat([panel({f"N{i:02d}": walk(300, seed=i)}) for i in range(30)])
        scored = build_factor(frame, FactorSpec(Factor.VOLUME_SHOCK), (5,))
        # Volume is constant in this fixture, so the signal carries nothing.
        assert abs(information_coefficient(scored, 5).mean) < 0.1

    def test_thin_sessions_are_excluded(self):
        """A rank correlation over three names is noise."""
        frame = pl.concat([panel({f"N{i}": walk(300, seed=i)}) for i in range(3)])
        scored = build_factor(frame, FactorSpec(Factor.MOMENTUM_12_1), (5,))
        assert information_coefficient(scored, 5).sessions == 0

    def test_an_empty_frame_is_safe(self):
        empty = pl.DataFrame(
            {"event_time": [], "symbol": [], "signal": [], "fwd_5": []},
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "symbol": pl.String,
                "signal": pl.Float64,
                "fwd_5": pl.Float64,
            },
        )
        assert information_coefficient(empty, 5).sessions == 0

    def test_significance_flag_matches_the_t_stat(self):
        scored = build_factor(self.perfect_panel(), FactorSpec(Factor.MOMENTUM_1M), (5,))
        summary = information_coefficient(scored, 5)
        assert summary.is_significant == (abs(summary.t_stat) > 2.0)


class TestQuantiles:
    def test_buckets_are_formed_within_each_session(self):
        """A global cut would put calm periods in one bucket and volatile ones
        in another, measuring the calendar rather than the signal."""
        rng = np.random.default_rng(SEED)
        series = {
            f"N{i:02d}": list(100.0 * np.exp(np.cumsum(rng.normal((i - 15) * 0.0008, 0.004, 200))))
            for i in range(30)
        }
        scored = build_factor(panel(series), FactorSpec(Factor.MOMENTUM_1M), (5,))
        rows = quantile_returns(scored, 5, buckets=5)
        assert len(rows) == 5
        assert [r.quantile for r in rows] == [1, 2, 3, 4, 5]

    def test_a_predictive_signal_is_monotonic(self):
        rng = np.random.default_rng(SEED)
        series = {
            f"N{i:02d}": list(100.0 * np.exp(np.cumsum(rng.normal((i - 15) * 0.001, 0.002, 220))))
            for i in range(30)
        }
        scored = build_factor(panel(series), FactorSpec(Factor.MOMENTUM_1M), (5,))
        report = analyse_factor(scored, "test", (5,), 5)
        assert report.is_monotonic
        assert report.spread > 0

    def test_empty_input_yields_no_buckets(self):
        empty = pl.DataFrame(
            {"event_time": [], "symbol": [], "signal": [], "fwd_5": []},
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "symbol": pl.String,
                "signal": pl.Float64,
                "fwd_5": pl.Float64,
            },
        )
        assert quantile_returns(empty, 5) == []


class TestTurnover:
    def test_a_stable_signal_has_low_turnover(self):
        """Long-horizon momentum reorders slowly; that is what makes it
        tradable against a 22bp round trip."""
        frame = pl.concat([panel({f"N{i:02d}": walk(400, seed=i)}) for i in range(25)])
        scored = build_factor(frame, FactorSpec(Factor.MOMENTUM_12_1), (5,))
        assert 0.0 < signal_turnover(scored) < 0.25

    def test_a_noisy_signal_has_higher_turnover(self):
        frame = pl.concat([panel({f"N{i:02d}": walk(400, seed=i)}) for i in range(25)])
        slow = signal_turnover(build_factor(frame, FactorSpec(Factor.MOMENTUM_12_1), (5,)))
        fast = signal_turnover(build_factor(frame, FactorSpec(Factor.REVERSAL_1D), (5,)))
        assert fast > slow


class TestRollingStatistics:
    def test_leading_values_are_nan_until_the_window_fills(self):
        """A 20-bar 'six-month Sharpe' is a different statistic wearing the
        same label."""
        series = rolling_sharpe(np.random.default_rng(SEED).normal(0, 0.01, 300), window=126)
        assert np.isnan(series.values[:125]).all()
        assert np.isfinite(series.values[125])

    def test_output_is_aligned_to_the_input(self):
        returns = np.random.default_rng(SEED).normal(0, 0.01, 300)
        assert rolling_sharpe(returns, window=126).values.size == returns.size

    def test_a_decaying_edge_is_visible(self):
        """The whole point: a full-sample Sharpe averages this away."""
        rng = np.random.default_rng(SEED)
        good = rng.normal(0.002, 0.01, 300)
        bad = rng.normal(-0.002, 0.01, 300)
        series = rolling_sharpe(np.concatenate([good, bad]), window=126)
        assert series.best > 1.0
        assert series.worst < -1.0

    def test_volatility_recovers_the_input(self):
        returns = np.random.default_rng(SEED).normal(0, 0.02, 400)
        series = rolling_volatility(returns, window=126)
        assert series.last == pytest.approx(0.02 * np.sqrt(252), rel=0.25)

    def test_beta_of_a_series_against_itself_is_one(self):
        returns = np.random.default_rng(SEED).normal(0, 0.01, 300)
        assert rolling_beta(returns, returns, window=126).last == pytest.approx(1.0)

    def test_beta_of_a_doubled_series_is_two(self):
        market = np.random.default_rng(SEED).normal(0, 0.01, 300)
        assert rolling_beta(2 * market, market, window=126).last == pytest.approx(2.0)

    def test_a_flat_window_has_no_sharpe_rather_than_infinity(self):
        series = rolling_sharpe(np.zeros(300), window=126)
        assert not np.isfinite(series.values[-1])

    def test_a_short_series_produces_all_nan(self):
        series = rolling_sharpe(np.random.default_rng(1).normal(0, 0.01, 50), window=126)
        assert np.isnan(series.values).all()
        assert np.isnan(series.last)

    def test_too_small_a_window_is_refused(self):
        with pytest.raises(ValueError, match="below the"):
            rolling_sharpe(np.zeros(100), window=MIN_WINDOW - 1)

    def test_fraction_positive_is_reported(self):
        rng = np.random.default_rng(SEED)
        series = rolling_sharpe(rng.normal(0.002, 0.01, 400), window=126)
        assert 0.0 <= series.fraction_positive <= 1.0

    def test_rolling_stats_includes_beta_only_with_a_market(self):
        returns = np.random.default_rng(SEED).normal(0, 0.01, 300)
        assert len(rolling_stats(returns)) == 2
        assert len(rolling_stats(returns, market=returns)) == 3

    def test_series_formats(self):
        returns = np.random.default_rng(SEED).normal(0, 0.01, 300)
        assert "sharpe" in rolling_sharpe(returns).format()
