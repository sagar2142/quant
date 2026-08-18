"""Gauntlet input generators (§5.4 tests 8, 9, 10).

These generators decide whether three of the twelve checks run at all, so the
thing worth testing is not that they return arrays — it is that they *detect the
failure each check exists to catch*. Each class below builds a strategy with a
known defect and asserts the generator surfaces it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from core.instruments import InstrumentId
from engine.validation.generators import (
    SamplingSpec,
    market_proxy,
    placebo_sharpes,
    regime_slices,
    universe_dropout_sharpes,
)

UNIVERSE = tuple(InstrumentId(f"NSE:{i:03d}") for i in range(20))


class TestUniverseDropout:
    """Test 8 — does the edge survive without its best names?"""

    def test_broad_edge_survives_dropout(self):
        """Every name contributes, so no subset is much worse than another."""

        def run(universe):
            rng = np.random.default_rng(len(universe))
            return rng.normal(0.001, 0.01, size=500)

        sharpes = universe_dropout_sharpes(run, UNIVERSE, SamplingSpec(seed=1, samples=20))
        assert sharpes.size == 20
        assert float(np.quantile(sharpes, 0.05)) > 0

    def test_concentrated_edge_is_exposed(self):
        """All the edge lives in one name.

        In aggregate this is indistinguishable from a broad strategy. Here it is
        obvious: every subset that drops the one name collapses to noise, and
        the 5th percentile goes negative — which is exactly the verdict test 8
        is looking for.
        """
        hero = UNIVERSE[0]

        def run(universe):
            rng = np.random.default_rng(abs(hash(universe)) % 2**31)
            drift = 0.002 if hero in universe else -0.001
            return rng.normal(drift, 0.01, size=500)

        sharpes = universe_dropout_sharpes(run, UNIVERSE, SamplingSpec(seed=1, samples=40))
        assert float(np.quantile(sharpes, 0.05)) < 0

    def test_same_seed_reproduces_the_samples(self):
        def run(universe):
            return np.full(100, len(universe) * 1e-4)

        first = universe_dropout_sharpes(run, UNIVERSE, SamplingSpec(seed=7, samples=15))
        second = universe_dropout_sharpes(run, UNIVERSE, SamplingSpec(seed=7, samples=15))
        assert np.array_equal(first, second)

    def test_empty_runs_are_dropped_not_counted_as_zero(self):
        """A subset that produced nothing is absent, not a zero Sharpe.

        Recording it as zero would drag the 5th percentile toward a number no
        run actually produced.
        """
        calls = {"n": 0}

        def run(universe):
            calls["n"] += 1
            return np.array([]) if calls["n"] % 2 else np.full(100, 0.001)

        sharpes = universe_dropout_sharpes(run, UNIVERSE, SamplingSpec(seed=3, samples=10))
        assert sharpes.size == 5

    def test_universe_too_small_to_perturb_is_refused(self):
        with pytest.raises(ValueError, match="too small to drop"):
            universe_dropout_sharpes(
                lambda universe: np.zeros(10), UNIVERSE[:2], SamplingSpec(seed=1, samples=10)
            )

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5])
    def test_nonsense_drop_fraction_is_refused(self, fraction):
        with pytest.raises(ValueError, match="drop_fraction"):
            universe_dropout_sharpes(
                lambda universe: np.zeros(10),
                UNIVERSE,
                SamplingSpec(seed=1, samples=10),
                drop_fraction=fraction,
            )


class TestMarketProxy:
    def panel(self, closes: dict[str, list[float]]) -> pl.DataFrame:
        rows = []
        for name, series in closes.items():
            for i, close in enumerate(series):
                rows.append(
                    {
                        "event_time": datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i),
                        "instrument_id": name,
                        "close": close,
                    }
                )
        return pl.DataFrame(rows)

    def test_equal_weight_return_is_the_mean_of_the_names(self):
        panel = self.panel({"NSE:000": [100.0, 110.0], "NSE:001": [100.0, 90.0]})
        proxy = market_proxy(panel, [InstrumentId("NSE:000"), InstrumentId("NSE:001")])
        assert proxy.height == 1
        assert proxy["market_return"][0] == pytest.approx(0.0)

    def test_names_outside_the_universe_are_excluded(self):
        """The yardstick is the market the strategy actually faces."""
        panel = self.panel({"NSE:000": [100.0, 110.0], "NSE:999": [100.0, 10.0]})
        proxy = market_proxy(panel, [InstrumentId("NSE:000")])
        assert proxy["market_return"][0] == pytest.approx(0.10)

    def test_empty_universe_yields_no_rows(self):
        panel = self.panel({"NSE:000": [100.0, 110.0]})
        assert market_proxy(panel, []).height == 0


class TestRegimeSlices:
    """Test 9 — does it work in more than one market condition?"""

    def market(self, up: int, down: int) -> np.ndarray:
        return np.concatenate([np.full(up, 0.002), np.full(down, -0.002)])

    def test_bull_and_bear_are_both_found(self):
        market = self.market(300, 300)
        slices = regime_slices(np.full(600, 0.001), market, window=60, min_bars=30)
        assert "bull" in slices
        assert "bear" in slices

    def test_a_one_regime_sample_yields_one_regime(self):
        """A backtest that only ever saw a bull market cannot claim otherwise.

        This is the case test 9 exists for: the strategy is not necessarily
        bad, but the evidence does not cover more than one condition and the
        report must say so rather than quietly counting one regime as enough.
        """
        slices = regime_slices(np.full(400, 0.001), np.full(400, 0.002), window=60, min_bars=30)
        assert "bear" not in slices

    def test_short_regimes_are_dropped(self):
        """A Sharpe on twelve bars is not evidence about a regime."""
        market = self.market(300, 12)
        slices = regime_slices(np.full(312, 0.001), market, window=60, min_bars=60)
        assert "bear" not in slices

    def test_high_volatility_is_labelled(self):
        calm = np.full(200, 0.001)
        rng = np.random.default_rng(0)
        wild = rng.normal(0.0, 0.05, size=200)
        market = np.concatenate([calm, wild])
        slices = regime_slices(np.full(400, 0.001), market, window=60, min_bars=30)
        assert "high_vol" in slices

    def test_too_short_a_sample_yields_nothing(self):
        assert regime_slices(np.full(30, 0.001), np.full(30, 0.001), window=60) == {}

    def test_labels_never_use_future_data(self):
        """Truncating the tail must not change the labels that precede it.

        A centred window would fail this. Nothing here feeds a trading decision,
        but a look-ahead habit in analysis code eventually becomes one in
        decision code.
        """
        market = self.market(200, 200)
        strategy = np.arange(400, dtype=np.float64)
        full = regime_slices(strategy, market, window=60, min_bars=30)
        # The bull regime is entirely inside the first 200 bars, so cutting the
        # bear tail must leave it untouched.
        truncated = regime_slices(strategy[:200], market[:200], window=60, min_bars=30)
        assert np.array_equal(truncated["bull"], full["bull"][: truncated["bull"].size])


class TestPlaceboSharpes:
    """Test 10 — does a coin flip do just as well?"""

    def test_one_sharpe_per_sample(self):
        sharpes = placebo_sharpes(
            lambda seed: np.random.default_rng(seed).normal(0, 0.01, 300),
            SamplingSpec(seed=5, samples=25),
        )
        assert sharpes.size == 25

    def test_same_seed_reproduces_the_distribution(self):
        def run(seed):
            return np.random.default_rng(seed).normal(0, 0.01, 300)

        assert np.array_equal(
            placebo_sharpes(run, SamplingSpec(seed=11, samples=20)),
            placebo_sharpes(run, SamplingSpec(seed=11, samples=20)),
        )

    def test_seed_sequence_is_independent_of_failures(self):
        """Seeds are drawn up front, so a run that produces nothing does not
        shift every later sample onto a different seed."""
        seen: list[int] = []

        def run(seed):
            seen.append(seed)
            return np.array([]) if len(seen) == 1 else np.full(100, 0.001)

        placebo_sharpes(run, SamplingSpec(seed=2, samples=5))
        first_pass = list(seen)

        seen.clear()

        def always_works(seed):
            seen.append(seed)
            return np.full(100, 0.001)

        placebo_sharpes(always_works, SamplingSpec(seed=2, samples=5))
        assert seen == first_pass

    def test_empty_runs_are_dropped(self):
        assert (
            placebo_sharpes(lambda seed: np.array([]), SamplingSpec(seed=1, samples=10)).size == 0
        )
