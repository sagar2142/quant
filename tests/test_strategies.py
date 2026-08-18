"""Strategy families — MASTER_PLAN Part 6.

Strategies are pure functions of an observable window, so they test cleanly
without an engine: construct a `MarketView`, assert the weights.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from core.instruments import InstrumentId
from quant.strategies.base import MarketView, TargetWeights
from quant.strategies.baselines import BuyAndHold, CrossSectionalMomentum, SmaCrossover
from quant.strategies.mean_reversion import PairsTrading, ZScoreReversion, half_life
from quant.strategies.overlays import VolatilityTarget, vol_scale

A = InstrumentId("X:A")
B = InstrumentId("X:B")
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def view(series: dict[InstrumentId, list[float]]) -> MarketView:
    """A MarketView from per-instrument close series of equal length."""
    length = len(next(iter(series.values())))
    times = [T0 + timedelta(days=i) for i in range(length)]
    frames = []
    for instrument_id, closes in series.items():
        frames.append(
            pl.DataFrame(
                {
                    "event_time": times,
                    "receive_time": times,
                    "instrument_id": [instrument_id] * length,
                    "open": closes,
                    "high": [c * 1.001 for c in closes],
                    "low": [c * 0.999 for c in closes],
                    "close": closes,
                    "volume": [1e6] * length,
                },
                schema_overrides={
                    "event_time": pl.Datetime("us", "UTC"),
                    "receive_time": pl.Datetime("us", "UTC"),
                },
            )
        )
    return MarketView(
        as_of=as_decision_time(times[-1] + timedelta(days=1)),
        history=pl.concat(frames),
        universe=tuple(series),
    )


class TestBuyAndHold:
    def test_equal_weights(self):
        weights = BuyAndHold()(view({A: [100.0] * 5, B: [50.0] * 5})).weights
        assert weights[A] == weights[B]

    def test_gross_respected(self):
        result = BuyAndHold(gross=Decimal("0.5"))(view({A: [100.0] * 5, B: [50.0] * 5}))
        assert result.gross == Decimal("0.5")

    def test_empty_universe(self):
        result = BuyAndHold()(
            MarketView(as_of=as_decision_time(T0), history=pl.DataFrame(), universe=())
        )
        assert result.weights == {}


class TestSmaCrossover:
    def test_long_when_fast_above_slow(self):
        rising = [100.0 + i for i in range(30)]
        assert SmaCrossover(fast=3, slow=10)(view({A: rising})).weights[A] > 0

    def test_flat_when_fast_below_slow(self):
        falling = [100.0 - i for i in range(30)]
        assert SmaCrossover(fast=3, slow=10)(view({A: falling})).weights == {}

    def test_short_when_permitted(self):
        falling = [100.0 - i for i in range(30)]
        strategy = SmaCrossover(fast=3, slow=10, allow_short=True)
        assert strategy(view({A: falling})).weights[A] < 0

    def test_insufficient_history_produces_nothing(self):
        assert SmaCrossover(fast=3, slow=10)(view({A: [100.0] * 5})).weights == {}

    def test_inverted_windows_rejected(self):
        with pytest.raises(ValueError, match="must be shorter"):
            SmaCrossover(fast=20, slow=5)


class TestCrossSectionalMomentum:
    """The plan's strongest first live candidate (§6)."""

    def test_longs_the_winners(self):
        winner = [100.0 * (1.003**i) for i in range(300)]
        loser = [100.0 * (0.998**i) for i in range(300)]
        strategy = CrossSectionalMomentum(
            lookback_bars=250, skip_bars=20, top_fraction=Decimal("0.5")
        )
        weights = strategy(view({A: winner, B: loser})).weights
        assert weights.get(A, Decimal(0)) > 0
        assert B not in weights or weights[B] <= 0

    def test_long_short_when_permitted(self):
        winner = [100.0 * (1.003**i) for i in range(300)]
        loser = [100.0 * (0.998**i) for i in range(300)]
        strategy = CrossSectionalMomentum(
            lookback_bars=250, skip_bars=20, top_fraction=Decimal("0.5"), long_only=False
        )
        weights = strategy(view({A: winner, B: loser})).weights
        assert weights[A] > 0
        assert weights[B] < 0

    def test_skip_window_must_be_shorter(self):
        with pytest.raises(ValueError, match="skip window"):
            CrossSectionalMomentum(lookback_bars=50, skip_bars=60)

    def test_insufficient_history(self):
        assert CrossSectionalMomentum(lookback_bars=250)(view({A: [100.0] * 30})).weights == {}


class TestZScoreReversion:
    def noisy(self, final_offset: float, seed: int = 3) -> list[float]:
        """Noisy series with a controlled final deviation.

        A flat series plus one spike is the wrong fixture: a single outlier
        among n points has z ~= sqrt(n-1), which for a 40-bar window is ~6.2 and
        trips `max_zscore` regardless of the spike's size. Real deviations sit
        against real dispersion.
        """
        rng = np.random.default_rng(seed)
        prices = list(100.0 + rng.normal(0, 1.0, 60))
        prices[-1] = 100.0 + final_offset
        return prices

    def test_shorts_a_spike(self):
        # ~+2.5 sigma against unit dispersion.
        assert ZScoreReversion(lookback=40)(view({A: self.noisy(2.5)})).weights[A] < 0

    def test_longs_a_dip(self):
        assert ZScoreReversion(lookback=40)(view({A: self.noisy(-2.5)})).weights[A] > 0

    def test_no_position_inside_the_band(self):
        rng = np.random.default_rng(1)
        prices = list(100.0 + rng.normal(0, 0.5, 60))
        assert ZScoreReversion(lookback=40, entry_z=3.0)(view({A: prices})).weights == {}

    def test_extreme_deviation_abandoned(self):
        """Beyond max_zscore, assume the level moved rather than deviated.

        A mean-reversion strategy adds to losers by construction; this is the
        guard that stops it doubling down on a broken relationship.
        """
        prices = [100.0] * 60 + [1000.0]
        strategy = ZScoreReversion(lookback=40, entry_z=2.0, max_zscore=4.0)
        assert strategy(view({A: prices})).weights == {}

    def test_constant_series_has_no_signal(self):
        assert ZScoreReversion(lookback=40)(view({A: [100.0] * 60})).weights == {}

    def test_exit_band_must_be_inside_entry(self):
        with pytest.raises(ValueError, match="must be inside"):
            ZScoreReversion(entry_z=1.0, exit_z=2.0)

    def test_max_zscore_must_exceed_entry(self):
        with pytest.raises(ValueError, match="must exceed"):
            ZScoreReversion(entry_z=3.0, max_zscore=2.0)


class TestHalfLife:
    """The number that decides whether a pair is tradable at all."""

    def test_fast_reverting_series(self):
        rng = np.random.default_rng(7)
        # Strong pull to zero: should decay in a handful of bars.
        series = np.zeros(300)
        for i in range(1, 300):
            series[i] = 0.5 * series[i - 1] + rng.normal(0, 1)
        assert 0 < half_life(series) < 5

    def test_random_walk_is_not_mean_reverting(self):
        rng = np.random.default_rng(7)
        walk = np.cumsum(rng.normal(0, 1, 500))
        # A drift is not a spread. Either infinite or implausibly long.
        assert half_life(walk) > 50

    def test_short_series_returns_infinity(self):
        assert half_life(np.array([1.0, 2.0, 3.0])) == float("inf")

    def test_constant_series_returns_infinity(self):
        assert half_life(np.zeros(100)) == float("inf")


class TestPairsTrading:
    def cointegrated(self, n: int = 200) -> dict[InstrumentId, list[float]]:
        rng = np.random.default_rng(11)
        base = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        # B tracks A with a fast-reverting spread.
        spread = np.zeros(n)
        for i in range(1, n):
            spread[i] = 0.6 * spread[i - 1] + rng.normal(0, 0.4)
        return {A: list(base + spread), B: list(base)}

    def test_no_position_when_spread_is_normal(self):
        strategy = PairsTrading(A, B, lookback=100, entry_z=3.0)
        assert strategy(view(self.cointegrated())).weights == {}

    def test_legs_are_opposite_signs(self):
        prices = self.cointegrated()
        prices[A][-1] += 8.0  # push the spread wide
        weights = PairsTrading(A, B, lookback=100, entry_z=1.5)(view(prices)).weights
        if weights:
            assert weights[A] * weights[B] < 0

    def test_drifting_pair_rejected_by_half_life(self):
        """Correlation is not cointegration."""
        rng = np.random.default_rng(3)
        # Two independent random walks: correlated-looking, never converging.
        left = list(100.0 + np.cumsum(rng.normal(0.1, 1.0, 200)))
        right = list(100.0 + np.cumsum(rng.normal(-0.1, 1.0, 200)))
        strategy = PairsTrading(A, B, lookback=100, entry_z=1.0, max_half_life=10.0)
        assert strategy(view({A: left, B: right})).weights == {}

    def test_insufficient_history(self):
        assert (
            PairsTrading(A, B, lookback=100)(view({A: [100.0] * 20, B: [100.0] * 20})).weights == {}
        )

    def test_exit_band_validated(self):
        with pytest.raises(ValueError, match="must be inside"):
            PairsTrading(A, B, entry_z=1.0, exit_z=1.5)


class TestVolatilityTarget:
    """§6 — nearly free alpha, because it needs no new edge."""

    def test_scale_is_inverse_to_volatility(self):
        calm = vol_scale(0.10, target_vol=0.15, max_leverage=3.0)
        wild = vol_scale(0.40, target_vol=0.15, max_leverage=3.0)
        assert calm > wild

    def test_leverage_is_capped(self):
        # A very quiet market must not produce an enormous position.
        assert vol_scale(0.001, target_vol=0.15, max_leverage=2.0) == Decimal(2)

    def test_unmeasurable_volatility_means_no_position(self):
        # Not 1.0: an unknown risk level is not an acceptable one.
        assert vol_scale(0.0, target_vol=0.15, max_leverage=2.0) == Decimal(0)
        assert vol_scale(float("nan"), target_vol=0.15, max_leverage=2.0) == Decimal(0)

    def test_preserves_relative_views(self):
        rng = np.random.default_rng(5)
        prices_a = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, 120)))
        prices_b = list(100.0 * np.cumprod(1 + rng.normal(0.001, 0.01, 120)))
        market = view({A: prices_a, B: prices_b})

        inner = BuyAndHold()
        overlay = VolatilityTarget(inner, target_vol=0.15, lookback=60)
        scaled = overlay(market).weights
        if scaled:
            # Equal-weighted in, equal-weighted out — only magnitude changed.
            assert scaled[A] == scaled[B]

    def test_quiet_market_scales_up(self):
        rng = np.random.default_rng(5)
        calm = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.002, 120)))
        wild = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.05, 120)))
        overlay = VolatilityTarget(BuyAndHold(), target_vol=0.15, lookback=60)
        calm_gross = overlay(view({A: calm})).gross
        # Fresh overlay: the band is stateful by design.
        overlay = VolatilityTarget(BuyAndHold(), target_vol=0.15, lookback=60)
        wild_gross = overlay(view({A: wild})).gross
        assert calm_gross > wild_gross

    def test_rebalance_band_suppresses_small_resizes(self):
        rng = np.random.default_rng(5)
        prices = list(100.0 * np.cumprod(1 + rng.normal(0.0, 0.01, 200)))
        overlay = VolatilityTarget(BuyAndHold(), target_vol=0.15, lookback=60, rebalance_band=0.50)
        first = overlay(view({A: prices})).gross
        second = overlay(view({A: prices[:-1]})).gross
        # A tiny volatility change must not trigger a resize.
        assert first == second

    def test_short_lookback_rejected(self):
        with pytest.raises(ValueError, match="noise"):
            VolatilityTarget(BuyAndHold(), lookback=5)

    def test_invalid_target_rejected(self):
        with pytest.raises(ValueError, match="target_vol"):
            VolatilityTarget(BuyAndHold(), target_vol=0.0)

    def test_empty_inner_stays_empty(self):
        overlay = VolatilityTarget(SmaCrossover(fast=3, slow=10), lookback=60)
        assert overlay(view({A: [100.0 - i for i in range(120)]})).weights == {}


class TestTargetWeights:
    def test_gross_and_net(self):
        weights = TargetWeights(
            as_of=as_decision_time(T0), weights={A: Decimal("0.6"), B: Decimal("-0.4")}
        )
        assert weights.gross == Decimal("1.0")
        assert weights.net == Decimal("0.2")

    def test_clipping_caps_single_names(self):
        weights = TargetWeights(as_of=as_decision_time(T0), weights={A: Decimal("0.9")})
        assert weights.clipped(Decimal("0.1"), Decimal(1)).weights[A] == Decimal("0.1")

    def test_zero_weights_dropped(self):
        weights = TargetWeights(
            as_of=as_decision_time(T0), weights={A: Decimal("0.5"), B: Decimal(0)}
        )
        assert B not in weights.clipped(Decimal(1), Decimal(1)).weights
