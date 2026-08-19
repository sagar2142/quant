"""Combining signals (§6).

Three things can go wrong when factors are combined, each producing a
plausible number rather than an error:

    the composite double-counts an effect two factors share
    a z-score is computed across the pooled sample, ranking calm periods
      against volatile ones
    a factor with a negative IC gets a negative weight and is silently
      inverted into the composite

Each is tested against a panel where the right answer is known.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.research.composite import (
    MIN_FACTORS,
    WINSOR_LIMIT,
    CompositeSpec,
    combine_factors,
    factor_correlations,
    orthogonalise,
    zscore_by_session,
)
from quant.research.factors import Factor
from quant.research.ic import ROLLING_WINDOW, rolling_ic

SEED = 20260820


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


def wide_panel(names: int = 40, sessions: int = 400, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return panel(
        {
            f"N{i:02d}": list(
                100.0 * np.exp(np.cumsum(rng.normal((i - names / 2) * 0.0004, 0.012, sessions)))
            )
            for i in range(names)
        }
    )


class TestZScore:
    def frame(self) -> pl.DataFrame:
        times = [datetime(2024, 1, 1, tzinfo=UTC)] * 4 + [datetime(2024, 1, 2, tzinfo=UTC)] * 4
        return pl.DataFrame(
            {"event_time": times, "raw": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0]},
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )

    def test_scored_within_each_session(self):
        """A pooled z-score would rank the second session entirely above the
        first, measuring the calendar rather than the cross-section."""
        out = self.frame().with_columns(zscore_by_session("raw").alias("z"))
        first = out.head(4)["z"].to_list()
        second = out.tail(4)["z"].to_list()
        assert first == pytest.approx(second)

    def test_mean_is_zero_per_session(self):
        out = self.frame().with_columns(zscore_by_session("raw").alias("z"))
        assert out.head(4)["z"].mean() == pytest.approx(0.0)

    def test_outliers_are_winsorised(self):
        """One name at 40 sigma — which a ratio factor produces whenever its
        denominator nears zero — would otherwise own the session."""
        times = [datetime(2024, 1, 1, tzinfo=UTC)] * 30
        values = [1.0] * 29 + [10_000.0]
        frame = pl.DataFrame(
            {"event_time": times, "raw": values},
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )
        out = frame.with_columns(zscore_by_session("raw").alias("z"))
        assert out["z"].max() == pytest.approx(WINSOR_LIMIT)

    def test_a_flat_session_scores_zero_not_infinity(self):
        times = [datetime(2024, 1, 1, tzinfo=UTC)] * 4
        frame = pl.DataFrame(
            {"event_time": times, "raw": [5.0] * 4},
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )
        out = frame.with_columns(zscore_by_session("raw").alias("z"))
        assert out["z"].to_list() == [0.0] * 4


class TestOrthogonalisation:
    def duplicated(self, n: int = 40) -> pl.DataFrame:
        """Two columns that are the same signal plus a little noise."""
        rng = np.random.default_rng(SEED)
        base = rng.normal(0, 1, n)
        return pl.DataFrame(
            {
                "event_time": [datetime(2024, 1, 1, tzinfo=UTC)] * n,
                "a": base,
                "b": base + rng.normal(0, 0.05, n),
            },
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )

    def test_overlap_is_removed(self):
        """Two near-identical factors must not both contribute the same bet."""
        frame = self.duplicated()
        before = abs(np.corrcoef(frame["a"], frame["b"])[0, 1])
        after_frame = orthogonalise(frame, ["a", "b"])
        after = abs(np.corrcoef(after_frame["a"], after_frame["b"])[0, 1])
        assert before > 0.9
        assert after < 0.05

    def test_the_first_factor_keeps_its_content(self):
        """Order decides who keeps the shared variance — the first listed.

        It is rescaled to unit size like every column, so the test is that it
        still carries the same information, not the same numbers.
        """
        frame = self.duplicated()
        out = orthogonalise(frame, ["a", "b"])
        assert abs(np.corrcoef(out["a"], frame["a"])[0, 1]) == pytest.approx(1.0)

    def test_residuals_are_rescaled_to_unit_size(self):
        """Without rescaling a later factor fades out of the composite."""
        out = orthogonalise(self.duplicated(), ["a", "b"])
        assert float(np.std(out["b"].to_numpy())) == pytest.approx(1.0, abs=0.05)

    def test_a_single_column_is_returned_unchanged(self):
        frame = self.duplicated()
        assert orthogonalise(frame, ["a"]).equals(frame)

    def test_thin_sessions_are_left_alone(self):
        """A cross-section of three names cannot support a regression."""
        rng = np.random.default_rng(SEED)
        frame = pl.DataFrame(
            {
                "event_time": [datetime(2024, 1, 1, tzinfo=UTC)] * 3,
                "a": rng.normal(0, 1, 3),
                "b": rng.normal(0, 1, 3),
            },
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )
        assert orthogonalise(frame, ["a", "b"])["b"].to_list() == frame["b"].to_list()


class TestFactorOverlap:
    def test_duplicate_factors_reduce_the_effective_count(self):
        """momentum_12_1 and momentum_6_1 share most of their construction."""
        overlap = factor_correlations(
            wide_panel(), (Factor.MOMENTUM_12_1, Factor.MOMENTUM_6_1), window=400
        )
        assert overlap.effective_factors < 2.0
        assert overlap.redundant_pairs

    def test_independent_factors_score_near_the_full_count(self):
        """Momentum and short-horizon reversal are close to unrelated."""
        overlap = factor_correlations(
            wide_panel(), (Factor.MOMENTUM_12_1, Factor.REVERSAL_1D), window=400
        )
        assert overlap.effective_factors > 1.5

    def test_report_names_the_overlapping_pairs(self):
        overlap = factor_correlations(
            wide_panel(), (Factor.MOMENTUM_12_1, Factor.MOMENTUM_6_1), window=400
        )
        assert "overlapping pairs" in overlap.format()


class TestComposite:
    def test_a_composite_needs_two_factors(self):
        with pytest.raises(ValueError, match="at least two"):
            CompositeSpec(factors=(Factor.MOMENTUM_12_1,))
        assert MIN_FACTORS == 2

    def test_a_factor_cannot_appear_twice(self):
        """Listing a factor twice would double its weight silently."""
        with pytest.raises(ValueError, match="twice"):
            CompositeSpec(factors=(Factor.MOMENTUM_12_1, Factor.MOMENTUM_12_1))

    def test_weights_sum_to_one(self):
        _, weights = combine_factors(
            wide_panel(),
            CompositeSpec(factors=(Factor.MOMENTUM_12_1, Factor.VOLATILITY_60), window=400),
            (21,),
        )
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_equal_weighting_is_available(self):
        _, weights = combine_factors(
            wide_panel(),
            CompositeSpec(
                factors=(Factor.MOMENTUM_12_1, Factor.VOLATILITY_60),
                window=400,
                ic_weight=False,
            ),
            (21,),
        )
        assert list(weights.values()) == pytest.approx([0.5, 0.5])

    def test_no_factor_receives_a_negative_weight(self):
        """A negative weight silently inverts a signal. When nothing has a
        positive IC the composite falls back to equal weights and the result's
        own IC says plainly that there is nothing there."""
        _, weights = combine_factors(
            wide_panel(),
            CompositeSpec(
                factors=(Factor.REVERSAL_1D, Factor.REVERSAL_5D, Factor.VOLUME_SHOCK),
                window=400,
            ),
            (21,),
        )
        assert all(w >= 0 for w in weights.values())

    def test_output_shape_matches_a_single_factor(self):
        """So every existing analysis works on a composite unchanged."""
        scored, _ = combine_factors(
            wide_panel(),
            CompositeSpec(factors=(Factor.MOMENTUM_12_1, Factor.VOLATILITY_60), window=400),
            (21,),
        )
        assert set(scored.columns) == {"event_time", "symbol", "signal", "fwd_21"}
        assert scored["signal"].is_finite().all()

    def test_orthogonalisation_can_be_disabled(self):
        spec = CompositeSpec(
            factors=(Factor.MOMENTUM_12_1, Factor.MOMENTUM_6_1),
            window=400,
            orthogonalise=False,
        )
        scored, _ = combine_factors(wide_panel(), spec, (21,))
        assert not scored.is_empty()


class TestRollingIC:
    def test_a_sample_shorter_than_the_window_produces_none(self):
        """A partial window is a different statistic wearing the same label."""
        from quant.research.factors import FactorSpec, build_factor

        scored = build_factor(wide_panel(sessions=200), FactorSpec(Factor.MOMENTUM_1M), (21,))
        assert rolling_ic(scored, 21, window=ROLLING_WINDOW) == []

    def test_windows_are_produced_on_a_long_sample(self):
        from quant.research.factors import FactorSpec, build_factor

        scored = build_factor(wide_panel(sessions=700), FactorSpec(Factor.MOMENTUM_1M), (21,))
        windows = rolling_ic(scored, 21, window=100)
        assert windows
        assert all(w.sessions == 100 for w in windows)

    def test_windows_are_ordered_in_time(self):
        from quant.research.factors import FactorSpec, build_factor

        scored = build_factor(wide_panel(sessions=700), FactorSpec(Factor.MOMENTUM_1M), (21,))
        ends = [w.end for w in rolling_ic(scored, 21, window=100)]
        assert ends == sorted(ends)
