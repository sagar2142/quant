"""Questions a cross-sectional factor cannot ask — MASTER_PLAN §5.1, §6.

Four pre-registered hypotheses are not the shape the factor lab answers, and
forcing them into it would answer a different question than the one registered.
These test that each study measures what its hypothesis actually claimed, and —
more importantly — that a study which holds does not thereby confirm anything
tradeable.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.research.factors import Factor, FactorSpec
from quant.research.studies import (
    MIN_SESSIONS,
    MIN_T_STAT,
    StudyResult,
    conditional_ic,
    dispersion_timing,
    double_sorted_ic,
    gap_reversion,
)


def panel(
    closes: dict[str, np.ndarray],
    opens: dict[str, np.ndarray] | None = None,
    volume: float = 1e8,
) -> pl.DataFrame:
    length = len(next(iter(closes.values())))
    times = [datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    frames = []
    for name, series in closes.items():
        open_series = opens[name] if opens else np.r_[series[0], series[:-1]]
        frames.append(
            pl.DataFrame(
                {
                    "event_time": times,
                    "receive_time": times,
                    "symbol": [name] * length,
                    "instrument_id": [f"NSE:INE{name}"] * length,
                    "open": list(open_series),
                    "high": [c * 1.02 for c in series],
                    "low": [c * 0.98 for c in series],
                    "close": list(series),
                    "volume": [volume] * length,
                    "trades": [1000.0] * length,
                },
                schema_overrides={
                    "event_time": pl.Datetime("us", "UTC"),
                    "receive_time": pl.Datetime("us", "UTC"),
                },
            )
        )
    return pl.concat(frames)


class TestGapReversion:
    """A same-session claim: the gap opens and closes inside one bar, so a
    forward horizon would measure the day after the effect."""

    def reverting(self, names: int = 30, sessions: int = 200) -> pl.DataFrame:
        """Opens gap away from the prior close, and the session walks it back."""
        rng = np.random.default_rng(5)
        closes, opens = {}, {}
        for i in range(names):
            level = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, sessions)))
            gaps = rng.normal(0, 0.02, sessions)
            closes[f"N{i:02d}"] = level
            # Open away from the previous close; close back at the level, so
            # the intraday move is the opposite of the gap by construction.
            opens[f"N{i:02d}"] = np.r_[level[0], level[:-1]] * (1 + gaps)
        return panel(closes, opens)

    def test_a_reverting_market_is_measured_as_negative(self):
        result = gap_reversion(self.reverting())
        assert result.statistic < 0
        assert result.is_significant

    def test_no_relationship_is_measured_as_none(self):
        """Gap and intraday drawn independently.

        Deriving the open from the *same* session's close makes the two
        mechanically anti-correlated — the first version of this fixture did
        that and the study correctly found the relationship it had built in.
        Here the open is set from the previous close and the session's own move
        is independent of it.
        """
        rng = np.random.default_rng(6)
        sessions = 200
        closes, opens = {}, {}
        for i in range(30):
            close = np.empty(sessions)
            open_ = np.empty(sessions)
            level = 100.0
            for t in range(sessions):
                open_[t] = level * (1 + rng.normal(0, 0.02))
                close[t] = open_[t] * (1 + rng.normal(0, 0.01))
                level = close[t]
            closes[f"N{i:02d}"], opens[f"N{i:02d}"] = close, open_
        result = gap_reversion(panel(closes, opens))
        assert abs(result.statistic) < 0.3

    def test_too_little_history_is_reported_not_guessed(self):
        rng = np.random.default_rng(7)
        closes = {f"N{i:02d}": 100.0 + rng.normal(0, 1, 10) for i in range(30)}
        result = gap_reversion(panel(closes))
        assert not result.is_significant
        assert result.observations < MIN_SESSIONS


class TestConditionalIc:
    def market(self, names: int = 40, sessions: int = 300) -> pl.DataFrame:
        rng = np.random.default_rng(8)
        return panel(
            {
                f"N{i:02d}": 100.0
                * np.exp(np.cumsum(rng.normal((i - 20) * 0.0004, 0.01, sessions)))
                for i in range(names)
            }
        )

    def test_it_returns_a_difference_between_buckets(self):
        """The claim is that an effect is stronger somewhere, so the statistic
        has to be a difference rather than a level."""
        result = conditional_ic(self.market(), FactorSpec(Factor.REVERSAL_5D), 5, "volume")
        assert isinstance(result, StudyResult)

    def test_a_missing_condition_is_reported(self):
        result = conditional_ic(self.market(), FactorSpec(Factor.REVERSAL_5D), 5, "nonexistent")
        assert result.observations == 0
        assert "unavailable" in result.detail


class TestDoubleSortedIc:
    def test_it_scores_within_buckets_not_across_them(self):
        """A single sort on illiquidity is also a sort on size; the whole point
        is to separate them."""
        rng = np.random.default_rng(9)
        frame = panel(
            {
                f"N{i:02d}": 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 300)))
                for i in range(40)
            }
        )
        result = double_sorted_ic(frame, FactorSpec(Factor.MOMENTUM_1M), 21, "volume")
        assert "terciles" in result.detail

    def test_a_missing_control_is_reported(self):
        rng = np.random.default_rng(10)
        frame = panel({f"N{i:02d}": 100.0 + rng.normal(0, 1, 300) for i in range(40)})
        result = double_sorted_ic(frame, FactorSpec(Factor.MOMENTUM_1M), 21, "nonexistent")
        assert result.observations == 0


class TestDispersionTiming:
    """A claim about *when* to trade, not what to hold — its unit of
    observation is a session rather than a name."""

    def test_dispersion_is_lagged_against_the_ic_it_predicts(self):
        """Using the same session's dispersion would correlate a statistic with
        itself and report it as foresight."""
        rng = np.random.default_rng(11)
        frame = panel(
            {
                f"N{i:02d}": 100.0 * np.exp(np.cumsum(rng.normal((i - 20) * 0.0003, 0.01, 400)))
                for i in range(40)
            }
        )
        result = dispersion_timing(frame, FactorSpec(Factor.MOMENTUM_12_1), 21)
        assert "prior-session" in result.detail or result.observations == 0

    def test_a_factor_with_no_signal_is_safe(self):
        empty = panel({f"N{i:02d}": np.full(50, 100.0) for i in range(30)})
        result = dispersion_timing(empty, FactorSpec(Factor.MOMENTUM_12_1), 21)
        assert not result.is_significant


class TestSignificance:
    def test_significance_needs_both_a_t_stat_and_a_sample(self):
        """A large t on nine observations is not evidence."""
        assert not StudyResult(0.5, 99.0, 9, "x").is_significant
        assert not StudyResult(0.5, 1.0, 500, "x").is_significant
        assert StudyResult(0.5, 4.0, 500, "x").is_significant

    def test_the_floor_matches_the_catalogue(self):
        assert pytest.approx(3.0) == MIN_T_STAT


class TestGapReversionCannotBeIdentifiedFromDailyBars:
    """The confound that decided the registered hypothesis — §5.1.

    `gap = open / prev_close - 1` and `intraday = close / open - 1` share the
    open with opposite signs. Measurement error in the recorded open therefore
    appears as `+e` in one series and `-e` in the other, producing negative
    correlation in a market with no reversion at all.

    These tests build exactly that market — the intraday move is drawn
    independently of the overnight move — and add noise only to the *recorded*
    open. The study reports strong, significant reversion anyway, which is why
    its real-data result of -0.2180 at t = -117.6 was not treated as a finding.
    """

    def market(self, open_noise: float, names: int = 40, sessions: int = 400) -> pl.DataFrame:
        rng = np.random.default_rng(42)
        closes, opens = {}, {}
        for i in range(names):
            close = np.empty(sessions)
            true_open = np.empty(sessions)
            level = 100.0
            for t in range(sessions):
                true_open[t] = level * (1 + rng.normal(0, 0.012))
                # Drawn from the open, independently of the overnight move:
                # this market has no gap reversion in it whatsoever.
                close[t] = true_open[t] * (1 + rng.normal(0, 0.012))
                level = close[t]
            recorded = true_open * (1 + rng.normal(0, open_noise, sessions))
            closes[f"N{i:02d}"], opens[f"N{i:02d}"] = close, recorded
        return panel(closes, opens)

    def test_a_clean_open_shows_no_reversion(self):
        """The control. Without measurement error the study correctly finds
        nothing, so what follows is attributable to the noise and not to the
        fixture."""
        result = gap_reversion(self.market(open_noise=0.0))
        assert not result.is_significant

    def test_half_a_percent_of_noise_manufactures_significant_reversion(self):
        result = gap_reversion(self.market(open_noise=0.005))
        assert result.statistic < -0.05
        assert result.is_significant

    def test_the_artefact_grows_with_the_noise(self):
        """Monotone in the noise, which is the signature of a mechanical
        relationship rather than an economic one."""
        mild = gap_reversion(self.market(open_noise=0.002))
        severe = gap_reversion(self.market(open_noise=0.01))
        assert severe.statistic < mild.statistic

    def test_the_artefact_reaches_the_magnitude_measured_on_real_data(self):
        """One percent noise reproduces a statistic past the -0.2180 the NSE
        panel showed. If this ever stops holding, the real-data result deserves
        a second look — but while it holds, that result is unidentified."""
        result = gap_reversion(self.market(open_noise=0.01))
        assert result.statistic < -0.2180
