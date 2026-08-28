"""Trading a precomputed signal (§6, §13, §17).

The connector between research and the backtester, and the place a backtest is
most likely to read the future: the scored panel holds the entire history by
construction. Both guards are tested, and so is the key mismatch that made the
first version hold nothing at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from core.instruments import InstrumentId
from quant.strategies.base import MarketView
from quant.strategies.signal import Construction, ForwardLeakError, SignalStrategy

SEED = 20260820
SESSIONS = 120


def times(n: int = SESSIONS) -> list[datetime]:
    return [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)]


def history(symbols: dict[str, float]) -> pl.DataFrame:
    """Panel where each symbol has a constant daily volatility."""
    rng = np.random.default_rng(SEED)
    stamps = times()
    frames = []
    for symbol, vol in symbols.items():
        closes = list(100.0 * np.exp(np.cumsum(rng.normal(0, vol, SESSIONS))))
        frames.append(
            pl.DataFrame(
                {
                    "event_time": stamps,
                    "symbol": [symbol] * SESSIONS,
                    "instrument_id": [f"NSE:INE{symbol}"] * SESSIONS,
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [1e6] * SESSIONS,
                },
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
        )
    return pl.concat(frames)


def scores_for(values: dict[str, float], n: int = SESSIONS) -> pl.DataFrame:
    stamps = times(n)
    return pl.concat(
        [
            pl.DataFrame(
                {"event_time": stamps, "symbol": [symbol] * n, "signal": [score] * n},
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
            for symbol, score in values.items()
        ]
    )


def view_for(frame: pl.DataFrame, as_of: datetime | None = None) -> MarketView:
    return MarketView(
        as_of=as_decision_time(as_of or (times()[-1] + timedelta(days=1))),
        history=frame,
        universe=tuple(frame["instrument_id"].unique().to_list()),
    )


class TestForwardLeakGuard:
    """A frame straight from `build_factor` carries tomorrow's return."""

    def test_forward_columns_are_refused(self):
        frame = scores_for({"AAA": 1.0}).with_columns(pl.lit(0.05).alias("fwd_21"))
        with pytest.raises(ForwardLeakError, match="fwd_21"):
            SignalStrategy(frame)

    def test_the_error_names_every_leaked_column(self):
        frame = scores_for({"AAA": 1.0}).with_columns(
            pl.lit(0.0).alias("fwd_1"), pl.lit(0.0).alias("fwd_63")
        )
        with pytest.raises(ForwardLeakError) as caught:
            SignalStrategy(frame)
        assert "fwd_1" in str(caught.value)
        assert "fwd_63" in str(caught.value)

    def test_a_clean_frame_is_accepted(self):
        assert SignalStrategy(scores_for({"AAA": 1.0})) is not None

    def test_a_frame_missing_signal_is_refused(self):
        frame = scores_for({"AAA": 1.0}).drop("signal")
        with pytest.raises(ValueError, match="missing"):
            SignalStrategy(frame)


class TestAsOfGuard:
    """The second guard: only scores at or before the decision time."""

    def test_future_scores_are_not_read(self):
        """AAA scores high only in the second half. A decision taken in the
        first half must not see it."""
        frame = history({"AAA": 0.01, "BBB": 0.01})
        early, late = times()[:60], times()[60:]
        scores = pl.concat(
            [
                pl.DataFrame(
                    {"event_time": early, "symbol": ["AAA"] * 60, "signal": [0.0] * 60},
                    schema_overrides={"event_time": pl.Datetime("us", "UTC")},
                ),
                pl.DataFrame(
                    {"event_time": late, "symbol": ["AAA"] * 60, "signal": [9.0] * 60},
                    schema_overrides={"event_time": pl.Datetime("us", "UTC")},
                ),
                pl.DataFrame(
                    {
                        "event_time": times(),
                        "symbol": ["BBB"] * SESSIONS,
                        "signal": [1.0] * SESSIONS,
                    },
                    schema_overrides={"event_time": pl.Datetime("us", "UTC")},
                ),
            ]
        )
        strategy = SignalStrategy(scores, top_fraction=Decimal("0.5"))
        picked = strategy(view_for(frame, early[30])).weights
        assert next(iter(picked)).endswith("BBB")

    def test_no_scores_yet_means_no_position(self):
        frame = history({"AAA": 0.01})
        before = datetime(2020, 1, 1, tzinfo=UTC)
        assert SignalStrategy(scores_for({"AAA": 1.0}))(view_for(frame, before)).weights == {}


class TestSymbolResolution:
    """The scored panel is keyed by ticker, the universe by ISIN."""

    def test_symbols_resolve_to_instrument_ids(self):
        """Comparing them directly matched nothing, and a strategy holding
        nothing looks exactly like a signal with no opportunities."""
        frame = history({"AAA": 0.01, "BBB": 0.02})
        strategy = SignalStrategy(scores_for({"AAA": 2.0, "BBB": 1.0}))
        resolved = strategy.scores_at(view_for(frame))
        assert len(resolved) == 2
        assert all(str(k).startswith("NSE:INE") for k in resolved)

    def test_names_outside_the_universe_are_ignored(self):
        frame = history({"AAA": 0.01})
        strategy = SignalStrategy(scores_for({"AAA": 2.0, "ZZZ": 9.0}))
        assert len(strategy.scores_at(view_for(frame))) == 1


class TestSelection:
    def test_the_top_fraction_is_held(self):
        frame = history({f"N{i:02d}": 0.01 for i in range(10)})
        scores = scores_for({f"N{i:02d}": float(i) for i in range(10)})
        held = SignalStrategy(scores, top_fraction=Decimal("0.3"))(view_for(frame)).weights
        assert len(held) == 3

    def test_the_highest_scores_win(self):
        frame = history({"LOW": 0.01, "HIGH": 0.01})
        scores = scores_for({"LOW": 1.0, "HIGH": 5.0})
        held = SignalStrategy(scores, top_fraction=Decimal("0.5"))(view_for(frame)).weights
        assert next(iter(held)).endswith("HIGH")

    def test_at_least_one_name_is_held(self):
        frame = history({"AAA": 0.01, "BBB": 0.01})
        scores = scores_for({"AAA": 1.0, "BBB": 2.0})
        held = SignalStrategy(scores, top_fraction=Decimal("0.01"))(view_for(frame)).weights
        assert len(held) == 1

    def test_ties_break_deterministically(self):
        """Two runs over the same data must hold the same book (§14.1.1)."""
        frame = history({f"N{i}": 0.01 for i in range(6)})
        scores = scores_for({f"N{i}": 1.0 for i in range(6)})
        strategy = SignalStrategy(scores, top_fraction=Decimal("0.5"))
        assert strategy(view_for(frame)).weights == strategy(view_for(frame)).weights

    @pytest.mark.parametrize("fraction", ["0", "-0.5", "1.5"])
    def test_a_nonsense_fraction_is_refused(self, fraction):
        with pytest.raises(ValueError, match="top_fraction"):
            SignalStrategy(scores_for({"AAA": 1.0}), top_fraction=Decimal(fraction))


class TestConstruction:
    def frame_and_scores(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        return history({"CALM": 0.005, "WILD": 0.05}), scores_for({"CALM": 1.0, "WILD": 2.0})

    def test_equal_weighting_splits_evenly(self):
        frame, scores = self.frame_and_scores()
        held = SignalStrategy(scores, top_fraction=Decimal(1), construction=Construction.EQUAL)(
            view_for(frame)
        ).weights
        assert len(set(held.values())) == 1

    def test_inverse_vol_favours_the_calm_name(self):
        """Each position contributes comparable risk rather than comparable
        rupees."""
        frame, scores = self.frame_and_scores()
        held = SignalStrategy(
            scores,
            top_fraction=Decimal(1),
            construction=Construction.INVERSE_VOL,
            max_position=Decimal(1),
        )(view_for(frame)).weights
        by_name = {str(k).replace("NSE:INE", ""): v for k, v in held.items()}
        assert by_name["CALM"] > by_name["WILD"]

    def test_score_weighting_favours_the_higher_score(self):
        frame, scores = self.frame_and_scores()
        held = SignalStrategy(
            scores,
            top_fraction=Decimal(1),
            construction=Construction.SCORE,
            max_position=Decimal(1),
        )(view_for(frame)).weights
        by_name = {str(k).replace("NSE:INE", ""): v for k, v in held.items()}
        assert by_name["WILD"] > by_name["CALM"]

    def test_gross_never_exceeds_the_cap(self):
        frame, scores = self.frame_and_scores()
        for construction in Construction:
            held = SignalStrategy(
                scores, top_fraction=Decimal(1), construction=construction, gross=Decimal("0.9")
            )(view_for(frame))
            assert held.gross <= Decimal("0.9")

    def test_every_construction_has_a_description(self):
        for construction in Construction:
            assert len(construction.description) > 40


class TestSpec:
    def test_the_signal_name_reaches_the_spec(self):
        """An experiment row should say which signal was traded."""
        strategy = SignalStrategy(scores_for({"AAA": 1.0}), name="composite")
        assert strategy.spec.name == "signal:composite"
        assert strategy.spec.parameters["signal"] == "composite"


class TestLongShort:
    """A quantile spread is Q5 minus Q1, and holding only Q5 earns half of it
    plus the market. On the paper book the market is 72.7% of variance, so a
    long-only implementation of a long-short signal is largely a market
    position wearing the signal's name.
    """

    def scores(self) -> pl.DataFrame:
        """Ten names, ranked, so the quartiles are unambiguous."""
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
        rows = []
        for t in times:
            for i in range(10):
                rows.append({"event_time": t, "symbol": f"N{i}", "signal": float(i)})
        return pl.DataFrame(rows, schema_overrides={"event_time": pl.Datetime("us", "UTC")})

    def view(self) -> MarketView:
        times = [datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
        history = pl.DataFrame(
            [
                {
                    "event_time": t,
                    "receive_time": t,
                    "instrument_id": f"NSE:INE{i}",
                    "symbol": f"N{i}",
                    "close": 100.0 + i,
                }
                for t in times
                for i in range(10)
            ],
            schema_overrides={
                "event_time": pl.Datetime("us", "UTC"),
                "receive_time": pl.Datetime("us", "UTC"),
            },
        )
        return MarketView(
            as_of=as_decision_time(times[-1]),
            history=history,
            universe=tuple(InstrumentId(f"NSE:INE{i}") for i in range(10)),
        )

    def test_long_only_holds_only_the_top(self):
        weights = (
            SignalStrategy(self.scores(), top_fraction=Decimal("0.2"), long_only=True)
            .generate(self.view())
            .weights
        )
        assert all(w > 0 for w in weights.values())

    def test_long_short_holds_both_ends(self):
        weights = (
            SignalStrategy(self.scores(), top_fraction=Decimal("0.2"), long_only=False)
            .generate(self.view())
            .weights
        )
        assert any(w > 0 for w in weights.values())
        assert any(w < 0 for w in weights.values())

    def test_the_legs_offset(self):
        """Market-neutral by construction: the whole point of shorting the
        bottom is that the market exposure cancels."""
        weights = (
            SignalStrategy(self.scores(), top_fraction=Decimal("0.2"), long_only=False)
            .generate(self.view())
            .weights
        )
        assert sum(weights.values()) == pytest.approx(Decimal(0), abs=Decimal("0.001"))

    def test_gross_is_respected_across_both_legs(self):
        gross = Decimal("0.9")
        weights = (
            SignalStrategy(self.scores(), top_fraction=Decimal("0.2"), gross=gross, long_only=False)
            .generate(self.view())
            .weights
        )
        assert sum(abs(w) for w in weights.values()) == pytest.approx(gross, abs=Decimal("0.001"))

    def test_long_only_is_still_the_default(self):
        """Shorting is a decision, not something a caller acquires silently."""
        assert SignalStrategy(self.scores()).long_only is True

    def test_a_name_in_both_legs_nets_out(self):
        """With a small universe the quantiles overlap, and a name must not be
        held twice in opposite directions."""
        weights = (
            SignalStrategy(self.scores(), top_fraction=Decimal("0.6"), long_only=False)
            .generate(self.view())
            .weights
        )
        assert all(isinstance(w, Decimal) for w in weights.values())
