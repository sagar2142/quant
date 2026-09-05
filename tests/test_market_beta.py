"""Market beta against a real index — MASTER_PLAN §6, §8.

The risk model gave every name a market exposure of exactly one, which makes
the market factor's return an equal-weighted average of everything that traded.
These tests check that a real index produces a real exposure, and — the part
that actually matters — that a high-beta book and a low-beta book no longer
look identical to the market factor.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.research.factors import Factor
from quant.research.market import BETA_CLIP, MIN_BETA_SESSIONS, index_returns, market_betas
from quant.research.riskmodel import MARKET, build_risk_model, decompose

SESSIONS = 400


def times(length: int = SESSIONS) -> list[datetime]:
    return [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]


def index_frame(returns: np.ndarray) -> pl.DataFrame:
    closes = 20000.0 * np.cumprod(1.0 + np.concatenate([[0.0], returns]))
    return pl.DataFrame({"event_time": times(len(closes)), "close": closes})


def panel(series: dict[str, np.ndarray]) -> pl.DataFrame:
    length = len(next(iter(series.values())))
    stamps = times(length)
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "event_time": stamps,
                    "symbol": [name] * length,
                    "instrument_id": [f"NSE:INE{name}"] * length,
                    "open": [closes[max(0, i - 1)] for i in range(length)],
                    "high": [c * 1.01 for c in closes],
                    "low": [c * 0.99 for c in closes],
                    "close": list(closes),
                    "volume": [1e8] * length,
                    "trades": [1000] * length,
                }
            )
            for name, closes in series.items()
        ]
    )


def levered(market_returns: np.ndarray, beta: float, start: float = 100.0) -> np.ndarray:
    """A price series whose returns are exactly `beta` times the market's."""
    return start * np.cumprod(1.0 + np.concatenate([[0.0], beta * market_returns]))


@pytest.fixture
def market_returns() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.0004, 0.011, SESSIONS - 1)


class TestIndexReturns:
    def test_drops_the_first_session(self) -> None:
        """It has no prior close, so it has no return. Dropped, not zeroed:
        a fabricated flat day is a real observation to a covariance."""
        frame = index_returns(index_frame(np.array([0.01, -0.02, 0.005])))
        assert frame.height == 3
        assert frame["market_return"][0] == pytest.approx(0.01)

    def test_empty_in_empty_out_with_the_right_schema(self) -> None:
        frame = index_returns(pl.DataFrame())
        assert frame.is_empty()
        assert frame.columns == ["event_time", "market_return"]


class TestBeta:
    def test_recovers_a_known_beta(self, market_returns: np.ndarray) -> None:
        """A series that is exactly 1.6x the market must measure 1.6.

        Not a tautology worth skipping: the rolling covariance is written out
        by hand because Polars has no rolling `cov`, and an algebra slip there
        would produce a number that is wrong but entirely believable.
        """
        frame = panel({"LEV": levered(market_returns, 1.6)})
        betas = market_betas(frame, index_frame(market_returns))
        assert betas["beta"].tail(1)[0] == pytest.approx(1.6, abs=1e-6)

    def test_a_defensive_name_measures_below_one(self, market_returns: np.ndarray) -> None:
        frame = panel({"DEF": levered(market_returns, 0.4)})
        betas = market_betas(frame, index_frame(market_returns))
        assert betas["beta"].tail(1)[0] == pytest.approx(0.4, abs=1e-6)

    def test_an_inverse_name_measures_negative(self, market_returns: np.ndarray) -> None:
        frame = panel({"INV": levered(market_returns, -0.8)})
        betas = market_betas(frame, index_frame(market_returns))
        assert betas["beta"].tail(1)[0] == pytest.approx(-0.8, abs=1e-6)

    def test_beta_is_clipped(self, market_returns: np.ndarray) -> None:
        """A beta of 9 is not a stock nine times the market; it is a stock
        whose returns are dominated by something else. Left unclipped it drags
        a whole session's regression with it."""
        frame = panel({"WILD": levered(market_returns, 9.0)})
        betas = market_betas(frame, index_frame(market_returns))
        assert betas["beta"].max() == pytest.approx(BETA_CLIP)

    def test_a_short_history_gets_no_beta(self, market_returns: np.ndarray) -> None:
        """A newly listed name has no measurable market sensitivity, and is
        left without one rather than assigned a fabricated exposure of 1."""
        short = MIN_BETA_SESSIONS // 2
        frame = panel({"NEW": levered(market_returns[:short], 1.2)})
        assert market_betas(frame, index_frame(market_returns)).is_empty()

    def test_beta_only_uses_the_past(self, market_returns: np.ndarray) -> None:
        """The beta at session t must not change when session t+1 arrives.

        This is the look-ahead check. A rolling window that centred itself, or
        a group-wise mean computed over the whole series, would pass every
        other test here and quietly leak the future into a backtest.
        """
        full = panel({"LEV": levered(market_returns, 1.6)})
        truncated = full.head(full.height - 50)

        cut = truncated["event_time"].max()
        a = market_betas(full, index_frame(market_returns)).filter(pl.col("event_time") <= cut)
        b = market_betas(truncated, index_frame(market_returns[: len(market_returns) - 50]))
        assert a["beta"].to_list() == pytest.approx(b["beta"].to_list())

    def test_a_session_the_index_does_not_cover_yields_no_betas(
        self, market_returns: np.ndarray
    ) -> None:
        """A partly backfilled index must visibly shrink the model's history,
        not silently estimate some names against a market and some against
        nothing."""
        frame = panel({"LEV": levered(market_returns, 1.2)})
        partial = index_frame(market_returns).head(100)
        betas = market_betas(frame, partial)
        assert betas["event_time"].max() <= partial["event_time"].max()

    def test_empty_inputs_return_the_right_schema(self) -> None:
        empty = market_betas(pl.DataFrame(), pl.DataFrame())
        assert empty.is_empty()
        assert empty.columns == ["event_time", "instrument_id", "beta"]


class TestRiskModelWiring:
    """The reason all of the above exists."""

    def build(self, betas: pl.DataFrame | None, market_returns: np.ndarray):
        rng = np.random.default_rng(11)
        names = {}
        for i in range(30):
            beta = 0.3 + 0.06 * i
            noise = rng.normal(0.0, 0.004, SESSIONS - 1)
            path = np.cumprod(1.0 + np.concatenate([[0.0], beta * market_returns + noise]))
            names[f"N{i:02d}"] = 100.0 * path
        frame = panel(names)
        estimated = betas if betas is None else market_betas(frame, index_frame(market_returns))
        return frame, build_risk_model(
            frame, (Factor.MOMENTUM_1M, Factor.REVERSAL_5D), betas=estimated
        )

    def test_without_an_index_the_market_is_a_column_of_ones(
        self, market_returns: np.ndarray
    ) -> None:
        _, model = self.build(None, market_returns)
        assert model is not None
        assert model.market_source == "equal_weight"
        assert all(np.isclose(v[0], 1.0) for v in model.exposures.values())

    def test_with_an_index_the_market_column_is_beta(self, market_returns: np.ndarray) -> None:
        _, model = self.build(pl.DataFrame({"x": [1]}), market_returns)
        assert model is not None
        assert model.market_source == "index"
        loaded = [v[0] for v in model.exposures.values()]
        assert min(loaded) < 0.8 < max(loaded)

    def test_the_market_column_stays_first(self, market_returns: np.ndarray) -> None:
        _, model = self.build(pl.DataFrame({"x": [1]}), market_returns)
        assert model is not None
        assert model.factor_names[0] == MARKET

    def test_a_high_beta_book_carries_more_market_risk(self, market_returns: np.ndarray) -> None:
        """The finding this whole module exists for.

        With a column of ones these two books have *identical* market exposure
        by construction, because every name loads 1.0 regardless of how it
        moves. With real betas they do not, and the difference is the thing a
        market-neutral book is trying to control.
        """
        _, model = self.build(pl.DataFrame({"x": [1]}), market_returns)
        assert model is not None

        loadings = sorted(model.exposures.items(), key=lambda kv: kv[1][0])
        low = {k: 1.0 / 5 for k, _ in loadings[:5]}
        high = {k: 1.0 / 5 for k, _ in loadings[-5:]}

        assert (
            decompose(model, high).contributions[MARKET]
            > (decompose(model, low).contributions[MARKET])
        )

    def test_the_format_says_which_market_it_used(self, market_returns: np.ndarray) -> None:
        """A beta is only as meaningful as the market it was measured against,
        so the reader is told which one it was."""
        _, proxy = self.build(None, market_returns)
        _, real = self.build(pl.DataFrame({"x": [1]}), market_returns)
        assert proxy is not None and real is not None
        assert "equal-weight proxy" in proxy.format()
        assert "benchmark index" in real.format()
