"""Fundamentals joined to the panel, and the factors built on them — §3.3, §6.

The join is the whole difficulty and it is a point-in-time join. Matching a
March quarter to March reads a filing published in July from a decision made in
April: an enormous distortion that looks like skill rather than like a bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC
from quant.research.fundamentals import attach_fundamentals


def panel(days: int = 12, symbol: str = "AAA", isin: str = "INE000A01011") -> pl.DataFrame:
    start = datetime(2026, 3, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "instrument_id": [f"NSE:{isin}"] * days,
            "symbol": [symbol] * days,
            "event_time": [start + timedelta(days=d) for d in range(days)],
            "close": [100.0 + d for d in range(days)],
        }
    )


def store_with(tmp_path, filings: list[dict]):
    from data.store.fundamentals import FundamentalStore

    if filings:
        FundamentalStore(tmp_path).write(pl.DataFrame(filings))
    return tmp_path


def filing(isin="INE000A01011", period_end="2026-02-28", published="2026-03-05", **over):
    row = {
        "isin": isin,
        "symbol": "AAA",
        "company": "AAA Ltd",
        "period_start": datetime.fromisoformat("2025-12-01").date(),
        "period_end": datetime.fromisoformat(period_end).date(),
        "receive_time": datetime.fromisoformat(published).replace(tzinfo=UTC),
        "consolidated": True,
        "audited": False,
        "xbrl_url": "x.xml",
        "revenue": 1000.0,
        "other_income": None,
        "total_income": None,
        "profit_before_tax": None,
        "net_profit": 150.0,
        "eps_basic": 5.0,
        "eps_diluted": 5.0,
    }
    return {**row, **over}


class TestPointInTimeJoin:
    def test_a_filing_is_invisible_before_it_was_published(self, tmp_path) -> None:
        """The distortion this module exists to prevent: a quarter that ended
        in February is not readable until the day it was disseminated."""
        lake = store_with(tmp_path, [filing(published="2026-03-05")])
        joined = attach_fundamentals(panel(), lake).sort("event_time")
        before = joined.filter(pl.col("event_time").dt.date() < datetime(2026, 3, 5).date())
        assert before["net_profit"].null_count() == before.height

    def test_it_becomes_visible_on_the_day_it_was_published(self, tmp_path) -> None:
        """Available to a decision taken on that close, which fills the next
        session -- trading on it, not peeking at it."""
        lake = store_with(tmp_path, [filing(published="2026-03-05")])
        joined = attach_fundamentals(panel(), lake).sort("event_time")
        on_day = joined.filter(pl.col("event_time").dt.date() == datetime(2026, 3, 5).date())
        assert on_day["net_profit"][0] == 150.0

    def test_the_newest_readable_filing_wins(self, tmp_path) -> None:
        lake = store_with(
            tmp_path,
            [
                filing(period_end="2025-11-30", published="2026-03-02", net_profit=100.0),
                filing(period_end="2026-02-28", published="2026-03-08", net_profit=150.0),
            ],
        )
        joined = attach_fundamentals(panel(), lake).sort("event_time")
        last = joined.filter(pl.col("event_time").dt.date() == datetime(2026, 3, 10).date())
        assert last["net_profit"][0] == 150.0

    def test_a_name_with_no_filing_gets_nulls_not_zeros(self, tmp_path) -> None:
        """Zero revenue would rank as the cheapest name in the market."""
        lake = store_with(tmp_path, [filing(isin="INE999Z01011")])
        joined = attach_fundamentals(panel(), lake)
        assert joined["revenue"].null_count() == joined.height

    def test_an_empty_store_returns_the_columns_anyway(self, tmp_path) -> None:
        """A factor built on this scores nothing rather than failing."""
        joined = attach_fundamentals(panel(), tmp_path)
        assert "revenue" in joined.columns
        assert joined["revenue"].null_count() == joined.height

    def test_a_long_stale_filing_is_flagged(self, tmp_path) -> None:
        """A company that stopped reporting keeps its last figures, and the
        staleness is a fact about the company rather than a reason to vanish."""
        lake = store_with(tmp_path, [filing(period_end="2024-06-30", published="2024-08-01")])
        joined = attach_fundamentals(panel(), lake)
        assert bool(joined["filing_stale"][0]) is True
        assert joined["filing_age_days"][0] > 250


class TestFactorGuards:
    def test_the_factors_declare_they_need_fundamentals(self) -> None:
        from quant.research.factors import Factor

        assert Factor.NET_MARGIN.needs_fundamentals
        assert Factor.EARNINGS_YIELD.needs_fundamentals
        assert not Factor.MOMENTUM_12_1.needs_fundamentals

    def test_a_stale_filing_scores_nothing(self, tmp_path) -> None:
        """An earnings yield on figures nobody has confirmed in a year is a
        number about the past wearing the current price."""
        from quant.research.expressions import signal_expression
        from quant.research.factors import Factor

        frame = pl.DataFrame(
            {
                "close": [100.0, 100.0],
                "eps_basic": [5.0, 5.0],
                "revenue": [1000.0, 1000.0],
                "net_profit": [150.0, 150.0],
                "filing_stale": [True, False],
            }
        )
        scored = frame.with_columns(signal_expression(Factor.EARNINGS_YIELD).alias("s"))
        assert scored["s"][0] is None
        assert scored["s"][1] == pytest.approx(0.05)

    def test_non_positive_revenue_scores_nothing(self, tmp_path) -> None:
        """Real filings carry negative revenue -- an NBFC booking a fair-value
        reversal -- and dividing by it produces a sign that means nothing."""
        from quant.research.expressions import signal_expression
        from quant.research.factors import Factor

        frame = pl.DataFrame(
            {
                "close": [100.0, 100.0],
                "eps_basic": [5.0, 5.0],
                "revenue": [-500.0, 1000.0],
                "net_profit": [150.0, 150.0],
                "filing_stale": [False, False],
            }
        )
        scored = frame.with_columns(signal_expression(Factor.NET_MARGIN).alias("s"))
        assert scored["s"][0] is None
        assert scored["s"][1] == pytest.approx(0.15)
