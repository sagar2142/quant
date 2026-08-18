"""Cross-sectional panel store (§M2)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from data.store.bars import NoDataError
from data.store.panel import PANEL_SCHEMA, PanelStore

SESSION = date(2024, 3, 15)
CLOSE = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
RECEIVE = CLOSE + timedelta(hours=2, minutes=30)


def rows(ids: list[str], receive: datetime = RECEIVE) -> pl.DataFrame:
    n = len(ids)
    return pl.DataFrame(
        {
            "event_time": [CLOSE] * n,
            "receive_time": [receive] * n,
            "instrument_id": ids,
            "symbol": [i.split(":")[-1] for i in ids],
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000.0] * n,
            "trades": [10] * n,
        },
        schema_overrides=PANEL_SCHEMA,
    )


@pytest.fixture
def panel(tmp_path) -> PanelStore:
    return PanelStore(tmp_path, venue="NSE")


class TestWriteRead:
    def test_roundtrip(self, panel):
        panel.write_session(SESSION, rows(["NSE:A", "NSE:B"]))
        got = panel.view(as_of=as_decision_time(RECEIVE + timedelta(hours=1)))
        assert got.height == 2

    def test_rewrite_replaces_whole_session(self, panel):
        panel.write_session(SESSION, rows(["NSE:A", "NSE:B"]))
        panel.write_session(SESSION, rows(["NSE:A"]))
        got = panel.view(as_of=as_decision_time(RECEIVE + timedelta(hours=1)))
        assert got.height == 1

    def test_sessions_listed_ascending(self, panel):
        for day in (date(2024, 3, 15), date(2024, 1, 2), date(2023, 12, 1)):
            panel.write_session(day, rows(["NSE:A"]))
        assert panel.sessions() == [date(2023, 12, 1), date(2024, 1, 2), date(2024, 3, 15)]

    def test_no_data_raises(self, panel):
        with pytest.raises(NoDataError):
            panel.view(as_of=as_decision_time(RECEIVE))

    def test_session_view_of_absent_date_is_empty(self, panel):
        panel.write_session(SESSION, rows(["NSE:A"]))
        got = panel.session_view(date(2024, 3, 14), as_of=as_decision_time(RECEIVE))
        assert got.is_empty()


class TestPointInTime:
    def test_unpublished_session_invisible(self, panel):
        panel.write_session(SESSION, rows(["NSE:A"]))
        before = panel.view(as_of=as_decision_time(RECEIVE - timedelta(minutes=1)))
        assert before.is_empty()

    def test_published_session_visible(self, panel):
        panel.write_session(SESSION, rows(["NSE:A"]))
        after = panel.view(as_of=as_decision_time(RECEIVE))
        assert after.height == 1

    def test_start_filter(self, panel):
        panel.write_session(date(2024, 1, 2), rows(["NSE:A"]))
        panel.write_session(SESSION, rows(["NSE:B"]))
        got = panel.view(
            as_of=as_decision_time(RECEIVE + timedelta(days=1)), start=date(2024, 2, 1)
        )
        assert got["instrument_id"].to_list() == ["NSE:B"]


class TestValidation:
    def test_missing_column_rejected(self, panel):
        with pytest.raises(ValueError, match="missing columns"):
            panel.write_session(SESSION, rows(["NSE:A"]).drop("volume"))

    def test_empty_session_rejected(self, panel):
        with pytest.raises(ValueError, match="empty"):
            panel.write_session(SESSION, rows(["NSE:A"]).head(0))

    def test_duplicate_instrument_rejected(self, panel):
        # A cross-section must carry each instrument exactly once.
        with pytest.raises(ValueError, match="duplicate"):
            panel.write_session(SESSION, rows(["NSE:A", "NSE:A"]))

    def test_causality_violation_rejected(self, panel):
        bad = rows(["NSE:A"], receive=CLOSE - timedelta(hours=1))
        with pytest.raises(ValueError, match="received before"):
            panel.write_session(SESSION, bad)
