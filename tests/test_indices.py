"""NSE index closes — MASTER_PLAN §6, §13.4.

The risk model's market factor was a column of ones, so every beta in the
research was measured against an equal-weighted average of two thousand mostly
small names rather than against a market. These tests cover the feed and store
that fix that, and they lean hardest on the failures that would produce a
*plausible wrong number* rather than an error: a web page parsed as data, a
holiday's dashes cast to zero, a session read before it was published.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from data.feeds.nse_indices import (
    BENCHMARK,
    IndexFormatError,
    nse_index_url,
    parse_index_close,
)
from data.store.bars import NoDataError
from data.store.indices import IndexStore

HEADER = (
    "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
    "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield"
)


def csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join([HEADER, *rows]) + "\n").encode()


def row(name: str, close: str = "22000.50", open_: str = "21900.00") -> str:
    return f"{name},02-01-2024,{open_},22100.00,21850.00,{close},100.5,0.46,1000,5000,22.1,4.2,1.2"


class TestUrl:
    def test_uses_day_month_year(self) -> None:
        """The index archive is DDMMYYYY where the bhavcopy is YYYYMMDD.

        Naming it in a test because the two feeds sit beside each other and the
        wrong format 404s rather than returning the wrong day — the better
        failure, but a surprising one.
        """
        assert nse_index_url(date(2024, 1, 2)).endswith("ind_close_all_02012024.csv")


class TestParse:
    def test_reads_a_session(self) -> None:
        session = parse_index_close(csv_bytes([row(BENCHMARK)]), date(2024, 1, 2))
        assert session.session_date == date(2024, 1, 2)
        assert session.rows.height == 1
        assert session.rows["index_name"][0] == BENCHMARK
        assert session.rows["close"][0] == pytest.approx(22000.50)

    def test_refuses_a_web_page(self) -> None:
        """NSE answers a missing file with its homepage and HTTP 200.

        Parsed as CSV that becomes a frame of nulls, and a null close is a
        session with no market in it. Refused instead.
        """
        with pytest.raises(IndexFormatError, match="web page"):
            parse_index_close(b"<!DOCTYPE html>\n<html><body>NSE</body></html>", date(2024, 1, 2))

    def test_refuses_an_unexpected_layout(self) -> None:
        with pytest.raises(IndexFormatError, match="missing columns"):
            parse_index_close(b"Index,Close\nNifty 50,22000\n", date(2024, 1, 2))

    def test_drops_indices_that_did_not_trade(self) -> None:
        """A holiday or a suspended index prints a dash, not a number.

        Cast strictly that fails the whole session; cast to zero it becomes an
        index that lost all its value. Dropped.
        """
        payload = csv_bytes([row(BENCHMARK), row("Nifty Dormant", close="-", open_="-")])
        session = parse_index_close(payload, date(2024, 1, 2))
        assert session.rows["index_name"].to_list() == [BENCHMARK]

    def test_drops_zero_closes(self) -> None:
        payload = csv_bytes([row(BENCHMARK), row("Nifty Zero", close="0.00")])
        assert parse_index_close(payload, date(2024, 1, 2)).rows.height == 1

    def test_timestamps_come_from_the_argument_not_the_file(self) -> None:
        """A mislabelled download must not write itself into the file's date.

        The row says 02-01-2024; the caller says it fetched the 3rd. The
        caller wins, because the caller is the one that knows which URL it
        asked for.
        """
        session = parse_index_close(csv_bytes([row(BENCHMARK)]), date(2024, 1, 3))
        assert session.rows["event_time"][0].date() == date(2024, 1, 3)

    def test_receive_time_is_not_before_event_time(self) -> None:
        session = parse_index_close(csv_bytes([row(BENCHMARK)]), date(2024, 1, 2))
        assert session.rows["receive_time"][0] >= session.rows["event_time"][0]


class TestStore:
    def test_round_trips(self, tmp_path) -> None:
        store = IndexStore(tmp_path)
        session = parse_index_close(csv_bytes([row(BENCHMARK)]), date(2024, 1, 2))
        assert store.write_session(date(2024, 1, 2), session.rows) == 1

        series = store.series(BENCHMARK, as_of=as_decision_time(datetime(2024, 6, 1, tzinfo=UTC)))
        assert series.height == 1
        assert series["close"][0] == pytest.approx(22000.50)

    def test_rejects_a_frame_missing_columns(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            IndexStore(tmp_path).write_session(date(2024, 1, 2), pl.DataFrame({"close": [1.0]}))

    def test_a_decision_cannot_see_a_later_session(self, tmp_path) -> None:
        """The whole point of the store: `receive_time <= as_of`.

        Without it a backtest standing on the 2nd would regress against a
        market that had not been published yet.
        """
        store = IndexStore(tmp_path)
        for day in (date(2024, 1, 2), date(2024, 1, 3)):
            store.write_session(day, parse_index_close(csv_bytes([row(BENCHMARK)]), day).rows)

        early = store.series(
            BENCHMARK, as_of=as_decision_time(datetime(2024, 1, 2, 13, tzinfo=UTC))
        )
        assert early.height == 1
        assert early["event_time"][0].date() == date(2024, 1, 2)

    def test_selects_one_index_not_all_of_them(self, tmp_path) -> None:
        store = IndexStore(tmp_path)
        payload = csv_bytes([row(BENCHMARK), row("Nifty Bank", close="47000.00")])
        store.write_session(date(2024, 1, 2), parse_index_close(payload, date(2024, 1, 2)).rows)

        as_of = as_decision_time(datetime(2024, 6, 1, tzinfo=UTC))
        assert store.series(BENCHMARK, as_of=as_of).height == 1
        assert store.names(as_of=as_of) == ["Nifty 50", "Nifty Bank"]

    def test_start_bounds_the_read(self, tmp_path) -> None:
        store = IndexStore(tmp_path)
        for i in range(5):
            day = date(2024, 1, 2) + timedelta(days=i)
            store.write_session(day, parse_index_close(csv_bytes([row(BENCHMARK)]), day).rows)

        as_of = as_decision_time(datetime(2024, 6, 1, tzinfo=UTC))
        assert store.series(BENCHMARK, as_of=as_of, start=date(2024, 1, 5)).height == 2

    def test_an_empty_lake_says_so(self, tmp_path) -> None:
        """NoDataError, not an empty frame.

        An empty frame flows downstream and silently becomes "the market did
        nothing", which is a different claim from "no market was ingested".
        """
        with pytest.raises(NoDataError):
            IndexStore(tmp_path).series(
                BENCHMARK, as_of=as_decision_time(datetime(2024, 6, 1, tzinfo=UTC))
            )

    def test_a_session_present_but_unobservable_is_empty_not_missing(self, tmp_path) -> None:
        """Held but not yet visible is a different state from never ingested."""
        store = IndexStore(tmp_path)
        store.write_session(
            date(2024, 1, 2), parse_index_close(csv_bytes([row(BENCHMARK)]), date(2024, 1, 2)).rows
        )
        early = store.series(BENCHMARK, as_of=as_decision_time(datetime(2023, 1, 1, tzinfo=UTC)))
        assert early.is_empty()

    def test_rewriting_a_session_replaces_it(self, tmp_path) -> None:
        """A corrected file needs no merge — MASTER_PLAN §13.4."""
        store = IndexStore(tmp_path)
        day = date(2024, 1, 2)
        store.write_session(day, parse_index_close(csv_bytes([row(BENCHMARK)]), day).rows)
        store.write_session(
            day, parse_index_close(csv_bytes([row(BENCHMARK, close="22500.00")]), day).rows
        )
        series = store.series(BENCHMARK, as_of=as_decision_time(datetime(2024, 6, 1, tzinfo=UTC)))
        assert series.height == 1
        assert series["close"][0] == pytest.approx(22500.00)


class TestBenchmarkSelection:
    """165 indices are ingested per session; all of them must be reachable.

    Storing them and exposing only one made the other 164 decoration. A
    midcap book measured against NIFTY 50 is being labelled by the wrong
    market, and NSE publishes the right one in the same file.
    """

    def test_the_risk_endpoint_offers_every_held_index(self, tmp_path, monkeypatch) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import apps.api.analytics as analytics_module
        from apps.api.research import build_research_router

        store = IndexStore(tmp_path)
        payload = csv_bytes(
            [row(BENCHMARK), row("Nifty Bank", close="47000.00"), row("India VIX", close="13.20")]
        )
        store.write_session(date(2024, 1, 2), parse_index_close(payload, date(2024, 1, 2)).rows)

        monkeypatch.setattr(
            analytics_module,
            "benchmark_names",
            lambda: store.names(as_of=as_decision_time(datetime(2024, 6, 1, tzinfo=UTC))),
        )

        app = FastAPI()
        app.include_router(build_research_router())
        names = TestClient(app).get("/benchmarks").json()
        assert set(names) == {"India VIX", "Nifty 50", "Nifty Bank"}

    def test_an_empty_lake_offers_nothing_rather_than_failing(self, tmp_path, monkeypatch) -> None:
        """A console with no indices ingested must still render — it falls back
        to the equal-weight proxy and says so."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import apps.api.analytics as analytics_module
        from apps.api.research import build_research_router

        monkeypatch.setattr(analytics_module, "benchmark_names", list)
        app = FastAPI()
        app.include_router(build_research_router())
        assert TestClient(app).get("/benchmarks").json() == []
