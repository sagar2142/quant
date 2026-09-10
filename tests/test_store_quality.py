"""Quality checks for the feeds that are not bars — MASTER_PLAN §M2, §9.

A gate covering one of five inputs reports on the input least likely to be
wrong, because it is the one that has been checked longest. These ask the
questions bar checks cannot: is the feed current, does it cover what is traded,
and does it contradict itself.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from core.clock import UTC, utc_now
from data.feeds.nse_events import parse_events, resolve
from data.feeds.nse_results import parse_filings
from data.quality.checks import Severity
from data.quality.stores import (
    CALENDAR_STALE_DAYS,
    SECTOR_STALE_DAYS,
    check_events,
    check_fundamentals,
    check_sectors,
)
from data.store.events import EventStore
from data.store.fundamentals import FundamentalStore
from data.store.sectors import SectorStore


def severities(report) -> dict[str, Severity]:
    return {f.check: f.severity for f in report.findings}


class TestSectors:
    def write(self, lake: Path, observed: date, rows: list[tuple[str, str]]):
        frame = pl.DataFrame(
            {
                "isin": [r[0] for r in rows],
                "symbol": [r[0][:6] for r in rows],
                "company": [r[0] for r in rows],
                "industry": [r[1] for r in rows],
                "source": ["test"] * len(rows),
            }
        )
        SectorStore(lake).write(observed, frame)

    def test_no_classification_is_critical(self, tmp_path) -> None:
        """Every exposure screen reads this; absent is not a degraded state."""
        assert severities(check_sectors(tmp_path))["absent"] is Severity.CRITICAL

    def test_a_current_classification_is_clean(self, tmp_path) -> None:
        self.write(tmp_path, utc_now().date(), [("INE001A01011", "Banks")])
        assert check_sectors(tmp_path).is_clean

    def test_an_old_classification_warns(self, tmp_path) -> None:
        old = utc_now().date() - timedelta(days=SECTOR_STALE_DAYS + 5)
        self.write(tmp_path, old, [("INE001A01011", "Banks")])
        assert severities(check_sectors(tmp_path))["stale"] is Severity.WARN

    def test_a_blank_industry_is_critical(self, tmp_path) -> None:
        """It joins successfully and then groups every blank name as one
        sector, which is worse than not joining at all."""
        self.write(tmp_path, utc_now().date(), [("INE001A01011", "Banks"), ("INE002A01018", " ")])
        assert severities(check_sectors(tmp_path))["blank_industry"] is Severity.CRITICAL

    def test_thin_coverage_of_the_traded_universe_warns(self, tmp_path) -> None:
        self.write(tmp_path, utc_now().date(), [("INE001A01011", "Banks")])
        universe = {f"INE{n:03d}A01011" for n in range(20)}
        assert "coverage" in severities(check_sectors(tmp_path, universe))

    def test_coverage_is_not_judged_without_a_universe(self, tmp_path) -> None:
        self.write(tmp_path, utc_now().date(), [("INE001A01011", "Banks")])
        assert "coverage" not in severities(check_sectors(tmp_path))


FILING = {
    "isin": "INE764D01017",
    "symbol": "VSTTILLERS",
    "companyName": "V.S.T Tillers",
    "fromDate": "01-Oct-2024",
    "toDate": "31-Dec-2024",
    "exchdisstime": "30-Jan-2025 17:18:42",
    "consolidated": "Consolidated",
    "audited": "Un-Audited",
    "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1.xml",
}


class TestFundamentals:
    def test_nothing_held_is_a_warning_not_a_failure(self, tmp_path) -> None:
        """A research install that has not fetched results yet is incomplete,
        not broken."""
        assert severities(check_fundamentals(tmp_path))["absent"] is Severity.WARN

    def test_filings_without_numbers_are_named(self, tmp_path) -> None:
        """Dates alone make an earnings calendar; a value factor needs figures."""
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(json.dumps([FILING]).encode()))
        assert severities(check_fundamentals(tmp_path))["numbers_missing"] is Severity.WARN

    def test_a_filing_published_before_its_period_ended_is_critical(self, tmp_path) -> None:
        """Either a parsing error or a date that must not be trusted -- it
        would let a decision read a quarter that had not finished."""
        early = {**FILING, "exchdisstime": "01-Nov-2024 10:00:00"}
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(json.dumps([early]).encode()))
        found = severities(check_fundamentals(tmp_path))
        assert found["published_before_period_end"] is Severity.CRITICAL

    def test_populated_filings_pass(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        frame = parse_filings(json.dumps([FILING]).encode()).with_columns(
            pl.lit(2191000000.0, dtype=pl.Float64).alias("revenue")
        )
        store.write(frame)
        assert check_fundamentals(tmp_path).is_clean


class TestEvents:
    def write(self, lake: Path, observed: date, event_date: date, resolved: bool = True):
        payload = json.dumps(
            [
                {
                    "symbol": "RELIANCE",
                    "company": "Reliance",
                    "date": event_date.strftime("%d-%b-%Y"),
                    "purpose": "Financial Results",
                    "bm_desc": "",
                }
            ]
        ).encode()
        events = parse_events(payload, observed)
        mapping = {"RELIANCE": "NSE:INE002A01018"} if resolved else {}
        EventStore(lake).write(observed, resolve(events, mapping).rows)

    def test_nothing_held_is_a_warning(self, tmp_path) -> None:
        assert severities(check_events(tmp_path))["absent"] is Severity.WARN

    def test_a_stale_calendar_is_critical(self, tmp_path) -> None:
        """This feed has no archive, so a stale calendar reports quiet it
        cannot vouch for -- an alert rule reading it gives a false all-clear."""
        today = date(2026, 9, 20)
        self.write(tmp_path, today - timedelta(days=CALENDAR_STALE_DAYS + 3), today)
        found = check_events(tmp_path, datetime(2026, 9, 20, tzinfo=UTC))
        assert severities(found)["stale"] is Severity.CRITICAL

    def test_a_fresh_calendar_is_clean(self, tmp_path) -> None:
        today = date(2026, 9, 20)
        self.write(tmp_path, today, today + timedelta(days=2))
        assert check_events(tmp_path, datetime(2026, 9, 20, tzinfo=UTC)).is_clean
