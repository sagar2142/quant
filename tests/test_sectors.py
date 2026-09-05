"""Industry classification — MASTER_PLAN §1.1, §3.3, §8.

Nothing in this system knew what a company does. Every factor, cluster and
exposure came from price, which is defensible for concentration and leaves a
book eleven-thirtieths in one industry looking, from every screen, like thirty
independent positions.

Two things here are load-bearing rather than incidental:

  - **when a label was observed**, because NSE publishes no archive and a
    present-day classification quietly presenting itself as contemporaneous is
    the look-ahead this system exists to refuse
  - **coverage**, because a company delisted before the observation appears in
    no current constituent list, so history is missing exactly the names that
    failed — the direction that flatters
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from data.feeds.nse_sectors import (
    BROAD_LISTS,
    PRIMARY_LIST,
    SectorFormatError,
    constituent_url,
    parse_constituents,
)
from data.store.sectors import SectorStore

HEADER = "Company Name,Industry,Symbol,Series,ISIN Code"


def csv_bytes(rows: list[str]) -> bytes:
    return ("\n".join([HEADER, *rows]) + "\n").encode()


def row(company: str, industry: str, symbol: str, isin: str) -> str:
    return f"{company},{industry},{symbol},EQ,{isin}"


SAMPLE = csv_bytes(
    [
        row("HDFC Bank Ltd.", "Financial Services", "HDFCBANK", "INE040A01034"),
        row("Infosys Ltd.", "Information Technology", "INFY", "INE009A01021"),
        row("Sun Pharma Ltd.", "Healthcare", "SUNPHARMA", "INE044A01036"),
    ]
)


class TestUrl:
    def test_it_carries_no_date(self) -> None:
        """Unlike the bhavcopy and the index closes, NSE serves the *current*
        membership at a fixed address and publishes no archive. That absence is
        the whole reason observations are stamped with a fetch date."""
        url = constituent_url(PRIMARY_LIST)
        assert PRIMARY_LIST in url
        assert not any(ch.isdigit() for ch in url.rsplit("/", maxsplit=1)[-1])

    def test_the_primary_is_first(self) -> None:
        assert BROAD_LISTS[0] == PRIMARY_LIST


class TestParse:
    def test_it_reads_industry_and_isin(self) -> None:
        parsed = parse_constituents(SAMPLE, PRIMARY_LIST)
        assert parsed.rows.height == 3
        found = dict(zip(parsed.rows["isin"], parsed.rows["industry"], strict=True))
        assert found["INE040A01034"] == "Financial Services"

    def test_it_refuses_a_web_page(self) -> None:
        with pytest.raises(SectorFormatError, match="web page"):
            parse_constituents(b"<!DOCTYPE html><html></html>", PRIMARY_LIST)

    def test_it_refuses_an_unexpected_layout(self) -> None:
        with pytest.raises(SectorFormatError, match="missing columns"):
            parse_constituents(b"Symbol,Sector\nINFY,IT\n", PRIMARY_LIST)

    def test_a_row_without_an_isin_is_dropped(self) -> None:
        """It cannot be joined to anything this system holds, and an instrument
        id is ISIN-keyed (§1.1)."""
        payload = csv_bytes([row("Ghost Ltd.", "Healthcare", "GHOST", "")])
        assert parse_constituents(payload, PRIMARY_LIST).rows.is_empty()

    def test_a_row_without_an_industry_is_dropped(self) -> None:
        """A blank is not a classification, and storing it would create a
        category that reads as real."""
        payload = csv_bytes([row("Ghost Ltd.", "", "GHOST", "INE000A01001")])
        assert parse_constituents(payload, PRIMARY_LIST).rows.is_empty()

    def test_the_symbol_is_upper_cased(self) -> None:
        payload = csv_bytes([row("Infosys Ltd.", "Information Technology", "infy", "INE009A01021")])
        assert parse_constituents(payload, PRIMARY_LIST).rows["symbol"][0] == "INFY"

    def test_the_source_list_is_recorded(self) -> None:
        """So a coverage gap can be traced to a list that failed rather than to
        a name that genuinely has no industry."""
        parsed = parse_constituents(SAMPLE, "ind_nifty500list")
        assert set(parsed.rows["source"].to_list()) == {"ind_nifty500list"}


class TestStore:
    def store(self, tmp_path) -> SectorStore:
        store = SectorStore(tmp_path)
        store.write(date(2026, 9, 1), parse_constituents(SAMPLE, PRIMARY_LIST).rows)
        return store

    def test_it_round_trips(self, tmp_path) -> None:
        view = self.store(tmp_path).view()
        assert view.observed_at == date(2026, 9, 1)
        assert view.industry_of("NSE:INE009A01021") == "Information Technology"

    def test_an_instrument_id_is_accepted_whole(self, tmp_path) -> None:
        """Every caller holds `VENUE:ISIN`. Splitting it at each call site is
        how the two drift apart."""
        view = self.store(tmp_path).view()
        assert view.industry_of("BSE:INE040A01034") == "Financial Services"

    def test_an_unknown_name_is_none_not_a_category(self, tmp_path) -> None:
        assert self.store(tmp_path).view().industry_of("NSE:INE999Z01999") is None

    def test_an_empty_store_classifies_nothing(self, tmp_path) -> None:
        view = SectorStore(tmp_path).view()
        assert view.industries == {}
        assert view.observed_at is None

    def test_rewriting_a_date_replaces_it(self, tmp_path) -> None:
        store = self.store(tmp_path)
        moved = csv_bytes([row("Infosys Ltd.", "Services", "INFY", "INE009A01021")])
        store.write(date(2026, 9, 1), parse_constituents(moved, PRIMARY_LIST).rows)
        assert store.view().industry_of("NSE:INE009A01021") == "Services"

    def test_a_frame_missing_columns_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            SectorStore(tmp_path).write(date(2026, 9, 1), pl.DataFrame({"isin": ["X"]}))


class TestObservationDates:
    """NSE publishes no archive, so how far a label reaches has to be visible."""

    def store(self, tmp_path) -> SectorStore:
        store = SectorStore(tmp_path)
        for day in (date(2026, 1, 5), date(2026, 6, 1), date(2026, 9, 1)):
            store.write(day, parse_constituents(SAMPLE, PRIMARY_LIST).rows)
        return store

    def test_it_reads_the_newest_observation_at_or_before_the_decision(self, tmp_path) -> None:
        view = self.store(tmp_path).view(as_of=as_decision_time(datetime(2026, 7, 1, tzinfo=UTC)))
        assert view.observed_at == date(2026, 6, 1)

    def test_a_later_observation_cannot_reach_an_earlier_decision(self, tmp_path) -> None:
        """The September labels must not describe a June decision. That is the
        ordinary point-in-time rule, applied to a classification."""
        view = self.store(tmp_path).view(as_of=as_decision_time(datetime(2026, 6, 15, tzinfo=UTC)))
        assert view.observed_at == date(2026, 6, 1)
        assert view.reaching_back <= 0

    def test_reaching_back_is_reported_when_a_label_is_carried(self, tmp_path) -> None:
        """Before any observation there is no contemporaneous answer. The
        oldest is still the honest one for an industry — a bank was a bank —
        but the caller is told how far it is being carried."""
        view = self.store(tmp_path).view(as_of=as_decision_time(datetime(2019, 1, 1, tzinfo=UTC)))
        assert view.observed_at == date(2026, 1, 5)
        assert view.reaching_back > 2000

    def test_no_as_of_reads_the_newest(self, tmp_path) -> None:
        """Right for anything present-tense: a risk limit on the book as it
        stands has no look-ahead to commit."""
        assert self.store(tmp_path).view().observed_at == date(2026, 9, 1)

    def test_observations_are_listed_ascending(self, tmp_path) -> None:
        assert self.store(tmp_path).observations() == [
            date(2026, 1, 5),
            date(2026, 6, 1),
            date(2026, 9, 1),
        ]


class TestCoverage:
    def test_it_reports_the_share_classified(self, tmp_path) -> None:
        store = SectorStore(tmp_path)
        store.write(date(2026, 9, 1), parse_constituents(SAMPLE, PRIMARY_LIST).rows)
        view = store.view()
        members = ["NSE:INE040A01034", "NSE:INE009A01021", "NSE:INE999Z01999"]
        assert view.coverage(members) == pytest.approx(2 / 3)

    def test_an_empty_universe_is_zero_not_a_division(self, tmp_path) -> None:
        assert SectorStore(tmp_path).view().coverage([]) == 0.0


class TestGroupsAreNamespaced:
    """Two groupings feed one risk limit and must not be mistaken for each
    other: correlation answers "do these move together", industry answers "are
    these the same business". A breach message has to say which fired."""

    def test_correlation_labels_carry_their_prefix(self) -> None:
        from quant.analytics.clusters import CORRELATION_PREFIX

        assert CORRELATION_PREFIX == "corr:"

    def test_industry_labels_carry_theirs(self) -> None:
        from apps.api.trade import INDUSTRY_PREFIX, UNCLASSIFIED

        assert INDUSTRY_PREFIX == "industry:"
        assert UNCLASSIFIED.startswith(INDUSTRY_PREFIX)

    def test_the_two_namespaces_cannot_collide(self) -> None:
        from apps.api.trade import INDUSTRY_PREFIX
        from quant.analytics.clusters import CORRELATION_PREFIX

        assert not INDUSTRY_PREFIX.startswith(CORRELATION_PREFIX)
        assert not CORRELATION_PREFIX.startswith(INDUSTRY_PREFIX)
