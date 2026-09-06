"""Quarterly results — MASTER_PLAN §3.3, §9.

Two things here can be wrong in ways that produce a plausible number rather
than an error, and both are the kind that flatter a backtest:

  - reading a filing **before it was published**, which is look-ahead of the
    most valuable sort, because earnings move prices
  - taking the **cumulative** figure for the quarter's, which overstates
    revenue by roughly the number of quarters elapsed

Neither shows up as a failure downstream. They show up as alpha.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from data.feeds.nse_results import (
    ResultsFormatError,
    XbrlFacts,
    filings_url,
    has_xbrl_document,
    parse_filings,
    parse_xbrl,
    xbrl_document_expr,
    xbrl_is_plausible,
)
from data.store.fundamentals import FundamentalStore

# A Q3 filing: period Oct-Dec, disseminated at the end of January.
FILING = {
    "isin": "INE764D01017",
    "symbol": "VSTTILLERS",
    "companyName": "V.S.T Tillers Tractors Limited",
    "fromDate": "01-Oct-2024",
    "toDate": "31-Dec-2024",
    "filingDate": "30-Jan-2025 17:17",
    "broadCastDate": "30-Jan-2025 17:17:53",
    "exchdisstime": "30-Jan-2025 17:18:42",
    "consolidated": "Consolidated",
    "audited": "Un-Audited",
    "xbrl": "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1.xml",
}

# Both contexts declare the *same* dates. This is the real shape of an NSE
# Ind-AS document, and the reason facts are selected by context id.
XBRL = b"""<?xml version="1.0" encoding="UTF-8"?>
<xbrl xmlns="http://www.xbrl.org/2003/instance"
      xmlns:in-bse-fin="http://www.bseindia.com/xbrl/fin/2020-03-31/in-bse-fin">
  <context id="OneD">
    <period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>
  </context>
  <context id="FourD">
    <period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>
  </context>
  <in-bse-fin:RevenueFromOperations contextRef="OneD">2191000000.00</in-bse-fin:RevenueFromOperations>
  <in-bse-fin:RevenueFromOperations contextRef="FourD">6931200000.00</in-bse-fin:RevenueFromOperations>
  <in-bse-fin:ProfitBeforeTax contextRef="OneD">36000000.00</in-bse-fin:ProfitBeforeTax>
  <in-bse-fin:ProfitBeforeTax contextRef="FourD">108000000.00</in-bse-fin:ProfitBeforeTax>
  <in-bse-fin:ProfitLossForPeriod contextRef="OneD">12800000.00</in-bse-fin:ProfitLossForPeriod>
  <in-bse-fin:BasicEarningsLossPerShare contextRef="OneD">1.48</in-bse-fin:BasicEarningsLossPerShare>
</xbrl>
"""


def payload(*rows: dict) -> bytes:
    return json.dumps(list(rows)).encode()


class TestFilingParsing:
    def test_it_reads_a_filing(self) -> None:
        frame = parse_filings(payload(FILING))
        assert frame.height == 1
        assert frame["isin"][0] == "INE764D01017"
        assert frame["period_end"][0] == date(2024, 12, 31)

    def test_receive_time_is_the_dissemination_time(self) -> None:
        """`filingDate` is when the company submitted; `exchdisstime` is when
        the market could act. Taking the earlier one claims knowledge before
        the exchange published it."""
        frame = parse_filings(payload(FILING))
        assert frame["receive_time"][0] == datetime(2025, 1, 30, 17, 18, 42, tzinfo=UTC)

    def test_the_period_and_the_publication_are_months_apart(self) -> None:
        """The gap is the entire point of the store. A backtest reading the
        December quarter in December is reading the future."""
        frame = parse_filings(payload(FILING))
        gap = frame["receive_time"][0].date() - frame["period_end"][0]
        assert gap.days == 30

    def test_a_row_without_an_isin_is_dropped(self) -> None:
        """It cannot be joined to anything this system holds (§1.1)."""
        assert parse_filings(payload({**FILING, "isin": ""})).is_empty()

    def test_a_row_with_no_publication_time_is_dropped(self) -> None:
        """Unreadable point-in-time is worse than absent: it would be visible
        at every decision or none."""
        stripped = {
            k: v
            for k, v in FILING.items()
            if k not in ("exchdisstime", "broadCastDate", "filingDate")
        }
        assert parse_filings(payload(stripped)).is_empty()

    def test_consolidated_is_recorded(self) -> None:
        frame = parse_filings(payload(FILING, {**FILING, "consolidated": "Standalone"}))
        assert sorted(frame["consolidated"].to_list()) == [False, True]

    def test_html_is_refused(self) -> None:
        with pytest.raises(ResultsFormatError, match="HTML"):
            parse_filings(b"<html><body>blocked</body></html>")

    def test_a_response_without_filings_is_refused(self) -> None:
        with pytest.raises(ResultsFormatError, match="no list"):
            parse_filings(b'{"ok": true}')

    def test_a_changed_layout_is_refused_not_nulled(self) -> None:
        """A silent layout change would write a table of nulls that every
        factor reads as 'no company ever reported'."""
        with pytest.raises(ResultsFormatError, match="missing every one"):
            parse_filings(payload({"companyId": 1, "value": 2}))

    def test_the_window_reaches_the_archive(self) -> None:
        url = filings_url("01-01-2020", "31-12-2020")
        assert "from_date=01-01-2020" in url and "to_date=31-12-2020" in url


class TestXbrlContexts:
    """The trap that silently triples revenue."""

    def test_it_takes_the_quarter_not_the_cumulative(self) -> None:
        facts = parse_xbrl(XBRL)
        assert facts.values["revenue"] == Decimal("2191000000.00")
        assert facts.context == "OneD"

    def test_the_cumulative_value_is_never_returned(self) -> None:
        """Both contexts declare 2024-10-01 to 2024-12-31, so selecting by
        declared period would return either one depending on document order."""
        facts = parse_xbrl(XBRL)
        assert facts.values["revenue"] != Decimal("6931200000.00")
        assert facts.values["profit_before_tax"] == Decimal("36000000.00")

    def test_a_cumulative_only_document_is_refused(self) -> None:
        """Its numbers are real but they are not the quarter's, and a table
        mixing the two is worse than one missing the name."""
        only_four = XBRL.replace(b'contextRef="OneD"', b'contextRef="FourD"')
        with pytest.raises(ResultsFormatError, match="cumulative"):
            parse_xbrl(only_four)

    def test_a_segment_context_is_not_mistaken_for_the_company(self) -> None:
        """`OneReportableSegmentRevenue01D` is a real context in a real filing.
        It starts with `One` and holds one business segment's revenue, so a
        prefix match would record a division's numbers as the company's in any
        document that emits segments before totals."""
        segmented = XBRL.replace(
            b'<context id="OneD">',
            b'<context id="OneReportableSegmentRevenue01D">'
            b"<period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>"
            b'<scenario><explicitMember dimension="in-bse-fin:ReportableSegmentsAxis">'
            b"seg1</explicitMember></scenario></context>"
            b'<context id="OneD">',
        ).replace(
            b'<in-bse-fin:RevenueFromOperations contextRef="OneD">2191000000.00',
            b'<in-bse-fin:RevenueFromOperations contextRef="OneReportableSegmentRevenue01D">'
            b"7000000.00</in-bse-fin:RevenueFromOperations>"
            b'<in-bse-fin:RevenueFromOperations contextRef="OneD">2191000000.00',
        )
        facts = parse_xbrl(segmented)
        assert facts.values["revenue"] == Decimal("2191000000.00")
        assert facts.context == "OneD"

    def test_an_instant_context_is_not_used_for_a_flow(self) -> None:
        """`OneI` carries balance-sheet positions. Every fact here is a flow
        over the quarter, so an instant would be a different quantity wearing
        the same tag."""
        instant_only = XBRL.replace(b'id="OneD"', b'id="OneI"').replace(
            b'<startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period>\n  </context>\n  <context id="FourD">',
            b'<instant>2024-12-31</instant></period>\n  </context>\n  <context id="FourD">',
        )
        with pytest.raises(ResultsFormatError):
            parse_xbrl(instant_only.replace(b'contextRef="OneI"', b'contextRef="OneI"'))

    def test_every_expected_fact_is_extracted(self) -> None:
        facts = parse_xbrl(XBRL)
        assert facts.values["net_profit"] == Decimal("12800000.00")
        assert facts.values["eps_basic"] == Decimal("1.48")

    def test_a_document_with_no_known_facts_is_refused(self) -> None:
        with pytest.raises(ResultsFormatError, match="none of the expected"):
            parse_xbrl(b'<?xml version="1.0"?><xbrl><Something>1</Something></xbrl>')

    def test_broken_xml_is_refused(self) -> None:
        with pytest.raises(ResultsFormatError, match="not parseable"):
            parse_xbrl(b"<xbrl><unclosed>")

    def test_an_external_entity_is_not_resolved(self) -> None:
        """The document is fetched over the network. An entity that reads a
        local file is a capability worth denying outright."""
        hostile = (
            b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>'
            b'<xbrl><RevenueFromOperations contextRef="OneD">&e;</RevenueFromOperations></xbrl>'
        )
        try:
            facts = parse_xbrl(hostile)
        except ResultsFormatError:
            return
        assert "root:" not in str(facts.values)


class TestDocumentLink:
    """NSE writes the filename as a dash when no document was filed."""

    def test_a_dash_filename_is_not_a_document(self) -> None:
        """`.../corporate/xbrl/-` is well-formed, on the right host, and 404s.
        A scheme check passes it; measured on a real window, most filings in a
        year carry it."""
        assert not has_xbrl_document("https://nsearchives.nseindia.com/corporate/xbrl/-")

    def test_a_real_document_passes(self) -> None:
        assert has_xbrl_document(
            "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_121277_1705282.xml"
        )

    def test_an_empty_link_is_not_a_document(self) -> None:
        assert not has_xbrl_document("")
        assert not has_xbrl_document("-")


class TestPlausibility:
    def test_an_absurd_revenue_is_refused(self) -> None:
        """A units error looks like spectacular growth to a ranking factor."""
        facts = XbrlFacts(values={"revenue": Decimal("1e20")}, context="OneD")
        assert "units error" in xbrl_is_plausible(facts)

    def test_a_negative_revenue_is_refused(self) -> None:
        assert xbrl_is_plausible(XbrlFacts({"revenue": Decimal(-5)}, "OneD"))

    def test_a_real_filing_passes(self) -> None:
        assert xbrl_is_plausible(parse_xbrl(XBRL)) == ""

    def test_a_loss_is_not_implausible(self) -> None:
        """Negative profit is ordinary; only negative revenue is nonsense."""
        assert xbrl_is_plausible(XbrlFacts({"net_profit": Decimal(-100)}, "OneD")) == ""


class TestPointInTime:
    """The reason this store exists."""

    def store(self, tmp_path) -> FundamentalStore:
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        return store

    def test_a_filing_is_invisible_before_it_was_published(self, tmp_path) -> None:
        """The December quarter is not readable in December. It was published
        on 30 January."""
        view = self.store(tmp_path).view(as_decision_time(datetime(2025, 1, 15, tzinfo=UTC)))
        assert view.rows.is_empty()
        assert view.withheld == 1

    def test_it_becomes_visible_after_publication(self, tmp_path) -> None:
        view = self.store(tmp_path).view(as_decision_time(datetime(2025, 2, 1, tzinfo=UTC)))
        assert view.rows.height == 1
        assert view.withheld == 0

    def test_the_boundary_is_the_dissemination_moment(self, tmp_path) -> None:
        """Same day, minutes earlier, is still not published."""
        store = self.store(tmp_path)
        before = store.view(as_decision_time(datetime(2025, 1, 30, 17, 0, tzinfo=UTC)))
        after = store.view(as_decision_time(datetime(2025, 1, 30, 17, 19, tzinfo=UTC)))
        assert before.rows.is_empty()
        assert after.rows.height == 1

    def test_withheld_is_reported_not_silent(self, tmp_path) -> None:
        """A large count means the read is early, not that nobody reported."""
        view = self.store(tmp_path).view(as_decision_time(datetime(2024, 1, 1, tzinfo=UTC)))
        assert view.withheld == 1
        assert view.names == 0

    def test_no_as_of_reads_everything(self, tmp_path) -> None:
        """Correct only for present-tense questions, and it says so."""
        assert self.store(tmp_path).view().rows.height == 1


class TestStore:
    def test_a_refiled_correction_replaces_the_original(self, tmp_path) -> None:
        """Same period, later dissemination: what a reader later would see."""
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        corrected = {**FILING, "exchdisstime": "15-Feb-2025 10:00:00"}
        store.write(parse_filings(payload(corrected)))
        rows = store.view().rows
        assert rows.height == 1
        assert rows["receive_time"][0] == datetime(2025, 2, 15, 10, 0, tzinfo=UTC)

    def test_consolidated_and_standalone_both_survive(self, tmp_path) -> None:
        """A group and its parent are different companies."""
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING, {**FILING, "consolidated": "Standalone"})))
        assert store.view().rows.height == 2

    def test_it_partitions_by_period_year(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        assert store.years() == [2024]

    def test_rewriting_the_same_window_changes_nothing(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        store.write(parse_filings(payload(FILING)))
        assert store.view().rows.height == 1

    def test_a_frame_missing_columns_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            FundamentalStore(tmp_path).write(pl.DataFrame({"isin": ["X"]}))

    def test_an_empty_store_reads_empty(self, tmp_path) -> None:
        view = FundamentalStore(tmp_path).view()
        assert view.rows.is_empty()
        assert view.names == 0

    def test_latest_prefers_consolidated_on_a_tie(self, tmp_path) -> None:
        """The group's economics are what a factor on a group should read."""
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING, {**FILING, "consolidated": "Standalone"})))
        latest = store.view().latest()
        assert latest.height == 1
        assert bool(latest["consolidated"][0]) is True

    def test_age_says_how_stale_the_newest_result_is(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        aged = store.view(as_decision_time(datetime(2025, 3, 1, tzinfo=UTC))).age_days()
        assert aged["age_days"][0] == 29

    def test_the_vectorised_form_agrees_with_the_scalar_one(self) -> None:
        """Two definitions of one rule drift. This pins them together."""
        cases = [
            "https://nsearchives.nseindia.com/corporate/xbrl/-",
            "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1.xml",
            "https://nsearchives.nseindia.com/corporate/xbrl/INDAS_1.XML",
            "ftp://elsewhere/x.xml",
            "",
            "-",
        ]
        frame = pl.DataFrame({"xbrl_url": cases})
        vectorised = frame.select(xbrl_document_expr())["xbrl_url"].to_list()
        assert vectorised == [has_xbrl_document(c) for c in cases]


class TestStaleness:
    """How current the store is — a scheduling question, not a research one."""

    def test_an_empty_store_has_no_newest_filing(self, tmp_path) -> None:
        assert FundamentalStore(tmp_path).newest_filing() is None

    def test_it_reports_the_latest_publication(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        assert store.newest_filing() == datetime(2025, 1, 30, 17, 18, 42, tzinfo=UTC)

    def test_it_spans_every_year_held(self, tmp_path) -> None:
        """Partitioning is by period year, so the newest publication can live
        in an older file than the newest period."""
        store = FundamentalStore(tmp_path)
        store.write(parse_filings(payload(FILING)))
        store.write(
            parse_filings(
                payload(
                    {
                        **FILING,
                        "isin": "INE000A01001",
                        "fromDate": "01-Jan-2023",
                        "toDate": "31-Mar-2023",
                        "exchdisstime": "01-Jun-2026 09:00:00",
                    }
                )
            )
        )
        assert store.years() == [2023, 2024]
        assert store.newest_filing() == datetime(2026, 6, 1, 9, 0, tzinfo=UTC)


class TestMergePreservesNumbers:
    """The same filing fetched twice, once with its XBRL read and once without.

    Identical `receive_time`, so nothing distinguishes them but the numbers.
    Without an explicit preference the survivor is arbitrary and a re-run
    silently drops numbers already fetched — row count unchanged, no error.
    """

    def with_revenue(self, value):
        frame = parse_filings(payload(FILING))
        return frame.with_columns(pl.lit(value, dtype=pl.Float64).alias("revenue"))

    def test_numbers_survive_a_later_bare_write(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(self.with_revenue(2191000000.0))
        store.write(self.with_revenue(None))
        assert store.view().rows["revenue"][0] == 2191000000.0

    def test_numbers_are_picked_up_by_a_later_write(self, tmp_path) -> None:
        store = FundamentalStore(tmp_path)
        store.write(self.with_revenue(None))
        store.write(self.with_revenue(2191000000.0))
        assert store.view().rows["revenue"][0] == 2191000000.0

    def test_a_genuine_correction_still_wins_on_time(self, tmp_path) -> None:
        """Preferring facts must not override the point-in-time rule: a later
        filing replaces an earlier one even if the earlier had more numbers."""
        store = FundamentalStore(tmp_path)
        store.write(self.with_revenue(2191000000.0))
        later = parse_filings(payload({**FILING, "exchdisstime": "15-Feb-2025 10:00:00"}))
        store.write(later.with_columns(pl.lit(999.0, dtype=pl.Float64).alias("revenue")))
        rows = store.view().rows
        assert rows.height == 1
        assert rows["revenue"][0] == 999.0

    def test_the_row_count_does_not_change(self, tmp_path) -> None:
        """The bug's signature: nothing to see in the count."""
        store = FundamentalStore(tmp_path)
        store.write(self.with_revenue(2191000000.0))
        assert store.write(self.with_revenue(None)) == 1


class TestProvenanceIsARecord:
    def test_facts_come_from_exactly_one_context(self) -> None:
        """Measured on 25 real filings, every document has one usable context.
        Taking facts from two while reporting one would make `context` a claim
        rather than a record."""
        two = b"""<?xml version="1.0"?>
<xbrl>
  <context id="OneA"><period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period></context>
  <context id="OneB"><period><startDate>2024-10-01</startDate><endDate>2024-12-31</endDate></period></context>
  <RevenueFromOperations contextRef="OneA">100</RevenueFromOperations>
  <ProfitBeforeTax contextRef="OneB">999</ProfitBeforeTax>
</xbrl>"""
        facts = parse_xbrl(two)
        assert facts.context == "OneA"
        assert facts.values == {"revenue": Decimal("100")}


class TestNullPeriodIsRefusedByName:
    def test_a_filing_with_no_period_is_refused(self, tmp_path) -> None:
        """Partitioning is by period year, so a null period has nowhere to
        live. It should say so, not surface as int(None) in a group-by."""
        frame = parse_filings(payload(FILING)).with_columns(
            pl.lit(None, dtype=pl.Date).alias("period_end")
        )
        with pytest.raises(ValueError, match="no period_end"):
            FundamentalStore(tmp_path).write(frame)
