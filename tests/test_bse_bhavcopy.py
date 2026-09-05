"""BSE feed — MASTER_PLAN §1.1, §13.4.

The BSE adapter had no test file at all — an empty one was sitting in the
directory. Every property here fails silently rather than loudly if it breaks:
a wrong venue prefix produces an id that looks entirely plausible while
splitting or merging a name's history, and an unfiltered T+0 row doubles a
security in the cross-section at a handful of shares' liquidity.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from data.feeds.bse import (
    BSE_SERIES,
    T0_SUFFIX,
    bse_instrument_id,
    bse_udiff_url,
    is_parallel_settlement,
)
from data.feeds.nse import nse_instrument_id


class TestUrl:
    def test_it_names_the_session(self) -> None:
        """YYYYMMDD, unlike the NSE index archive's DDMMYYYY. The three feeds
        sit beside each other and each uses a different date format."""
        assert "20240101" in bse_udiff_url(date(2024, 1, 1))

    def test_it_is_served_uncompressed(self) -> None:
        """NSE ships a zip and BSE does not, so the loader sniffs the payload.
        Asserted because a `.zip` here would be unpacked into nothing."""
        assert not bse_udiff_url(date(2024, 6, 3)).endswith(".zip")


class TestSeries:
    def test_the_equity_groups_are_a_closed_set(self) -> None:
        """BSE groups by letter where NSE uses a series code. An `EQ` filter —
        the NSE rule — matches nothing at all in a BSE file, which reads as an
        empty session rather than as a bug."""
        assert "EQ" not in BSE_SERIES
        assert "A" in BSE_SERIES and "B" in BSE_SERIES


class TestInstrumentIdentity:
    """The scalar helper and the ingest expression must agree.

    `apps.cli.ingest_nse` builds instrument ids with a vectorised Polars
    expression because it does so for thousands of rows at a time; the feed
    modules carry the same rule as a scalar function with the reasoning
    attached. Two implementations of one rule drift, and this drift would be
    invisible.
    """

    def test_the_venue_prefixes_an_isin(self) -> None:
        assert bse_instrument_id("INE002A01018", "RELIANCE") == "BSE:INE002A01018"

    def test_a_missing_isin_falls_back_to_the_ticker(self) -> None:
        """Not a bare prefix. A row with no ISIN still has to be holdable, and
        `BSE:` alone would collide every such name onto one instrument."""
        assert bse_instrument_id("", "reliance") == "BSE:RELIANCE"

    def test_it_matches_the_vectorised_ingest_expression(self) -> None:
        rows = pl.DataFrame(
            {"isin": ["INE002A01018", "", "INE009A01021"], "symbol": ["RELIANCE", "NEWCO", "INFY"]}
        )
        vectorised = (
            "BSE:" + rows["isin"].zip_with(rows["isin"].str.len_chars() > 0, rows["symbol"])
        ).to_list()
        scalar = [
            bse_instrument_id(i, s) for i, s in zip(rows["isin"], rows["symbol"], strict=True)
        ]
        assert vectorised == scalar

    def test_the_two_venues_are_different_instruments(self) -> None:
        """A dual-listed name shares an ISIN and trades at two different
        prices. Collapsing them onto one id would let a backtest fill on
        whichever venue happened to be cheaper that session — not a strategy
        anyone can trade."""
        assert bse_instrument_id("INE002A01018", "RELIANCE") != nse_instrument_id(
            "INE002A01018", "RELIANCE"
        )


class TestParallelSettlement:
    def test_the_t0_copy_is_recognised(self) -> None:
        """From 2024-03-28 the bhavcopy carries a second row per participating
        name: same ISIN, same close, a handful of shares. INDHOTEL printed
        167,607 shares and INDHOTEL# printed one."""
        assert is_parallel_settlement(f"INDHOTEL{T0_SUFFIX}")

    def test_an_ordinary_symbol_is_not(self) -> None:
        assert not is_parallel_settlement("INDHOTEL")

    def test_a_suffix_elsewhere_does_not_count(self) -> None:
        """Only a trailing marker. A `#` in the middle is part of the ticker."""
        assert not is_parallel_settlement(f"IND{T0_SUFFIX}HOTEL")
