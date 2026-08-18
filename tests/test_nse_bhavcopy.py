"""NSE bhavcopy parsing (§M2). Both format eras, synthetic files — no network."""

from __future__ import annotations

import io
import zipfile
from datetime import date, datetime, timedelta

import pytest

from core.clock import UTC
from data.feeds.nse import (
    PUBLICATION_LAG,
    BhavcopyFormatError,
    legacy_url,
    nse_instrument_id,
    parse_bhavcopy,
    udiff_url,
)

SESSION = date(2024, 3, 15)

LEGACY_CSV = b"""SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
RELIANCE,EQ,2900.00,2950.00,2880.00,2940.00,2940.00,2895.00,5000000,14700000000.00,15-MAR-2024,120000,INE002A01018
TCS,EQ,3800.00,3850.00,3790.00,3820.00,3820.00,3805.00,2000000,7640000000.00,15-MAR-2024,80000,INE467B01029
ILLIQUID,BE,10.00,10.50,9.80,10.20,10.20,10.00,500,5100.00,15-MAR-2024,12,INE999X01011
"""

UDIFF_CSV = b"""TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd
2024-03-15,2024-03-15,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,2900.00,2950.00,2880.00,2940.00,2940.00,2895.00,5000000,14700000000.00,120000
2024-03-15,2024-03-15,CM,NSE,STK,11536,INE467B01029,TCS,EQ,3800.00,3850.00,3790.00,3820.00,3820.00,3805.00,2000000,7640000000.00,80000
"""


def zipped(payload: bytes, name: str = "bhav.csv") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, payload)
    return buffer.getvalue()


class TestLegacyFormat:
    def test_parses_equity_rows(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION)
        assert day.symbols == {"RELIANCE", "TCS"}

    def test_be_series_excluded_by_default(self):
        # Trade-for-trade segment: fills there are not realistic.
        assert "ILLIQUID" not in parse_bhavcopy(LEGACY_CSV, SESSION).symbols

    def test_series_filter_can_be_widened(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION, series=("EQ", "BE"))
        assert "ILLIQUID" in day.symbols

    def test_prices_parsed(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION)
        reliance = day.bars.filter(day.bars["symbol"] == "RELIANCE")
        assert reliance["close"][0] == pytest.approx(2940.0)
        assert reliance["volume"][0] == pytest.approx(5_000_000.0)

    def test_isin_captured(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION)
        isins = {listing.symbol: listing.isin for listing in day.listings}
        assert isins["RELIANCE"] == "INE002A01018"


class TestUdiffFormat:
    def test_parses_equity_rows(self):
        assert parse_bhavcopy(UDIFF_CSV, SESSION).symbols == {"RELIANCE", "TCS"}

    def test_matches_legacy_values(self):
        legacy = parse_bhavcopy(LEGACY_CSV, SESSION)
        udiff = parse_bhavcopy(UDIFF_CSV, SESSION)
        for frame in (legacy, udiff):
            row = frame.bars.filter(frame.bars["symbol"] == "TCS")
            assert row["close"][0] == pytest.approx(3820.0)

    def test_isin_captured(self):
        day = parse_bhavcopy(UDIFF_CSV, SESSION)
        assert {listing.isin for listing in day.listings} == {
            "INE002A01018",
            "INE467B01029",
        }


class TestZipHandling:
    def test_zipped_legacy(self):
        assert parse_bhavcopy(zipped(LEGACY_CSV), SESSION).symbols == {"RELIANCE", "TCS"}

    def test_zipped_udiff(self):
        assert parse_bhavcopy(zipped(UDIFF_CSV), SESSION).symbols == {"RELIANCE", "TCS"}

    def test_zip_without_csv_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", b"nope")
        with pytest.raises(BhavcopyFormatError, match="no CSV"):
            parse_bhavcopy(buffer.getvalue(), SESSION)


class TestTimestamps:
    """A strategy cannot act on today's close at today's close (§7.6)."""

    def test_event_time_is_session_close_utc(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION)
        # 15:30 IST == 10:00 UTC
        assert day.bars["event_time"][0] == datetime(2024, 3, 15, 10, 0, tzinfo=UTC)

    def test_receive_time_lags_publication(self):
        day = parse_bhavcopy(LEGACY_CSV, SESSION)
        gap = day.bars["receive_time"][0] - day.bars["event_time"][0]
        assert gap == PUBLICATION_LAG
        assert gap > timedelta(0)


class TestBadData:
    def test_unrecognised_layout_rejected(self):
        with pytest.raises(BhavcopyFormatError, match="unrecognised"):
            parse_bhavcopy(b"COL_A,COL_B\n1,2\n", SESSION)

    def test_empty_file_rejected(self):
        with pytest.raises(BhavcopyFormatError):
            parse_bhavcopy(b"", SESSION)

    def test_no_matching_series_rejected(self):
        with pytest.raises(BhavcopyFormatError, match="no rows in series"):
            parse_bhavcopy(LEGACY_CSV, SESSION, series=("XX",))

    def test_zero_price_rows_dropped(self):
        broken = LEGACY_CSV + (
            b"BROKEN,EQ,0.00,0.00,0.00,0.00,0.00,0.00,100,0.00,15-MAR-2024,1,INE000X01010\n"
        )
        assert "BROKEN" not in parse_bhavcopy(broken, SESSION).symbols

    def test_unparseable_numbers_dropped(self):
        broken = LEGACY_CSV + (b"JUNK,EQ,abc,def,ghi,jkl,x,y,100,0.00,15-MAR-2024,1,INE000X01011\n")
        assert "JUNK" not in parse_bhavcopy(broken, SESSION).symbols


class TestIdentity:
    def test_isin_preferred_over_symbol(self):
        # Symbols get renamed and recycled; ISINs do not.
        assert nse_instrument_id("INE002A01018", "RELIANCE") == "NSE:INE002A01018"

    def test_falls_back_to_symbol(self):
        assert nse_instrument_id("", "RELIANCE") == "NSE:RELIANCE"


class TestArchiveUrls:
    def test_legacy_url_shape(self):
        assert legacy_url(date(2020, 1, 3)).endswith("2020/JAN/cm03JAN2020bhav.csv.zip")

    def test_udiff_url_shape(self):
        assert "20240315" in udiff_url(SESSION)
