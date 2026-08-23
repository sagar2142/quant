"""Delayed quotes for display — MASTER_PLAN §3.3, §12.7.

The risk this carries is not a wrong price, it is a *stale price wearing a
fresh label*. NSE publishes no free real-time feed, so every quote here is
delayed; the tests are mostly about that delay surviving intact to the caller.
"""

from __future__ import annotations

import httpx
import pytest

from core.clock import utc_now
from data.feeds.quotes import Quote, fetch_quotes


def chart(price: float | None, struck: int | None, symbol: str = "RELIANCE.NS") -> dict:
    meta: dict[str, object] = {"symbol": symbol, "currency": "INR"}
    if price is not None:
        meta["regularMarketPrice"] = price
    if struck is not None:
        meta["regularMarketTime"] = struck
    return {"chart": {"result": [{"meta": meta}]}}


def client_returning(payloads: dict[str, object], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        symbol = request.url.path.rsplit("/", 1)[-1]
        body = payloads.get(symbol)
        if body is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestFetching:
    def test_a_quote_carries_the_vendor_timestamp_not_arrival(self):
        """Stamping arrival time onto a delayed quote is how a fifteen-minute-old
        price starts looking current."""
        struck = int(utc_now().timestamp()) - 900
        with client_returning({"RELIANCE.NS": chart(1300.0, struck)}) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        quote = result.quotes["RELIANCE"]
        assert int(quote.quoted_at.timestamp()) == struck
        assert quote.age_seconds >= 890

    def test_the_ns_suffix_is_not_the_caller_s_problem(self):
        with client_returning({"TCS.NS": chart(3000.0, 1787305502, "TCS.NS")}) as c:
            result = fetch_quotes(("TCS",), client=c)
        assert "TCS" in result.quotes
        assert result.quotes["TCS"].symbol == "TCS"

    def test_prices_are_decimal_not_float(self):
        """Money does not stay a float (§14.1.2)."""
        with client_returning({"RELIANCE.NS": chart(1300.55, 1787305502)}) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        assert str(result.quotes["RELIANCE"].price) == "1300.55"

    def test_symbols_are_deduplicated(self):
        with client_returning({"RELIANCE.NS": chart(1300.0, 1787305502)}) as c:
            result = fetch_quotes(("RELIANCE", "reliance", " RELIANCE "), client=c)
        assert len(result.quotes) == 1

    def test_no_symbols_is_not_a_request(self):
        result = fetch_quotes(())
        assert result.quotes == {}
        assert result.failures == {}


class TestFailuresAreReported:
    """A missing quote and a quote of zero are different, and a caller that
    cannot tell them apart will eventually treat one as the other."""

    def test_an_unknown_ticker_is_a_named_failure(self):
        with client_returning({}) as c:
            result = fetch_quotes(("NOTAREAL",), client=c)
        assert "NOTAREAL" in result.failures
        assert result.quotes == {}

    def test_a_row_without_a_price_is_dropped_not_defaulted(self):
        with client_returning({"RELIANCE.NS": chart(None, 1787305502)}) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        assert "RELIANCE" not in result.quotes
        assert "RELIANCE" in result.failures

    def test_a_row_without_a_timestamp_is_dropped(self):
        """An unaged quote is indistinguishable from a fresh one on screen."""
        with client_returning({"RELIANCE.NS": chart(1300.0, None)}) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        assert "RELIANCE" not in result.quotes

    def test_a_server_error_does_not_raise(self):
        """A console that 500s because a quote vendor is down is less useful
        than one showing last night's close and saying so."""
        with client_returning({"RELIANCE.NS": chart(1300.0, 1)}, status=500) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        assert result.quotes == {}
        assert "RELIANCE" in result.failures

    def test_one_bad_symbol_does_not_lose_the_good_ones(self):
        with client_returning({"TCS.NS": chart(3000.0, 1787305502, "TCS.NS")}) as c:
            result = fetch_quotes(("TCS", "NOTAREAL"), client=c)
        assert "TCS" in result.quotes
        assert "NOTAREAL" in result.failures


class TestStaleness:
    def test_stalest_is_the_oldest_quote(self):
        now = int(utc_now().timestamp())
        payloads = {
            "AAA.NS": chart(1.0, now - 60, "AAA.NS"),
            "BBB.NS": chart(2.0, now - 3600, "BBB.NS"),
        }
        with client_returning(payloads) as c:
            result = fetch_quotes(("AAA", "BBB"), client=c)
        assert result.stalest_seconds == pytest.approx(3600, abs=30)

    def test_no_quotes_means_no_staleness_rather_than_zero(self):
        """Zero seconds stale would mean a price that just printed."""
        assert fetch_quotes(()).stalest_seconds is None

    def test_a_future_timestamp_is_not_negative_age(self):
        future = int(utc_now().timestamp()) + 600
        with client_returning({"RELIANCE.NS": chart(1300.0, future)}) as c:
            result = fetch_quotes(("RELIANCE",), client=c)
        assert result.quotes["RELIANCE"].age_seconds >= 0


class TestResearchIsolation:
    def test_quotes_are_not_panel_rows(self):
        """A Quote carries no receive_time, and that is deliberate: it has no
        point-in-time history and must never be joined to the panel (§3)."""
        fields = set(Quote.__dataclass_fields__)
        assert "receive_time" not in fields
        assert fields == {"symbol", "price", "quoted_at", "currency"}
