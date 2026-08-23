"""Live-ish quotes for display — MASTER_PLAN §3.3, §12.7.

**This is not a research feed and must never become one.** Everything in
`quant/` and `engine/` reads the panel, which carries `receive_time` on every
row so a decision can only see what had actually arrived. A quote fetched now
has no such history: it is today's price with no record of when it would have
been knowable, and joining it to the panel would silently destroy the
point-in-time discipline the whole system rests on (§3). Nothing here is
importable from the research path, and the import contract enforces that.

**What it is for.** The console marks a book at the previous session's close,
because that is the freshest price the panel holds. During a trading day that
number is hours old, and an operator looking at a position is entitled to know
what it is worth now rather than what it was worth last night.

**Delay is reported, never hidden.** NSE does not publish a free real-time
feed. Yahoo's quote endpoint carries a delay that varies by exchange and is
typically around fifteen minutes for NSE, and the response says when the price
was struck. That timestamp is passed through untouched: a quote labelled with
its own age is useful, and one presented as live is a lie that costs money the
first time somebody sizes an order against it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

import httpx

from core.clock import UTC, utc_now

__all__ = ["QUOTE_TIMEOUT", "Quote", "QuoteFetchResult", "fetch_quotes"]

#: Yahoo's chart endpoint, whose `meta` block carries the last traded price.
#:
#: The v7 batch quote endpoint would be one request for a whole book and has
#: required a crumb and cookie since 2023; anonymous calls get a bare 401. This
#: one still answers unauthenticated, at the cost of a request per symbol.
QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}.NS"

#: Seconds before a quote request is abandoned. Short on purpose: this decorates
#: a console that must stay responsive, and a slow quote is worth less than a
#: fast admission that there is none.
QUOTE_TIMEOUT = 4.0

#: Concurrent requests. One symbol per call means a thirty-name book is thirty
#: round trips, and doing them in series would take longer than the console's
#: poll interval. Modest on purpose — this is a courtesy endpoint, not one to
#: hammer.
CONCURRENCY = 8

#: One session at minute resolution: enough for `meta` to carry a current
#: price, small enough that the candle payload is not downloaded for nothing.
CHART_PARAMS = {"range": "1d", "interval": "1m"}


@dataclass(frozen=True)
class Quote:
    """One instrument's most recent traded price, and when it was struck."""

    symbol: str
    price: Decimal
    #: When the exchange recorded this trade, from the vendor's own field.
    #: Never `utc_now()`: stamping arrival time onto a delayed quote is how a
    #: fifteen-minute-old price starts looking current.
    quoted_at: datetime
    currency: str = "INR"

    @property
    def age_seconds(self) -> float:
        return max(0.0, (utc_now() - self.quoted_at).total_seconds())


@dataclass(frozen=True)
class QuoteFetchResult:
    """Quotes that arrived, and the symbols that did not.

    Failures are returned rather than logged, for the same reason the corporate
    action loader returns them: a missing quote and a quote of zero are
    different, and a caller that cannot tell them apart will eventually treat
    one as the other.
    """

    quotes: dict[str, Quote]
    failures: dict[str, str]

    @property
    def stalest_seconds(self) -> float | None:
        """Age of the oldest quote, or None when none arrived."""
        return max((q.age_seconds for q in self.quotes.values()), default=None)


def _to_decimal(value: object) -> Decimal | None:
    """Vendor numbers arrive as floats; money does not stay one (§14.1.2)."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _parse(row: dict[str, object]) -> tuple[str, Quote] | None:
    """One quote row, or None when it lacks a price or a timestamp.

    A row missing either is dropped rather than defaulted. A quote with an
    invented timestamp cannot be aged, and an unaged quote is indistinguishable
    from a fresh one on screen.
    """
    raw_symbol = row.get("symbol")
    price = _to_decimal(row.get("regularMarketPrice"))
    struck = row.get("regularMarketTime")
    if not isinstance(raw_symbol, str) or price is None or not isinstance(struck, int | float):
        return None

    symbol = raw_symbol.removesuffix(".NS")
    return symbol, Quote(
        symbol=symbol,
        price=price,
        quoted_at=datetime.fromtimestamp(float(struck), tz=UTC),
        currency=str(row.get("currency") or "INR"),
    )


def fetch_quotes(
    symbols: tuple[str, ...],
    client: httpx.Client | None = None,
    timeout: float = QUOTE_TIMEOUT,
) -> QuoteFetchResult:
    """Most recent traded price per NSE symbol, with its own timestamp.

    Args:
        symbols: Bare NSE tickers. The `.NS` suffix is added and stripped here
            so no caller has to know about Yahoo's naming.
        client: Injected for tests. A real client is created per call otherwise,
            because this runs a few times a minute at most.

    Returns:
        A `QuoteFetchResult`. Network failure is reported per batch rather than
        raised: a console that 500s because a quote vendor is down is less
        useful than one that shows last night's close and says so.
    """
    wanted = tuple(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not wanted:
        return QuoteFetchResult({}, {})

    quotes: dict[str, Quote] = {}
    failures: dict[str, str] = {}
    owned = client is None
    session = client or httpx.Client(
        timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (neutron)"}
    )

    def one(symbol: str) -> tuple[str, Quote | str]:
        try:
            response = session.get(QUOTE_URL.format(symbol=symbol), params=CHART_PARAMS)
            response.raise_for_status()
            results = response.json()["chart"]["result"]
            meta = results[0]["meta"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            return symbol, f"{type(exc).__name__}: {exc}"
        parsed = _parse(meta) if isinstance(meta, dict) else None
        return (symbol, parsed[1]) if parsed else (symbol, "no price in the response")

    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            for symbol, outcome in pool.map(one, wanted):
                if isinstance(outcome, Quote):
                    quotes[symbol] = outcome
                else:
                    failures[symbol] = outcome
    finally:
        if owned:
            session.close()

    return QuoteFetchResult(quotes=quotes, failures=failures)
