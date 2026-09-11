"""Quarterly results, and when they became public — MASTER_PLAN §1.1, §3.3, §9.

**Every signal in this system was a function of price.** Momentum, reversal,
volatility, turnover — all of it computed from the same bhavcopy. That is a
real constraint on what can be found, not a stylistic one: a book built only
from price is betting on the behaviour of prices, and the register is full of
rejected hypotheses that were all variations on the same input.

NSE's corporate-filings API publishes every quarterly result, and each filing
carries three things this system needs:

    `isin`          the identity it already keys on (§1.1)
    `exchdisstime`  when the exchange *disseminated* it — the receive_time
    `fromDate`/`toDate`  the period the numbers describe — the event_time

That third-and-second pairing is what makes fundamentals usable here at all. A
vendor fundamentals table gives you the quarter and the numbers; it does not
tell you the day the market learned them, so a backtest reading it positions on
a March quarter in April when the filing landed in July. The distortion is
enormous and invisible — it looks like skill.

**receive_time is the dissemination time, not the filing time.** The API gives
both, and they differ by up to a few minutes (`difference` carries the gap).
`filingDate` is when the company submitted; `exchdisstime` is when the exchange
published it and the market could act. The later of the two is the honest
answer, and is what `parse_filings` records.

**The context trap.** Ind-AS XBRL documents carry each fact several times under
different contexts, and — measured, not assumed — *the contexts declare the
same start and end dates*. In a Q3 filing both `OneD` and `FourD` claim
2024-10-01 to 2024-12-31, but `OneD` holds the quarter and `FourD` holds the
nine-month cumulative. Selecting a fact by its declared period therefore
returns the year-to-date figure roughly a third of the time, overstating
revenue by ~3x with nothing to indicate anything went wrong. Confirmed across
independent filings: TIMETECHNO 2.87, COFFEEDAY 2.89, SMARTLINK 2.78,
MAHLIFE 2.17 — every one the shape of a cumulative against its quarter.

A prefix alone is not enough either. `OneReportableSegmentRevenue01D` is a real
context in a real filing: it starts with `One`, and it holds a single business
segment's revenue. Taking the first `One*` fact encountered would record that
as the company's, in whatever documents happen to emit segments before totals.

So facts are selected **structurally** — a quarter context that carries no
dimension and spans a duration — and a document with no such context is refused
rather than falling back to whatever else is present. A wrong number here would
propagate into every factor built on it.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import polars as pl

from core.clock import UTC

# Re-exported: the document parser moved to `nse_xbrl`, and callers that have
# always imported it from here should not have to learn which file it is in.
from data.feeds.nse_xbrl import (
    FUNDAMENTAL_FACTS,
    QUARTER_CONTEXT,
    ResultsFormatError,
    XbrlFacts,
    parse_xbrl,
    xbrl_is_plausible,
)

__all__ = [
    "FILING_SCHEMA",
    "FUNDAMENTAL_FACTS",
    "QUARTER_CONTEXT",
    "ResultsFormatError",
    "XbrlFacts",
    "filings_url",
    "has_xbrl_document",
    "parse_filings",
    "parse_xbrl",
    "xbrl_document_expr",
    "xbrl_is_plausible",
]


_BASE = "https://www.nseindia.com/api/corporates-financial-results"

#: Keys the API response must carry. A layout change fails loudly rather than
#: writing a table of nulls that every factor then reads as "no earnings".
REQUIRED_KEYS = ("isin", "symbol", "fromDate", "toDate")

FILING_SCHEMA: dict[str, pl.DataType] = {
    "isin": pl.String(),
    "symbol": pl.String(),
    "company": pl.String(),
    #: Period the numbers describe.
    "period_start": pl.Date(),
    "period_end": pl.Date(),
    #: When the exchange disseminated it. The only column that may be compared
    #: against a decision time.
    "receive_time": pl.Datetime(time_unit="us", time_zone="UTC"),
    #: Consolidated and standalone are both filed and both kept. Collapsing
    #: them would silently mix a group's numbers with its parent's.
    "consolidated": pl.Boolean(),
    "audited": pl.Boolean(),
    "xbrl_url": pl.String(),
}

#: NSE serves `DD-Mon-YYYY`, sometimes with a time attached.
_DATE = "%d-%b-%Y"
_DATETIME = "%d-%b-%Y %H:%M:%S"
_DATETIME_NO_SECONDS = "%d-%b-%Y %H:%M"


def filings_url(from_date: str, to_date: str, period: str = "Quarterly") -> str:
    """The corporate-filings endpoint for one window.

    Args:
        from_date: Inclusive start, `DD-MM-YYYY`.
        to_date: Inclusive end, `DD-MM-YYYY`.
        period: `Quarterly` or `Annual`.

    Note:
        Without a window the endpoint returns only the current financial year.
        The window is what reaches the archive — measured back to FY2019-20,
        which is where the NSE panel starts, so the two line up.
    """
    return f"{_BASE}?index=equities&period={period}&from_date={from_date}&to_date={to_date}"


def _parse_date(value: str) -> datetime | None:
    for layout in (_DATETIME, _DATETIME_NO_SECONDS, _DATE):
        try:
            return datetime.strptime(value.strip(), layout).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _receive_time(row: dict[str, Any]) -> datetime | None:
    """When the market could have known.

    The later of dissemination and filing. They normally differ by seconds, but
    taking the earlier one would claim knowledge before the exchange published
    it, and this is the field every point-in-time read depends on.
    """
    candidates = [
        parsed
        for key in ("exchdisstime", "broadCastDate", "filingDate")
        if (raw := row.get(key)) and (parsed := _parse_date(str(raw))) is not None
    ]
    return max(candidates) if candidates else None


def parse_filings(payload: bytes) -> pl.DataFrame:
    """Turn the API response into filings.

    Raises:
        ResultsFormatError: if the payload is not the expected JSON.

    Rows without an ISIN or without a dissemination time are dropped: the first
    cannot be joined to anything this system holds (§1.1), and the second cannot
    be read point-in-time, which would make it worse than absent.
    """
    text = payload.decode("utf-8", errors="replace").lstrip()
    if text[:1] in ("<", ""):
        raise ResultsFormatError("expected JSON, got HTML or an empty body")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResultsFormatError(f"not valid JSON: {exc}") from exc

    rows = raw if isinstance(raw, list) else raw.get("data")
    if not isinstance(rows, list):
        raise ResultsFormatError("response holds no list of filings")
    if rows and not any(key in rows[0] for key in REQUIRED_KEYS):
        raise ResultsFormatError(f"filings are missing every one of {REQUIRED_KEYS}")

    built: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        isin = str(row.get("isin") or "").strip()
        start = _parse_date(str(row.get("fromDate") or ""))
        end = _parse_date(str(row.get("toDate") or ""))
        received = _receive_time(row)
        if not isin or start is None or end is None or received is None:
            continue
        built.append(
            {
                "isin": isin,
                "symbol": str(row.get("symbol") or "").strip(),
                "company": str(row.get("companyName") or "").strip(),
                "period_start": start.date(),
                "period_end": end.date(),
                "receive_time": received,
                # Absent means standalone: the API labels consolidated filings
                # explicitly and leaves the rest alone.
                "consolidated": "consolidat" in str(row.get("consolidated") or "").lower(),
                "audited": str(row.get("audited") or "").strip().lower() == "audited",
                "xbrl_url": str(row.get("xbrl") or "").strip(),
            }
        )

    if not built:
        return pl.DataFrame(schema=FILING_SCHEMA)
    return pl.DataFrame(built, schema=FILING_SCHEMA).sort(["receive_time", "isin"])


def has_xbrl_document(url: str) -> bool:
    """Whether this filing actually links to a document.

    NSE writes the *filename* as a dash when none was filed, producing
    ``.../corporate/xbrl/-`` — a well-formed URL, on the right host, that 404s.
    Checking the scheme passes it; checking the suffix is what catches it.

    `xbrl_document_expr` is the vectorised form of exactly this rule, and a
    test holds the two to the same answers so they cannot drift apart.
    """
    return url.startswith("http") and url.lower().endswith(".xml")


def xbrl_document_expr(column: str = "xbrl_url") -> pl.Expr:
    """`has_xbrl_document` as a Polars expression, for filtering a frame."""
    return pl.col(column).str.starts_with("http") & pl.col(column).str.to_lowercase().str.ends_with(
        ".xml"
    )
