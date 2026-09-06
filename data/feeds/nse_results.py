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
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import polars as pl
from lxml import etree

from core.clock import UTC

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

logger = logging.getLogger(__name__)

_BASE = "https://www.nseindia.com/api/corporates-financial-results"

#: Context id prefix holding the reporting period itself.
#:
#: The Ind-AS taxonomy numbers its periods: `One` is the current quarter, `Four`
#: the year-to-date. They are *not* distinguishable by their declared dates —
#: see the module docstring — so this prefix is the only reliable selector.
QUARTER_CONTEXT = "One"

#: The cumulative context, named so the guard below can say what it rejected.
CUMULATIVE_CONTEXT = "Four"

#: Facts worth extracting, mapped to the column they become.
#:
#: Deliberately small. Every one of these appears in the standard Ind-AS
#: statement of profit and loss, so a name missing one is a real gap rather
#: than a taxonomy variant, and there is no temptation to infer it.
FUNDAMENTAL_FACTS: dict[str, str] = {
    "RevenueFromOperations": "revenue",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "ProfitBeforeTax": "profit_before_tax",
    "ProfitLossForPeriod": "net_profit",
    "BasicEarningsLossPerShare": "eps_basic",
    "DilutedEarningsLossPerShare": "eps_diluted",
}

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

#: A revenue above this is a parsing failure, not a company. India's largest
#: listed revenue is ~10 trillion rupees a year; 100 trillion in one quarter is
#: two orders of magnitude beyond anything real and indicates a units error.
IMPLAUSIBLE_ABOVE = Decimal("1e14")


class ResultsFormatError(ValueError):
    """The payload was not what this feed parses. Never guessed at."""


@dataclass(frozen=True)
class XbrlFacts:
    """One filing's numbers, for the reporting period only."""

    #: Column name -> value, for whatever was present.
    values: dict[str, Decimal]
    #: Context id the facts were taken from, kept for provenance.
    context: str

    def __bool__(self) -> bool:
        return bool(self.values)


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


def _local_name(tag: object) -> str:
    """Strip the namespace. Ind-AS documents use several and vary by filer."""
    text = str(tag)
    return text.rsplit("}", 1)[-1]


def _to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.strip().replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _statement_contexts(root: etree._Element) -> set[str]:
    """Quarter contexts that carry statement totals rather than breakdowns.

    Two filters, both structural rather than positional:

    *Undimensioned.* A context with a dimension reports one slice — a segment,
    a product line, an expense category. `OneReportableSegmentRevenue01D` is a
    real context in a real filing, it starts with `One`, and it holds one
    business segment's revenue. Selecting the first `One*` fact encountered
    would record that as the company's, silently, in whatever documents happen
    to emit segments before totals.

    *Duration, not instant.* Every fact here is a flow over the quarter.
    Instant contexts (`OneI`) carry balance-sheet positions and would be a
    different quantity wearing the same tag.
    """
    usable: set[str] = set()
    for element in root.iter():
        if _local_name(element.tag) != "context":
            continue
        context_id = str(element.get("id") or "")
        if not context_id.startswith(QUARTER_CONTEXT):
            continue
        names = {_local_name(node.tag) for node in element.iter()}
        if "explicitMember" in names or "typedMember" in names:
            continue
        if "endDate" not in names or "startDate" not in names:
            continue
        usable.add(context_id)
    return usable


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


def parse_xbrl(payload: bytes) -> XbrlFacts:
    """Extract the reporting period's figures from an Ind-AS XBRL document.

    Raises:
        ResultsFormatError: if the document does not parse, or carries no
            quarter context.

    **Facts are selected by context id prefix, never by declared period.** The
    quarter and the year-to-date contexts carry identical start and end dates,
    so a date-based selection silently returns cumulative figures for a name
    that filed them in a different order. See the module docstring.

    A document holding only a cumulative context is refused. Its numbers are
    real, but they are not the quarter's, and a table mixing the two is worse
    than one missing the name entirely.
    """
    # `resolve_entities=False` and `no_network=True`: this parses a document
    # fetched over the network, and an entity that reads a local file or
    # re-fetches a URL is a real capability to hand it.
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        root = etree.fromstring(payload, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise ResultsFormatError(f"not parseable XML: {exc}") from exc

    usable = _statement_contexts(root)
    values: dict[str, Decimal] = {}
    contexts: set[str] = set()
    skipped: set[str] = set()
    chosen = ""

    for element in root.iter():
        name = _local_name(element.tag)
        if name not in FUNDAMENTAL_FACTS:
            continue
        context = str(element.get("contextRef") or "")
        contexts.add(context)
        if context not in usable:
            continue
        # Pinned to the first usable context that yields a fact. Measured on 25
        # real filings, every document has exactly one, so this costs nothing —
        # but taking facts from two while reporting one context would make the
        # provenance a claim rather than a record.
        if chosen and context != chosen:
            skipped.add(context)
            continue
        value = _to_decimal(element.text or "")
        if value is None:
            continue
        chosen = chosen or context
        values.setdefault(FUNDAMENTAL_FACTS[name], value)

    if not values:
        cumulative = sorted(c for c in contexts if c.startswith(CUMULATIVE_CONTEXT))
        if cumulative:
            raise ResultsFormatError(
                f"no usable {QUARTER_CONTEXT}* context; document holds only cumulative "
                f"figures ({', '.join(cumulative)}), which are not this quarter's"
            )
        raise ResultsFormatError("document carries none of the expected facts")
    if skipped:
        # Never seen in real filings. If it starts happening the taxonomy has
        # changed shape and the selection needs revisiting, so it is loud.
        logger.warning(
            "%s usable contexts in one document; kept %s, ignored %s",
            len(skipped) + 1,
            chosen,
            ", ".join(sorted(skipped)),
        )
    return XbrlFacts(values=values, context=chosen)


def xbrl_is_plausible(facts: XbrlFacts) -> str:
    """Why these numbers should be refused, or an empty string.

    A units error in a filing does not look like an error downstream — it looks
    like a company that grew ten-thousandfold, which is exactly the kind of
    outlier a ranking factor puts straight at the top.
    """
    revenue = facts.values.get("revenue")
    if revenue is not None and abs(revenue) > IMPLAUSIBLE_ABOVE:
        return f"revenue {revenue:.0f} is beyond any real company; likely a units error"
    if revenue is not None and revenue < 0:
        return f"negative revenue {revenue:.0f}"
    return ""
