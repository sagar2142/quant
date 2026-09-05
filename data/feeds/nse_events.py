"""What companies have said they will do, and when — MASTER_PLAN §1.1, §3.3.

`data.feeds.nse_results` records what a company *has* reported. This records
what it has *announced it will* report: NSE publishes the board-meeting
calendar, and a board meeting called to consider financial results is an
earnings date.

**The use is positional, not predictive.** Nobody here is forecasting the
number. The question a calendar answers is whether a position is about to be
held through an event that reprices it — which is a risk question, and one the
book could not previously ask. It pairs with `ops.watch`: a rule can fire
because a held name reports in three days.

**Symbols, not ISINs — and that is a real cost.** This endpoint carries no
`isin`, so an event has to be resolved through the panel, and a symbol that is
not trading in the recent panel cannot be resolved at all. Measured on a live
calendar: 27 of 34 resolved, the remainder SME-board names the EQ bhavcopy does
not carry. Unresolved events are *reported*, never dropped in silence — the
same rule as an unevaluable alert, and for the same reason.

**No archive.** NSE serves the calendar ahead, not behind, so this is a
present-tense feed like the sector classification. It says what is coming; it
cannot tell a backtest what was coming on some past date, and nothing here
pretends otherwise. The historical earnings dates already exist, properly
timestamped, in the results store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl

from core.clock import UTC

__all__ = [
    "EVENT_SCHEMA",
    "RESULTS_PURPOSE",
    "EventFormatError",
    "event_calendar_url",
    "is_results_event",
    "parse_events",
]

EVENT_CALENDAR_URL = "https://www.nseindia.com/api/event-calendar"

#: The substring NSE uses when a board meeting is called to consider results.
#:
#: A meeting's stated purpose is free text and often compound — "Financial
#: Results/Other business matters" — so this is a containment test rather than
#: an equality one. Matching exactly would silently miss every compound
#: purpose, which is most of the ones that matter.
RESULTS_PURPOSE = "financial results"

#: Keys the response must carry, so a layout change fails loudly rather than
#: producing an empty calendar that reads as "nobody is reporting".
REQUIRED_KEYS = ("symbol", "date", "purpose")

EVENT_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String(),
    "company": pl.String(),
    #: The announced date of the meeting.
    "event_date": pl.Date(),
    "purpose": pl.String(),
    "description": pl.String(),
    #: Whether the stated purpose includes considering financial results.
    "is_results": pl.Boolean(),
    #: When this calendar was observed. NSE serves no archive, so as with the
    #: sector classification the fetch date is the only provenance there is.
    "observed_at": pl.Date(),
}

_DATE_LAYOUTS = ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d")


class EventFormatError(ValueError):
    """The payload was not the event calendar. Never guessed at."""


@dataclass(frozen=True)
class Resolution:
    """Events matched to instruments, and the ones that could not be."""

    rows: pl.DataFrame
    #: Symbols with no instrument in the panel. Reported, never dropped
    #: silently — an event nobody can join to a position is still an event, and
    #: pretending the calendar is complete is how it stops being trusted.
    unresolved: tuple[str, ...]

    @property
    def coverage(self) -> float:
        total = self.rows.height + len(self.unresolved)
        return self.rows.height / total if total else 0.0


def event_calendar_url() -> str:
    """The upcoming-events endpoint.

    Takes no date: NSE serves what is ahead, and there is no archive to ask for.
    """
    return EVENT_CALENDAR_URL


def is_results_event(purpose: str, description: str = "") -> bool:
    """Whether this meeting was called to consider financial results."""
    haystack = f"{purpose} {description}".lower()
    return RESULTS_PURPOSE in haystack


def _parse_date(value: str) -> date | None:
    for layout in _DATE_LAYOUTS:
        try:
            return datetime.strptime(value.strip(), layout).replace(tzinfo=UTC).date()
        except ValueError:
            continue
    return None


def parse_events(payload: bytes, observed: date) -> pl.DataFrame:
    """Turn the calendar response into dated events.

    Raises:
        EventFormatError: if the payload is not the expected JSON.

    Rows without a symbol or a readable date are dropped: an event with no date
    is not a calendar entry, and one with no symbol cannot be resolved to
    anything.
    """
    text = payload.decode("utf-8", errors="replace").lstrip()
    if text[:1] in ("<", ""):
        raise EventFormatError("expected JSON, got HTML or an empty body")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EventFormatError(f"not valid JSON: {exc}") from exc

    rows = raw if isinstance(raw, list) else raw.get("data")
    if not isinstance(rows, list):
        raise EventFormatError("response holds no list of events")
    if rows and not any(key in rows[0] for key in REQUIRED_KEYS):
        raise EventFormatError(f"events are missing every one of {REQUIRED_KEYS}")

    built = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        when = _parse_date(str(row.get("date") or ""))
        if not symbol or when is None:
            continue
        purpose = str(row.get("purpose") or "").strip()
        description = str(row.get("bm_desc") or "").strip()
        built.append(
            {
                "symbol": symbol,
                "company": str(row.get("company") or "").strip(),
                "event_date": when,
                "purpose": purpose,
                "description": description,
                "is_results": is_results_event(purpose, description),
                "observed_at": observed,
            }
        )

    if not built:
        return pl.DataFrame(schema=EVENT_SCHEMA)
    return pl.DataFrame(built, schema=EVENT_SCHEMA).sort(["event_date", "symbol"])


def resolve(events: pl.DataFrame, symbols: dict[str, str]) -> Resolution:
    """Attach instrument ids, keeping the symbols that could not be matched.

    Args:
        events: Parsed events.
        symbols: Trading symbol -> instrument id, from the panel.

    The join is the weak point of this feed and is therefore explicit. NSE's
    calendar carries no ISIN, so a symbol absent from the recent panel — an
    SME-board listing, a suspended name — cannot be tied to an instrument. Those
    come back in `unresolved` rather than vanishing from the count.
    """
    if events.is_empty():
        return Resolution(
            rows=events.with_columns(pl.lit(None, pl.String).alias("instrument_id")), unresolved=()
        )
    mapped = events.with_columns(
        pl.col("symbol").replace_strict(symbols, default=None).alias("instrument_id")
    )
    resolved = mapped.filter(pl.col("instrument_id").is_not_null())
    missing = mapped.filter(pl.col("instrument_id").is_null())["symbol"].unique().sort().to_list()
    return Resolution(rows=resolved, unresolved=tuple(missing))
