"""BSE bhavcopy loader — MASTER_PLAN §13.4, §M2.

**The parser is NSE's, unchanged.** Since SEBI's UDiFF standardisation both
exchanges publish the same column layout — `TradDt, ISIN, TckrSymb, SctySrs,
OpnPric, HghPric, LwPric, ClsPric, TtlTradgVol` — so `parse_bhavcopy` reads a
BSE file without knowing it is one. What differs is three things, and only
three, which is why this module is short:

    the URL              bseindia.com rather than nsearchives
    the series codes     A/B/T/X and friends rather than EQ
    the id prefix        BSE: rather than NSE:

**The series codes are the part that matters.** BSE groups equities by
settlement and surveillance history rather than by instrument type: A is the
large liquid set, B the ordinary one, T is trade-to-trade (no intraday netting),
and X/XT are the SME platform. Feeding NSE's `EQ` filter to a BSE file matches
nothing at all, which is exactly what happened the first time — a file that
parsed perfectly and yielded zero rows.

**T, XT and MT are included deliberately.** Trade-to-trade names must be
delivered rather than squared off intraday, which is a real constraint on how
they can be traded, not a reason to pretend they do not exist. Excluding them
would quietly reintroduce survivorship bias: a name is usually moved *into* T
group because it is in trouble.

**The two exchanges are separate venues, not one merged tape.** A name listed on
both has a BSE row and an NSE row with the same ISIN and different prices, and
the panel keeps them apart — `PanelStore(venue=...)` is keyed on exchange. Which
venue a strategy trades is a decision it makes; merging them here would make it
silently for everyone.
"""

from __future__ import annotations

from datetime import date

__all__ = [
    "BSE_SERIES",
    "T0_SUFFIX",
    "bse_instrument_id",
    "bse_udiff_url",
    "is_parallel_settlement",
]

#: Equity groups on BSE. Settlement and surveillance categories rather than
#: instrument types, which is why NSE's `EQ` matches none of them.
#:
#:     A    large, liquid, rolling settlement
#:     B    ordinary rolling settlement
#:     T    trade-to-trade: delivery compulsory, no intraday netting
#:     M    SME
#:     MT   SME under trade-to-trade
#:     X    SME/other platform
#:     XT   X group under trade-to-trade
#:
#: `F` is excluded: it is the debt segment, not equity.
BSE_SERIES: tuple[str, ...] = ("A", "B", "T", "M", "MT", "X", "XT")


def bse_udiff_url(session_date: date) -> str:
    """Daily bhavcopy URL in the current UDiFF layout.

    BSE serves this uncompressed, unlike NSE's zip — the loader already sniffs
    for a zip header, so neither caller has to care.
    """
    stamp = session_date.strftime("%Y%m%d")
    return (
        "https://www.bseindia.com/download/BhavCopy/Equity/"
        f"BhavCopy_BSE_CM_0_0_0_{stamp}_F_0000.CSV"
    )


def bse_instrument_id(isin: str, symbol: str) -> str:
    """Canonical internal id, prefixed by venue.

    The prefix is not decoration. A name dual-listed on both exchanges shares an
    ISIN and trades at two different prices, so `BSE:INE002A01018` and
    `NSE:INE002A01018` are genuinely different instruments to hold — different
    liquidity, different closes, different costs. Collapsing them onto one id
    would let a backtest fill on whichever venue happened to be cheaper that
    session, which is not a strategy anyone can trade.
    """
    return f"BSE:{isin}" if isin else f"BSE:{symbol.upper()}"


#: BSE marks the T+0 settlement copy of a scrip by suffixing its ticker.
#:
#: The optional T+0 rolling settlement segment opened on 2024-03-28, and from
#: that session the bhavcopy carries a second row for each participating name:
#: same ISIN, same close, a handful of shares traded. `INDHOTEL` printed
#: 167,607 shares that day and `INDHOTEL#` printed one.
T0_SUFFIX = "#"


def is_parallel_settlement(symbol: str) -> bool:
    """Whether this row is the T+0 copy of a security listed elsewhere in the file.

    **Same ISIN means same security.** The suffixed row is not another
    instrument; it is the same one settling on a different cycle, which is why
    including it puts two rows with one identity into a cross-section that must
    carry each name once. The panel store refuses that outright — correctly —
    and the backfill stopped dead at 2024-03-28 rather than quietly writing a
    doubled session.

    Dropping the T+0 copy rather than the ordinary one is deliberate: T+0 is a
    thin parallel market whose volume is a rounding error against the main
    book, so keeping it would misreport the liquidity of every name in it. The
    day this system trades T+0 it needs its own venue, not a shared identity.
    """
    return symbol.endswith(T0_SUFFIX)
