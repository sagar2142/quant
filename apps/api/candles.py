"""Candles at any interval — MASTER_PLAN §12.6, §3.

**Two sources, and the boundary between them is the session.** Daily and weekly
candles come from the panel this system stores: free, offline, and complete
back to 2019 for NSE. Anything finer has to come from the broker, because the
lake holds bhavcopy and a bhavcopy is one row per session — there is no
intraday in it to slice.

**Intraday is never written into the panel.** Everything in `quant/` and
`engine/` reads rows carrying a `receive_time`, so a decision can only see what
had arrived by then (§3). Broker candles carry no such stamp, and joining them
to the panel would quietly destroy the point-in-time discipline the research
plane rests on. They are for looking at, not for backtesting.

**The limits are the broker's, and they are reported rather than hidden.**
Groww keeps three months of intraday and caps the span of a single request by
interval — seven days at one minute, a hundred and fifty at an hour. Asking for
more returns an error, so this refuses first, naming the limit it hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from core.clock import utc_now

__all__ = ["INTERVALS", "build_candles_router"]


@dataclass(frozen=True)
class Interval:
    """One candle size, and what the data behind it allows."""

    key: str
    label: str
    minutes: int
    #: Longest span a single request may cover, in days. None means no limit.
    max_span_days: int | None
    #: How far back the source holds this interval, in days. None means all.
    history_days: int | None
    #: Daily and coarser come from the panel; the rest need a broker.
    from_panel: bool


#: Intervals offered, in the order a chart should show them.
INTERVALS: tuple[Interval, ...] = (
    Interval("1m", "1m", 1, 7, 90, False),
    Interval("5m", "5m", 5, 15, 90, False),
    Interval("10m", "10m", 10, 30, 90, False),
    Interval("1h", "1H", 60, 150, 90, False),
    Interval("4h", "4H", 240, 365, 90, False),
    Interval("1d", "1D", 1440, None, None, True),
    Interval("1w", "1W", 10080, None, None, True),
)

BY_KEY = {interval.key: interval for interval in INTERVALS}


class IntervalInfo(BaseModel):
    key: str
    label: str
    minutes: int
    intraday: bool
    max_span_days: int | None
    history_days: int | None
    available: bool
    detail: str


class CandleResponse(BaseModel):
    symbol: str
    venue: str
    interval: str
    source: str
    dates: list[str]
    open: list[float]
    high: list[float]
    low: list[float]
    close: list[float]
    volume: list[float]


def _broker_available() -> tuple[bool, str]:
    """Whether intraday can be fetched at all, and why not when it cannot."""
    from apps.api.trade import live_gates  # noqa: PLC0415 - avoids an import cycle
    from trading.risk.engine import RiskEngine  # noqa: PLC0415

    blocked = [gate for gate in live_gates(RiskEngine()) if not gate.ready]
    if not blocked:
        return True, "available"
    return False, f"needs a broker connection ({blocked[0].name})"


def _from_panel(symbol: str, venue: str, sessions: int, interval: Interval) -> CandleResponse:
    """Daily and weekly, from the stored panel."""
    import polars as pl  # noqa: PLC0415

    from apps.api.analytics import _panel, _venue, _windowed  # noqa: PLC0415
    from apps.cli.terminal import load_actions, series_for  # noqa: PLC0415

    history = _windowed(_panel(_venue(venue)), sessions)
    name = symbol.upper()
    rows = series_for(history, name, load_actions(history, [name]))
    if rows.is_empty():
        raise HTTPException(status_code=404, detail=f"{name} is not in the {venue} panel")

    if interval.key == "1w":
        # Weekly bars are built from daily ones rather than fetched: the open
        # of the week, the extremes within it, the last close, the summed
        # volume. Grouped on the week start so a partial final week is still a
        # bar rather than being dropped.
        rows = (
            rows.sort("event_time")
            .group_by_dynamic("event_time", every="1w")
            .agg(
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            )
        )

    return CandleResponse(
        symbol=name,
        venue=venue.upper(),
        interval=interval.key,
        source="panel",
        dates=[d.date().isoformat() for d in rows["event_time"].to_list()],
        open=[float(v) for v in rows["open"].to_list()],
        high=[float(v) for v in rows["high"].to_list()],
        low=[float(v) for v in rows["low"].to_list()],
        close=[float(v) for v in rows["close"].to_list()],
        volume=[float(v) for v in rows["volume"].to_list()],
    )


def _from_broker(symbol: str, venue: str, sessions: int, interval: Interval) -> CandleResponse:
    """Intraday, from the broker."""
    usable, reason = _broker_available()
    if not usable:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{interval.label} candles are intraday, and the stored panel holds daily bars "
                f"only. They come from the broker, which is not connected: {reason}."
            ),
        )

    span_days = sessions or (interval.max_span_days or 30)
    if interval.max_span_days and span_days > interval.max_span_days:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{interval.label} candles can be fetched {interval.max_span_days} days at a "
                f"time; {span_days} were asked for."
            ),
        )
    if interval.history_days and span_days > interval.history_days:
        raise HTTPException(
            status_code=422,
            detail=f"{interval.label} history goes back {interval.history_days} days.",
        )

    end = utc_now()
    start = end - timedelta(days=span_days)

    from apps.api.trade import _broker  # noqa: PLC0415

    broker = _broker()
    rows = broker.candles(  # type: ignore[attr-defined]
        symbol.upper(),
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        interval.minutes,
    )

    return CandleResponse(
        symbol=symbol.upper(),
        venue=venue.upper(),
        interval=interval.key,
        source="broker",
        dates=[datetime.fromtimestamp(row[0], tz=utc_now().tzinfo).isoformat() for row in rows],
        open=[row[1] for row in rows],
        high=[row[2] for row in rows],
        low=[row[3] for row in rows],
        close=[row[4] for row in rows],
        volume=[row[5] for row in rows],
    )


def build_candles_router() -> APIRouter:
    """Interval metadata, and candles at any of them."""
    router = APIRouter(tags=["analytics"])

    @router.get("/intervals", response_model=list[IntervalInfo], dependencies=[ReadAccess])
    def intervals() -> list[IntervalInfo]:
        """Which candle sizes exist, and which are usable right now.

        Reported rather than assumed by the console: intraday depends on a
        broker connection, and a chart offering a 1-minute button that silently
        returns nothing is worse than one that says why it is unavailable.
        """
        usable, reason = _broker_available()
        return [
            IntervalInfo(
                key=interval.key,
                label=interval.label,
                minutes=interval.minutes,
                intraday=not interval.from_panel,
                max_span_days=interval.max_span_days,
                history_days=interval.history_days,
                available=interval.from_panel or usable,
                detail=(
                    "from the stored panel"
                    if interval.from_panel
                    else (f"{interval.history_days} days of history" if usable else reason)
                ),
            )
            for interval in INTERVALS
        ]

    @router.get(
        "/security/{symbol}/candles", response_model=CandleResponse, dependencies=[ReadAccess]
    )
    def candles(
        symbol: str,
        interval: str = Query("1d"),
        sessions: int = Query(0, ge=0),
        venue: str = Query("NSE"),
    ) -> CandleResponse:
        """Candles for one security at one interval."""
        chosen = BY_KEY.get(interval.lower())
        if chosen is None:
            raise HTTPException(
                status_code=422,
                detail=f"unknown interval {interval!r}; expected one of {list(BY_KEY)}",
            )

        if chosen.from_panel:
            return _from_panel(symbol, venue, sessions, chosen)
        return _from_broker(symbol, venue, sessions, chosen)

    return router
