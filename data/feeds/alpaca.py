"""Alpaca US equity bar loader — MASTER_PLAN §13.4, §0.0.

The paper environment. Free account, free IEX market data, a real venue with
real fills — which is what makes paper results evidence about the live system
rather than a simulation of one.

Two correctness details, the same two that mattered for every other feed:

1. **Incomplete bars are dropped.** The most recent bar is still forming, and
   storing it means the backtester sees a partial bar as final. Its "close"
   keeps changing, which is look-ahead of the worst kind.

2. **`receive_time` carries a real publication lag**, never a copy of
   `event_time`. If they were equal a strategy could act on a bar's close at
   the instant of that close, which no live system can do (§7.6).

**On IEX versus SIP.** The free tier is IEX-only, which is roughly 2-3% of US
consolidated volume. Prices track the consolidated tape closely for liquid
names and diverge for illiquid ones, and IEX volume is *not* total market
volume. Any liquidity or participation calculation built on it will be far too
conservative — which is the safe direction, but know that it is happening.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

import httpx
import polars as pl

from core.clock import UTC, require_utc, utc_now
from core.events import Timeframe
from data.store.bars import BAR_SCHEMA

__all__ = ["MAX_BARS_PER_PAGE", "AlpacaBarLoader", "AlpacaError", "alpaca_instrument_id"]

DATA_API = "https://data.alpaca.markets/v2"
MAX_BARS_PER_PAGE = 10_000

#: Alpaca's timeframe vocabulary. Explicit so an upstream rename is a
#: compile-time concern rather than a silently wrong request.
_TIMEFRAMES: dict[Timeframe, str] = {
    Timeframe.M1: "1Min",
    Timeframe.M5: "5Min",
    Timeframe.M15: "15Min",
    Timeframe.M30: "30Min",
    Timeframe.H1: "1Hour",
    Timeframe.H4: "4Hour",
    Timeframe.D1: "1Day",
}


class AlpacaError(RuntimeError):
    """The venue returned something unusable. Never swallowed (§14.1.5)."""


def alpaca_instrument_id(symbol: str) -> str:
    """Canonical internal id for a US-listed symbol."""
    return f"US:{symbol.upper()}"


class AlpacaBarLoader:
    """Historical bar loader for US equities.

    Args:
        api_key: Alpaca key id. Free paper accounts have full data access.
        api_secret: Alpaca secret.
        client: Injected so tests never touch the network.
        publication_lag: Gap between a bar closing and this system being able to
            act on it.
        feed: "iex" on the free tier, "sip" on a paid one.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        client: httpx.Client | None = None,
        publication_lag: timedelta = timedelta(seconds=1),
        feed: str = "iex",
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self._client = client
        self._owns_client = client is None
        self.publication_lag = publication_lag
        self.feed = feed

    def __enter__(self) -> AlpacaBarLoader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Fetch complete bars in [start, end), normalised to BAR_SCHEMA.

        Raises:
            AlpacaError: on an unusable response.
            ValueError: if the window is inverted.
        """
        start_utc = require_utc(start)
        end_utc = require_utc(end) if end is not None else utc_now()
        if end_utc <= start_utc:
            raise ValueError(f"end {end_utc} must be after start {start_utc}")

        interval = timedelta(seconds=timeframe.seconds)
        # A bar is complete only once its whole span lies in the past.
        horizon = min(end_utc, utc_now() - interval)

        rows: list[dict[str, object]] = []
        page_token: str | None = None
        while True:
            payload = self._request(symbol, timeframe, start_utc, horizon, page_token)
            rows.extend(cast("list[dict[str, object]]", payload.get("bars") or []))
            page_token = cast("str | None", payload.get("next_page_token"))
            if not page_token:
                break

        return self._to_frame(rows, horizon)

    def _request(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        page_token: str | None,
    ) -> dict[str, object]:
        params: dict[str, str | int] = {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "timeframe": _TIMEFRAMES[timeframe],
            "limit": MAX_BARS_PER_PAGE,
            "feed": self.feed,
            # Split- and dividend-adjusted is Alpaca's own adjustment, and it
            # embeds future information the same way any back-adjusted series
            # does. Raw is requested deliberately; corporate actions are applied
            # to positions instead (data.corpactions).
            "adjustment": "raw",
        }
        if page_token:
            params["page_token"] = page_token

        response = self.client.get(
            f"{DATA_API}/stocks/{symbol.upper()}/bars",
            headers=self._headers(),
            params=params,
        )
        if response.status_code != httpx.codes.OK:
            raise AlpacaError(
                f"bars {symbol} {timeframe.value} returned "
                f"HTTP {response.status_code}: {response.text[:200]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise AlpacaError(f"unexpected payload type: {type(payload).__name__}")
        return cast("dict[str, object]", payload)

    def _to_frame(self, rows: list[dict[str, object]], horizon: datetime) -> pl.DataFrame:
        """Normalise Alpaca bars into BAR_SCHEMA.

        Alpaca stamps `t` with the bar's *open*. We convert to close-stamped,
        because a bar labelled with its open can be observed before it has
        finished forming (`core.events.Bar`).
        """
        if not rows:
            return pl.DataFrame(schema=BAR_SCHEMA)

        lag_us = int(self.publication_lag.total_seconds() * 1_000_000)
        frame = pl.DataFrame(
            {
                "open_time": [str(r["t"]) for r in rows],
                "open": [float(cast("float", r["o"])) for r in rows],
                "high": [float(cast("float", r["h"])) for r in rows],
                "low": [float(cast("float", r["l"])) for r in rows],
                "close": [float(cast("float", r["c"])) for r in rows],
                "volume": [float(cast("float", r["v"])) for r in rows],
                "trades": [int(cast("int", r.get("n", 0))) for r in rows],
            }
        )

        return (
            frame.with_columns(
                pl.col("open_time")
                .str.to_datetime(time_zone="UTC")
                .dt.cast_time_unit("us")
                .alias("bar_open")
            )
            # Alpaca returns the requested interval, so the close boundary is
            # the next bar's open. Derived per-row rather than assumed constant,
            # because a session boundary breaks a fixed offset.
            .with_columns(
                pl.col("bar_open").shift(-1).alias("next_open"),
            )
            .with_columns(
                pl.coalesce(
                    pl.col("next_open"),
                    pl.col("bar_open") + pl.duration(microseconds=lag_us),
                ).alias("event_time")
            )
            .with_columns(
                (pl.col("event_time") + pl.duration(microseconds=lag_us)).alias("receive_time")
            )
            .filter(pl.col("event_time") <= pl.lit(horizon).cast(pl.Datetime("us", "UTC")))
            .unique(subset=["event_time"], keep="last")
            .sort("event_time")
            .select(list(BAR_SCHEMA.keys()))
        )


def default_start() -> datetime:
    """Sensible backfill origin. Alpaca's free IEX history begins in 2016."""
    return datetime(2016, 1, 1, tzinfo=UTC)
