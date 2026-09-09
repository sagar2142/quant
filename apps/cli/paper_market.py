"""Turning a panel into the inputs one cycle needs — MASTER_PLAN §20.

Split from `apps.cli.paper` because they are different jobs. That module reads
arguments, loads and saves state, and decides what a halt means; this one only
answers "given these bars, what does the market look like right now" — no
arguments, no files, no clock beyond the panel's own timestamps.

The seam matters for the reason every seam here does: these three functions are
the ones a test wants to call directly, and reaching them through a CLI's
argument parsing is how a test ends up asserting on a `Namespace`.
"""

from __future__ import annotations

from datetime import datetime, time
from decimal import Decimal

import polars as pl

from core.clock import UTC, as_decision_time
from core.instruments import InstrumentId
from quant.strategies.base import MarketView

__all__ = ["ADV_WINDOW", "latest_marks", "latest_view", "trailing_adv"]

#: Trailing window for average daily traded value, matching the risk limits.
ADV_WINDOW = 20


def latest_view(history: pl.DataFrame, universe: tuple[InstrumentId, ...]) -> MarketView:
    """Everything observable as of the latest session's publication."""
    last = history["event_time"].max()
    assert isinstance(last, datetime)
    as_of = as_decision_time(datetime.combine(last.date(), time(23, 59), tzinfo=UTC))
    return MarketView(as_of=as_of, history=history, universe=universe)


def latest_marks(history: pl.DataFrame) -> dict[InstrumentId, Decimal]:
    latest = history.sort("event_time").group_by("instrument_id").agg(pl.col("close").last())
    return {
        InstrumentId(i): Decimal(str(c))
        for i, c in zip(latest["instrument_id"], latest["close"], strict=True)
    }


def trailing_adv(history: pl.DataFrame) -> dict[InstrumentId, Decimal]:
    """Mean traded value over the trailing window, per name."""
    sessions = history["event_time"].unique().sort().tail(ADV_WINDOW)
    window = history.filter(pl.col("event_time").is_in(sessions.implode()))
    value = (
        window.with_columns((pl.col("close") * pl.col("volume")).alias("traded"))
        .group_by("instrument_id")
        .agg(pl.col("traded").mean())
    )
    return {
        InstrumentId(i): Decimal(str(round(v, 2)))
        for i, v in zip(value["instrument_id"], value["traded"], strict=True)
        if v is not None and v > 0
    }
