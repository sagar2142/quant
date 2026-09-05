"""Ops console API — MASTER_PLAN §13.6, §26.

Serves the React console and the state it renders. Runs on the same process as
nothing else: the trading engine is a separate daemon, and **closing this API
must never affect trading** (§13.6). It reads state; it does not own it.

Two safety properties:

**Bind to 127.0.0.1, never 0.0.0.0** (§13.7). This surface holds order entry,
approvals and a kill switch. An internet-reachable trading console is a
catastrophic, unrecoverable hole; remote access is Tailscale's job, not
uvicorn's.

**The kill switch is the only mutating endpoint.** Everything else is a read.
Halting is the one action safe to expose over a network, because its failure
mode is "trading stopped when it need not have", which is recoverable.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from apps.api.accounts import build_accounts_router, write_settings
from apps.api.analytics import build_analytics_router
from apps.api.auth import ReadAccess, WriteAccess
from apps.api.book import DEFAULT_STATE_DIR, _latest_marks, build_book_router
from apps.api.broker_keys import build_broker_keys_router
from apps.api.candles import build_candles_router
from apps.api.journal import build_journal_router
from apps.api.lab import build_lab_router
from apps.api.options import build_options_router
from apps.api.research import build_research_router
from apps.api.snapshot import book_snapshot
from apps.api.trade import build_trade_router
from core.clock import UTC, utc_now
from core.config import settings
from ops.alerts import AlertRouter, ConsoleSink
from trading.risk.engine import RiskEngine

__all__ = ["create_app"]

#: Host headers this API answers to. A request arriving under any other name
#: is not from the local console, whatever address it reached.
#: Starlette strips the port before comparing, so these are port-independent
#: and do not need updating when the API moves.
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "testserver")

#: Staleness thresholds mirrored from §12.7 so the UI and the alerts agree.
STALE_WARN_SECONDS = 2.0
STALE_CRITICAL_SECONDS = 10.0

#: The paper loop runs once per trading session, so its staleness is measured
#: in hours, not the seconds a tick feed is judged by. Applying the tick
#: thresholds above to a daily batch would paint the feed red permanently and
#: teach the operator to ignore the colour.


#: The lake holds one bar per session, so a weekend alone is two days old and
#: healthy. Four days means a missed ingest; ten means the research on screen is
#: being computed on prices from a fortnight ago.
LAKE_WARN_SECONDS = 4 * 24 * 3600
LAKE_CRITICAL_SECONDS = 10 * 24 * 3600


class FeedStatus(BaseModel):
    name: str
    health: Literal["ok", "degraded", "down"]


def _worst(*ages: float | None) -> float | None:
    """The oldest of several staleness readings, or None if none is known."""
    known = [age for age in ages if age is not None]
    return max(known) if known else None


def _lake_staleness(venue: str = "NSE") -> float | None:
    """Seconds since the most recent session in one venue's panel.

    Per venue, because they are ingested separately and fail separately: NSE
    runs from 2019 and BSE only from 2024, so one being current says nothing
    about the other. A single light labelled for the whole lake would go green
    on NSE while BSE quietly stopped.
    """
    from apps.api.analytics import _panel  # noqa: PLC0415 - shares the cached panel

    try:
        latest = _panel(venue)["event_time"].max()
    except Exception:  # noqa: BLE001 - a monitoring surface must not 500 on data
        return None
    if not isinstance(latest, datetime):
        return None
    return max(0.0, float((utc_now() - latest).total_seconds()))


def _derivatives_staleness() -> float | None:
    """Seconds since the most recent derivatives session, or None if unread."""
    from data.store.derivatives import DerivativesStore  # noqa: PLC0415

    try:
        latest = DerivativesStore(settings.lake).latest_session()
    except Exception:  # noqa: BLE001 - as above
        return None
    if latest is None:
        return None
    age = utc_now() - datetime(latest.year, latest.month, latest.day, tzinfo=UTC)
    return max(0.0, float(age.total_seconds()))


def _health(
    staleness_seconds: float | None, warn: float, critical: float
) -> Literal["ok", "degraded", "down"]:
    """Health from age against a pair of thresholds.

    Unknown reports "down" rather than "ok": a feed nobody could read is not a
    healthy one, and green on an unstarted system is the same lie as a zero
    drawdown on a losing book.
    """
    if staleness_seconds is None:
        return "down"
    if staleness_seconds >= critical:
        return "down"
    if staleness_seconds >= warn:
        return "degraded"
    return "ok"


# `_cycle_health` and the CYCLE light are gone.
#
# They reported how long ago the paper-trading loop last completed a cycle.
# `apps.cli.paper` is a command you run, not a daemon that runs itself, so on
# an install where nobody had scheduled it the light read "down" permanently
# and dragged the bar's staleness figure to 101 hours with it. A health
# indicator that can only ever be red is worse than no indicator — it is the
# thing that teaches an operator to stop reading the health bar, which is the
# one part of the screen that has to be believed.
#
# What replaces it is the three data sources that actually exist and can
# actually go stale: the NSE panel, the BSE panel and the derivatives store.
# Paper trading itself is very much still here — `trading/paper/` holds the
# session and its state, and the book, snapshot and risk screens all read it.
# When the paper loop is scheduled the way `apps.cli.daily` now is, a cycle
# light becomes meaningful again and belongs back on this bar.


class VitalsResponse(BaseModel):
    """The bar that never scrolls away (§12.7).

    Every quantity is nullable, and that is the point: `None` means "not
    measured", which the console renders as an em dash. These fields were
    previously hardcoded to zero, so a book three days stale with a 0.15%
    drawdown displayed as a live, flat, healthy one.
    """

    feeds: list[FeedStatus]
    staleness_seconds: float | None
    day_pnl: Decimal | None
    day_pnl_pct: Decimal | None
    drawdown: Decimal | None
    ladder_rungs: list[Decimal]
    risk_utilisation: Decimal | None
    kill_engaged: bool
    #: False when no paper state exists. The console says "not started" rather
    #: than showing a flat book that was never traded.
    book_present: bool = False
    cycles: int = 0


class KillRequest(BaseModel):
    reason: str = Field(min_length=3, description="Why. Required for review.")
    operator: str = Field(min_length=1, description="Who. Required for attribution.")


class KillResponse(BaseModel):
    engaged: bool
    reason: str
    operator: str


def create_app(
    engine: RiskEngine | None = None,
    alerts: AlertRouter | None = None,
) -> FastAPI:
    """Build the console API.

    Args:
        engine: The risk engine whose kill switch this exposes. Injected so the
            API never constructs trading state of its own.
        alerts: Where kill-switch events are announced. The console is never the
            only place a halt is recorded (§12.7).
    """
    risk = engine or RiskEngine()
    router = alerts or AlertRouter([ConsoleSink()])

    app = FastAPI(
        title="Neutron ops console",
        version="0.1.0",
        docs_url="/docs",
    )

    # §13.7 binds this to 127.0.0.1, which is necessary and not sufficient. A
    # browser is already inside the loopback boundary: a page the operator
    # visits can re-resolve its own hostname to 127.0.0.1 after loading (DNS
    # rebinding), at which point requests to this API are same-origin and CORS
    # does not apply. Without a Host check, that page could reach /kill/release.
    #
    # Engaging a halt is fail-safe; releasing one is not, and this is the
    # switch §21 treats as the last line before real money. Rejecting unknown
    # Host headers closes the rebinding path, because the attacking page must
    # send its own hostname.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    # Analytics is read-only and shares its implementation with
    # `apps.cli.terminal`, so the console and the terminal cannot disagree
    # about what a security is.
    app.include_router(build_analytics_router())
    # Candles at any interval. Daily and weekly from the panel; intraday
    # from the broker, and never written back into the panel (3).
    app.include_router(build_candles_router())
    # Risk limits and the paper book. Read-only; the console's ops screens had
    # no source at all before this and rendered blank.
    app.include_router(build_book_router())
    # The fast research loop: score a signal without backtesting it.
    app.include_router(build_research_router())
    # Derivatives. Their own store and their own identity — a contract is
    # underlying, expiry, strike and right, not an ISIN.
    app.include_router(build_options_router())
    # The slow research loop: backtests and the twelve-check gauntlet, which
    # take minutes and so run as jobs rather than as requests. Until this
    # existed the console could display research but could start none.
    app.include_router(build_lab_router())
    # What execution actually cost, against what the model charged. The cost
    # model had never been checked against anything but the curve it produced.
    app.include_router(build_journal_router())
    # Operator accounts. Identification and preferences — the API token
    # is what decides whether the API answers at all (§13.7).
    app.include_router(build_accounts_router())
    # Per-account broker credentials, encrypted at rest. The router is
    # handed the settings writer so it need not know which database
    # holds accounts.
    app.include_router(build_broker_keys_router(write_settings))
    app.include_router(build_trade_router(risk))

    @app.get("/health", dependencies=[ReadAccess])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "environment": settings.env.value,
            "live_enabled": settings.live_enabled,
            "kill_engaged": risk.is_killed,
            "as_of": utc_now().isoformat(),
        }

    @app.get("/vitals", response_model=VitalsResponse, dependencies=[ReadAccess])
    def vitals() -> VitalsResponse:
        """The bar that never scrolls away (§12.7).

        Read from the paper state file the daemon writes each cycle — the same
        file the book and reconciliation read, so there is no second source to
        drift out of sync with. Anything not derivable from it is null, never
        zero.
        """
        snapshot = book_snapshot(DEFAULT_STATE_DIR, _latest_marks(None))
        lake_age = _lake_staleness()

        utilisation: Decimal | None = None
        if snapshot.gross_exposure is not None:
            limit = risk.limits.max_gross_exposure_pct
            if limit > 0:
                utilisation = snapshot.gross_exposure / limit

        bse_age = _lake_staleness("BSE")
        derivatives_age = _derivatives_staleness()

        return VitalsResponse(
            feeds=[
                FeedStatus(
                    name="nse",
                    health=_health(lake_age, LAKE_WARN_SECONDS, LAKE_CRITICAL_SECONDS),
                ),
                FeedStatus(
                    name="bse",
                    health=_health(bse_age, LAKE_WARN_SECONDS, LAKE_CRITICAL_SECONDS),
                ),
                FeedStatus(
                    name="nfo",
                    health=_health(derivatives_age, LAKE_WARN_SECONDS, LAKE_CRITICAL_SECONDS),
                ),
            ],
            # The worst of the three. One number on the bar must not be able to
            # hide a stopped feed behind a healthier one.
            staleness_seconds=_worst(lake_age, bse_age, derivatives_age),
            day_pnl=snapshot.day_pnl,
            day_pnl_pct=snapshot.day_pnl_pct,
            drawdown=snapshot.drawdown,
            ladder_rungs=[r.drawdown_pct for r in risk.ladder.rungs],
            risk_utilisation=utilisation,
            kill_engaged=risk.is_killed,
            book_present=snapshot.present,
            cycles=snapshot.cycles,
        )

    @app.post("/kill", response_model=KillResponse, dependencies=[WriteAccess])
    def engage_kill(request: KillRequest) -> KillResponse:
        """Halt all new orders.

        One of only two mutating endpoints, the other being its release. Both a
        reason and an operator are required: an unattributed halt with no stated
        cause cannot be reviewed afterwards, and the same constraint exists in
        the database and in the engine.
        """
        try:
            risk.engage_kill(request.reason, request.operator)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        router.kill_switch(engaged=True, by=request.operator, reason=request.reason)
        return KillResponse(engaged=True, reason=request.reason, operator=request.operator)

    @app.post("/kill/release", response_model=KillResponse, dependencies=[WriteAccess])
    def release_kill(request: KillRequest) -> KillResponse:
        """Resume trading. Manual only — there is deliberately no auto-release."""
        try:
            risk.release_kill(request.operator)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        router.kill_switch(engaged=False, by=request.operator, reason=request.reason)
        return KillResponse(engaged=False, reason=request.reason, operator=request.operator)

    return app


app = create_app()
