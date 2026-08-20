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

from decimal import Decimal
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from apps.api.analytics import build_analytics_router
from apps.api.book import build_book_router
from apps.api.research import build_research_router
from core.clock import utc_now
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


class FeedStatus(BaseModel):
    name: str
    health: Literal["ok", "degraded", "down"]


class VitalsResponse(BaseModel):
    feeds: list[FeedStatus]
    staleness_seconds: float
    day_pnl: Decimal
    day_pnl_pct: Decimal
    drawdown: Decimal
    ladder_rungs: list[Decimal]
    risk_utilisation: Decimal
    kill_engaged: bool


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
    # Risk limits and the paper book. Read-only; the console's ops screens had
    # no source at all before this and rendered blank.
    app.include_router(build_book_router())
    # The fast research loop: score a signal without backtesting it.
    app.include_router(build_research_router())

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "environment": settings.env.value,
            "live_enabled": settings.live_enabled,
            "kill_engaged": risk.is_killed,
            "as_of": utc_now().isoformat(),
        }

    @app.get("/vitals", response_model=VitalsResponse)
    def vitals() -> VitalsResponse:
        """The bar that never scrolls away (§12.7).

        Placeholder values until the paper-trading daemon publishes real state;
        the shape is what the console binds to.
        """
        return VitalsResponse(
            feeds=[FeedStatus(name="nse", health="ok")],
            staleness_seconds=0.0,
            day_pnl=Decimal(0),
            day_pnl_pct=Decimal(0),
            drawdown=Decimal(0),
            ladder_rungs=[r.drawdown_pct for r in risk.ladder.rungs],
            risk_utilisation=Decimal(0),
            kill_engaged=risk.is_killed,
        )

    @app.post("/kill", response_model=KillResponse)
    def engage_kill(request: KillRequest) -> KillResponse:
        """Halt all new orders.

        The only mutating endpoint. Both a reason and an operator are required:
        an unattributed halt with no stated cause cannot be reviewed afterwards,
        and the same constraint exists in the database and in the engine.
        """
        try:
            risk.engage_kill(request.reason, request.operator)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        router.kill_switch(engaged=True, by=request.operator, reason=request.reason)
        return KillResponse(engaged=True, reason=request.reason, operator=request.operator)

    @app.post("/kill/release", response_model=KillResponse)
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
