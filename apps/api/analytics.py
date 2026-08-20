"""Analytics endpoints — MASTER_PLAN §12.6.

Serves what `apps.cli.terminal` prints, as JSON, so the console and the
terminal read the same numbers from the same code. A second implementation of
"what is this security" would eventually disagree with the first, and the one
you were looking at would be the wrong one.

**Prices are back-adjusted here**, exactly as in the terminal and for the same
reason: the panel stores raw closes because the backtester applies corporate
actions to *positions*, and a 1:1 bonus read from raw closes is a -50% day
(§9). The panel itself is never mutated.

**The panel is loaded once and cached.** It is 3.3M rows and immutable between
ingests; re-reading it per request would make every screen slow enough to stop
being a terminal.
"""

from __future__ import annotations

from functools import lru_cache

import polars as pl
from fastapi import APIRouter, HTTPException, Query

from apps.api.auth import ReadAccess
from apps.api.schemas import (
    CrossSectionResponse,
    HorizonReturn,
    NameRow,
    ScreenResponse,
    ScreenRowResponse,
    SecurityResponse,
)
from apps.cli.terminal import aligned_returns, load_actions, series_for
from core.clock import as_decision_time, utc_now
from core.config import settings
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from quant.analytics.crosssection import analyse_cross_section
from quant.analytics.screener import ScreenCriteria, SortKey, screen_universe
from quant.analytics.security import profile_security

__all__ = ["build_analytics_router"]

#: Cap on symbols per cross-section request. The correlation work is O(n^2) and
#: a browser cannot read a 200-name matrix anyway.
MAX_SYMBOLS = 40

#: Rows returned by the symbol search. Enough to pick from, few enough to scan.
SEARCH_LIMIT = 60

#: Candidates named in a 404. Enough to spot the one you meant, few enough to
#: read in a single line.
SUGGESTIONS = 8

#: A cross-section is a comparison; one name is a security screen.
MIN_CROSS_SECTION = 2

#: Trailing window used to rank the symbol search by liquidity.
LIQUIDITY_WINDOW = 60


@lru_cache(maxsize=1)
def _panel() -> pl.DataFrame:
    """The whole panel, read once.

    Cached because it is immutable between ingests and large enough that a
    per-request read would be felt on every screen. A restart picks up new
    sessions, which is the right granularity for a daily-frequency system.
    """
    store = PanelStore(settings.lake, venue="NSE")
    return store.view(as_of=as_decision_time(utc_now()))


def _not_found(history: pl.DataFrame, name: str) -> str:
    """A 404 that names the near misses.

    "TATA is not in the panel" is true and useless when TATASTEEL, TATAPOWER
    and eight others are. An error that leaves the reader guessing the ticker
    is a dead end; one that lists candidates is a next step.
    """
    matches = (
        history.filter(pl.col("symbol").str.starts_with(name))["symbol"].unique().sort().to_list()
    )
    if not matches:
        # Fall back to a substring search: the typo may be a prefix, not a stem.
        matches = (
            history.filter(pl.col("symbol").str.contains(name, literal=True))["symbol"]
            .unique()
            .sort()
            .to_list()
        )
    if matches:
        listed = ", ".join(matches[:SUGGESTIONS])
        more = f" (+{len(matches) - SUGGESTIONS} more)" if len(matches) > SUGGESTIONS else ""
        return f"{name} is not a ticker. Did you mean: {listed}{more}"
    return f"{name} is not in the panel"


def _windowed(history: pl.DataFrame, sessions: int) -> pl.DataFrame:
    if sessions <= 0:
        return history
    recent = history["event_time"].unique().sort().tail(sessions)
    return history.filter(pl.col("event_time").is_in(recent.implode()))


def _register_search(router: APIRouter) -> None:
    @router.get("/symbols", dependencies=[ReadAccess])
    def symbols(q: str = Query("", max_length=32)) -> list[dict[str, object]]:
        """Symbol search, ranked by liquidity.

        Ranked rather than alphabetical: the name you want is almost always one
        you can actually trade, and an alphabetical list buries it under
        illiquid tickers sharing a prefix.
        """
        try:
            history = _panel()
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        recent = _windowed(history, LIQUIDITY_WINDOW)
        ranked = (
            recent.with_columns((pl.col("close") * pl.col("volume")).alias("traded"))
            .group_by("symbol")
            .agg(pl.col("traded").median().alias("adv"), pl.len().alias("bars"))
            .sort("adv", descending=True)
        )
        if q:
            ranked = ranked.filter(pl.col("symbol").str.starts_with(q.upper()))
        return [
            {"symbol": r["symbol"], "adv": float(r["adv"] or 0.0), "bars": int(r["bars"])}
            for r in ranked.head(SEARCH_LIMIT).to_dicts()
        ]


def _register_security(router: APIRouter) -> None:
    @router.get("/security/{symbol}", response_model=SecurityResponse, dependencies=[ReadAccess])
    def security(symbol: str, sessions: int = Query(0, ge=0)) -> SecurityResponse:
        try:
            history = _windowed(_panel(), sessions)
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        name = symbol.upper()
        actions = load_actions(history, [name])
        rows = series_for(history, name, actions)
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=_not_found(history, name))

        try:
            p = profile_security(name, rows["close"].to_list(), rows["volume"].to_list())
        except ValueError as exc:
            # Too little history is a 422, not a 500: the request was
            # well-formed and the answer is "not enough data", which the
            # console should show rather than swallow.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SecurityResponse(
            symbol=p.symbol,
            observations=p.observations,
            last_close=p.last_close,
            horizons=[HorizonReturn(label=k, value=v) for k, v in p.horizon_returns.items()],
            cagr=p.cagr,
            high_52w=p.high_52w,
            low_52w=p.low_52w,
            off_high=p.off_high,
            annual_volatility=p.annual_volatility,
            max_drawdown=p.max_drawdown,
            current_drawdown=p.current_drawdown,
            adv_value=p.adv_value,
            sharpe=p.sharpe,
            sortino=p.sortino,
            calmar=p.calmar,
            hit_rate=p.hit_rate,
            skewness=p.skewness,
            kurtosis=p.kurtosis,
            var_5=p.var_5,
            cvar_5=p.cvar_5,
            tail_ratio=p.tail_ratio,
            verdict=p.stationarity.verdict.value,
            adf_pvalue=p.stationarity.adf_pvalue,
            kpss_pvalue=p.stationarity.kpss_pvalue,
            hurst=p.stationarity.hurst,
            tradable_as_mean_reversion=p.stationarity.tradable_as_mean_reversion,
            autocorrelation={str(k): v for k, v in p.autocorrelation.items()},
            realised_vol=p.realised_vol.annualised,
            ewma_vol=p.ewma_vol.annualised,
            vol_regime=p.vol_regime,
            is_implausible=p.is_implausible,
            fat_left_tail=p.fat_left_tail,
        )

    @router.get("/security/{symbol}/series", dependencies=[ReadAccess])
    def series(symbol: str, sessions: int = Query(0, ge=0)) -> dict[str, list[object]]:
        """Back-adjusted close series, for charting."""
        history = _windowed(_panel(), sessions)
        name = symbol.upper()
        rows = series_for(history, name, load_actions(history, [name]))
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=_not_found(history, name))
        return {
            "dates": [d.date().isoformat() for d in rows["event_time"].to_list()],
            "closes": [float(c) for c in rows["close"].to_list()],
        }


def _register_cross_section(router: APIRouter) -> None:
    @router.get("/crosssection", response_model=CrossSectionResponse, dependencies=[ReadAccess])
    def crosssection(
        symbols: str = Query(..., description="Comma-separated symbols"),
        sessions: int = Query(750, ge=0),
    ) -> CrossSectionResponse:
        wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()][:MAX_SYMBOLS]
        if len(wanted) < MIN_CROSS_SECTION:
            raise HTTPException(status_code=422, detail="a cross-section needs 2+ symbols")

        history = _windowed(_panel(), sessions)
        actions = load_actions(history, wanted)
        kept, matrix = aligned_returns(history, wanted, actions)
        if not matrix.size:
            raise HTTPException(status_code=404, detail="none of those symbols are in the panel")

        try:
            section = analyse_cross_section(kept, matrix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return CrossSectionResponse(
            names=[
                NameRow(
                    symbol=n.symbol,
                    total_return=n.total_return,
                    annual_volatility=n.annual_volatility,
                    sharpe=n.sharpe,
                    beta=n.beta,
                    correlation_to_market=n.correlation_to_market,
                    weight_hrp=n.weight_hrp,
                    weight_erc=n.weight_erc,
                    cluster=n.cluster,
                )
                for n in section.ranked_by("total_return")
            ],
            sessions=section.sessions,
            mean_correlation=section.mean_correlation,
            clusters=len(section.clusters),
            effective_bets=section.effective_bets,
            diversification_ratio=section.diversification_ratio,
            condition_number=section.condition_number,
            shrinkage=section.shrinkage,
            market_return=section.market_return,
            market_volatility=section.market_volatility,
            is_ill_conditioned=section.is_ill_conditioned,
            concentration_warning=section.concentration_warning,
            correlation_labels=kept,
            correlation=[[float(v) for v in row] for row in section.correlation],
        )


def _register_screen(router: APIRouter) -> None:
    @router.get("/screen", response_model=ScreenResponse, dependencies=[ReadAccess])
    def screen(
        sort: str = Query("liquidity"),
        limit: int = Query(25, ge=1, le=200),
        window: int = Query(250, ge=60),
        min_adv: float = Query(1e7, ge=0),
        stationary_only: bool = Query(default=False),
    ) -> ScreenResponse:
        """Which names, rather than what is this name.

        Two-stage by construction: the vectorised filter runs over every symbol
        in a fraction of a second, and only the shortlist pays for ADF, KPSS
        and Hurst.
        """
        try:
            key = SortKey(sort)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown sort {sort!r}") from exc

        result = screen_universe(
            _panel(),
            ScreenCriteria(
                window=window,
                min_adv=min_adv,
                sort_by=key,
                limit=limit,
                stationary_only=stationary_only,
            ),
        )
        rows = []
        for row in result.rows:
            profile = row.profile
            if profile is None:
                continue
            rows.append(
                ScreenRowResponse(
                    symbol=row.symbol,
                    adv=row.adv,
                    bars=row.bars,
                    last_close=row.last_close,
                    window_return=row.window_return,
                    annual_volatility=profile.annual_volatility,
                    sharpe=profile.sharpe,
                    max_drawdown=profile.max_drawdown,
                    hurst=profile.stationarity.hurst,
                    verdict=row.verdict,
                    fadeable=row.fadeable,
                    is_implausible=profile.is_implausible,
                    fat_left_tail=profile.fat_left_tail,
                )
            )
        return ScreenResponse(
            rows=rows,
            considered=result.considered,
            passed_filters=result.passed_filters,
            profiled=result.profiled,
            suspected_actions=result.suspected_actions,
            sort_by=key.value,
        )


def build_analytics_router() -> APIRouter:
    """Every analytics endpoint, registered onto one router.

    Split into three registrars because they share nothing but the router: one
    function holding all of them grows past the complexity limit as endpoints
    are added, and the limit is there to catch exactly that.
    """
    router = APIRouter(tags=["analytics"])
    _register_search(router)
    _register_security(router)
    _register_cross_section(router)
    _register_screen(router)
    return router
