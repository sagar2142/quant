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
from pydantic import BaseModel

from apps.cli.terminal import aligned_returns, load_actions, series_for
from core.clock import as_decision_time, utc_now
from core.config import settings
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from quant.analytics.crosssection import analyse_cross_section
from quant.analytics.security import profile_security

__all__ = ["build_analytics_router"]

#: Cap on symbols per cross-section request. The correlation work is O(n^2) and
#: a browser cannot read a 200-name matrix anyway.
MAX_SYMBOLS = 40

#: Rows returned by the symbol search. Enough to pick from, few enough to scan.
SEARCH_LIMIT = 60

#: A cross-section is a comparison; one name is a security screen.
MIN_CROSS_SECTION = 2

#: Trailing window used to rank the symbol search by liquidity.
LIQUIDITY_WINDOW = 60


class HorizonReturn(BaseModel):
    label: str
    value: float | None


class SecurityResponse(BaseModel):
    """One security, fully decomposed. Mirrors `SecurityProfile`."""

    symbol: str
    observations: int
    last_close: float

    horizons: list[HorizonReturn]
    cagr: float
    high_52w: float
    low_52w: float
    off_high: float

    annual_volatility: float
    max_drawdown: float
    current_drawdown: float
    adv_value: float | None

    sharpe: float
    sortino: float
    calmar: float
    hit_rate: float

    skewness: float
    kurtosis: float
    var_5: float
    cvar_5: float
    tail_ratio: float

    verdict: str
    adf_pvalue: float
    kpss_pvalue: float
    hurst: float
    tradable_as_mean_reversion: bool
    autocorrelation: dict[str, float]

    realised_vol: float
    ewma_vol: float
    vol_regime: str

    is_implausible: bool
    fat_left_tail: bool


class NameRow(BaseModel):
    symbol: str
    total_return: float
    annual_volatility: float
    sharpe: float
    beta: float
    correlation_to_market: float
    weight_hrp: float
    weight_erc: float
    cluster: int


class CrossSectionResponse(BaseModel):
    names: list[NameRow]
    sessions: int
    mean_correlation: float
    clusters: int
    effective_bets: float
    diversification_ratio: float
    condition_number: float
    shrinkage: float
    market_return: float
    market_volatility: float
    is_ill_conditioned: bool
    concentration_warning: str | None
    #: Row/column order of `correlation`. Carried explicitly because `names` is
    #: ranked by return while the matrix keeps input order — indexing one by
    #: the other would render a heatmap of the wrong pairs.
    correlation_labels: list[str]
    correlation: list[list[float]]


@lru_cache(maxsize=1)
def _panel() -> pl.DataFrame:
    """The whole panel, read once.

    Cached because it is immutable between ingests and large enough that a
    per-request read would be felt on every screen. A restart picks up new
    sessions, which is the right granularity for a daily-frequency system.
    """
    store = PanelStore(settings.lake, venue="NSE")
    return store.view(as_of=as_decision_time(utc_now()))


def _windowed(history: pl.DataFrame, sessions: int) -> pl.DataFrame:
    if sessions <= 0:
        return history
    recent = history["event_time"].unique().sort().tail(sessions)
    return history.filter(pl.col("event_time").is_in(recent.implode()))


def _register_search(router: APIRouter) -> None:
    @router.get("/symbols")
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
    @router.get("/security/{symbol}", response_model=SecurityResponse)
    def security(symbol: str, sessions: int = Query(0, ge=0)) -> SecurityResponse:
        try:
            history = _windowed(_panel(), sessions)
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        name = symbol.upper()
        actions = load_actions(history, [name])
        rows = series_for(history, name, actions)
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=f"{name} is not in the panel")

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

    @router.get("/security/{symbol}/series")
    def series(symbol: str, sessions: int = Query(0, ge=0)) -> dict[str, list[object]]:
        """Back-adjusted close series, for charting."""
        history = _windowed(_panel(), sessions)
        name = symbol.upper()
        rows = series_for(history, name, load_actions(history, [name]))
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=f"{name} is not in the panel")
        return {
            "dates": [d.date().isoformat() for d in rows["event_time"].to_list()],
            "closes": [float(c) for c in rows["close"].to_list()],
        }


def _register_cross_section(router: APIRouter) -> None:
    @router.get("/crosssection", response_model=CrossSectionResponse)
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
    return router
