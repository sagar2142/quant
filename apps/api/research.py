"""Factor research endpoints — MASTER_PLAN §6.

Serves what `apps.cli.factor` prints. Same functions, same numbers: a second
implementation of "does this signal predict anything" would eventually
disagree with the first, and the one on screen would be the wrong one.

**A study is seconds, not milliseconds.** The cheap part is the signal
construction; the cost is scoring four forward horizons across fifteen hundred
names. That is fast enough to be interactive and slow enough that the console
must show a pending state rather than appearing frozen.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from apps.cli.factor import ROUND_TRIP_COST
from quant.research.factors import FORWARD_HORIZONS, Factor, FactorSpec, build_factor
from quant.research.ic import analyse_factor

__all__ = ["build_research_router"]


class HorizonRow(BaseModel):
    horizon: int
    ic: float
    information_ratio: float
    t_stat: float
    hit_rate: float
    sessions: int
    significant: bool


class BucketRow(BaseModel):
    quantile: int
    forward_return: float
    names: int


class FactorResponse(BaseModel):
    factor: str
    description: str
    names: int
    sessions: int
    horizons: list[HorizonRow]
    buckets: list[BucketRow]
    quantile_horizon: int
    spread: float
    monotonic: bool
    turnover: float
    #: Spread net of the round-trip charge implied by the turnover. The first
    #: thing that kills a signal with a real but tiny edge (§7.1).
    net_of_costs: float
    survives_costs: bool


class FactorListRow(BaseModel):
    name: str
    description: str


def build_research_router() -> APIRouter:
    router = APIRouter(tags=["research"])

    @router.get("/factors", dependencies=[ReadAccess])
    def factors() -> list[FactorListRow]:
        """The signal library. A closed set on purpose — a free-text formula
        field would let a typo become a discovery."""
        return [FactorListRow(name=f.value, description=f.description) for f in Factor]

    @router.get("/factor/{name}", response_model=FactorResponse, dependencies=[ReadAccess])
    def factor(
        name: str,
        horizon: int = Query(21, ge=1, le=252),
        sessions: int = Query(0, ge=0),
        min_adv: float = Query(1e7, ge=0),
        buckets: int = Query(5, ge=2, le=10),
    ) -> FactorResponse:
        from apps.api.analytics import _panel  # noqa: PLC0415 - shares the cached panel

        try:
            chosen = Factor(name)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=f"unknown factor {name!r}") from exc

        horizons = tuple(sorted({*FORWARD_HORIZONS, horizon}))
        scored = build_factor(
            _panel(), FactorSpec(chosen, min_adv=min_adv, window=sessions), horizons
        )
        if scored.is_empty():
            raise HTTPException(
                status_code=422,
                detail="no names survived the liquidity filter and lookback",
            )

        report = analyse_factor(scored, chosen.value, horizons, horizon, buckets)
        per_rebalance = min(1.0, report.turnover * report.quantile_horizon)
        net = report.spread - per_rebalance * ROUND_TRIP_COST

        return FactorResponse(
            factor=report.factor,
            description=chosen.description,
            names=report.names,
            sessions=report.sessions,
            horizons=[
                HorizonRow(
                    horizon=h.horizon,
                    ic=h.mean,
                    information_ratio=h.information_ratio,
                    t_stat=h.t_stat,
                    hit_rate=h.hit_rate,
                    sessions=h.sessions,
                    significant=h.is_significant,
                )
                for h in report.horizons
            ],
            buckets=[
                BucketRow(quantile=q.quantile, forward_return=q.mean_forward_return, names=q.names)
                for q in report.quantiles
            ],
            quantile_horizon=report.quantile_horizon,
            spread=report.spread,
            monotonic=report.is_monotonic,
            turnover=report.turnover,
            net_of_costs=net,
            survives_costs=net > 0,
        )

    return router
