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

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from apps.cli.factor import ROUND_TRIP_COST
from quant.research.factors import FORWARD_HORIZONS, Factor, FactorSpec, build_factor
from quant.research.ic import analyse_factor
from quant.research.riskmodel import RiskModel, build_risk_model, decompose

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
    #: Median of the same bucket. Beside the mean because the two disagree
    #: often enough to matter — the gap is the bucket's tail.
    median_forward_return: float = 0.0
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
    #: The same spread for the typical name rather than the average one. A
    #: portfolio earns the mean, so `spread` is the number that pays; this says
    #: whether that mean describes the holdings or a handful of them.
    median_spread: float = 0.0
    #: True when the mean spread and the median run in opposite directions —
    #: the factor is being paid by a few extreme names rather than by the
    #: effect it claims to harvest. Twelve of twenty-eight factors trip it.
    tail_driven: bool = False


#: The factor set the risk model is estimated on. Deliberately mixed rather
#: than four momentum variants: a set that is all one effect produces a
#: near-collinear covariance whose per-factor attribution means little, which
#: the model reports but cannot repair.
RISK_FACTORS = (
    Factor.MOMENTUM_12_1,
    Factor.VOLATILITY_60,
    Factor.REVERSAL_5D,
    Factor.HIGH_52W_PROXIMITY,
    Factor.ILLIQUIDITY,
)


@lru_cache(maxsize=4)
def _cached_risk_model(sessions: int, fingerprint: tuple[int, str]) -> RiskModel | None:
    """The risk model for one window, held until the lake changes.

    **Estimating it costs minutes**, not milliseconds: it is one cross-sectional
    regression per session over the whole panel, and a screen that recomputed
    that per request would time out — which it did, the first time this was
    served uncached. Keyed on the lake's fingerprint for the same reason the
    panel is, so an ingest invalidates it rather than a restart being required.
    """
    from apps.api.analytics import _panel  # noqa: PLC0415 - shares the cached panel

    del fingerprint  # cache identity only
    return build_risk_model(_panel(), RISK_FACTORS, window=sessions)


class RiskContribution(BaseModel):
    name: str
    #: Share of total portfolio variance. May be negative: a factor can reduce
    #: risk by offsetting another.
    share: float


class RiskResponse(BaseModel):
    """Where the paper book's risk actually comes from."""

    present: bool
    total_volatility: float = 0.0
    factor_volatility: float = 0.0
    specific_volatility: float = 0.0
    factor_share: float = 0.0
    specific_share: float = 0.0
    contributions: list[RiskContribution] = []
    factor_volatilities: list[RiskContribution] = []
    sessions: int = 0
    names: int = 0
    #: True when the factors are near-collinear, so the per-factor split below
    #: is unstable even though the total is sound.
    ill_conditioned: bool = False
    condition: float = 0.0
    note: str = ""


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

    @router.get("/risk/model", response_model=RiskResponse, dependencies=[ReadAccess])
    def risk_model(sessions: int = Query(756, ge=252, le=2520)) -> RiskResponse:
        """Decompose the paper book into factor and specific risk.

        The question no screen could answer: the library scores signals one at
        a time and the correlation matrix reports pairwise overlap, and neither
        says how much of *this book's* risk is a factor. On the real account
        two thirds of it turns out to be the market, which reframes what the
        factor work is actually moving.
        """
        from apps.api.analytics import _lake_fingerprint  # noqa: PLC0415 - shared cache
        from apps.api.book import DEFAULT_STATE_DIR, _latest_marks  # noqa: PLC0415
        from trading.paper.state import PaperStateStore, StateCorruptError  # noqa: PLC0415

        store = PaperStateStore(DEFAULT_STATE_DIR)
        if not store.exists():
            return RiskResponse(present=False, note="No paper book to decompose yet.")
        try:
            state = store.restore()
        except StateCorruptError:
            return RiskResponse(present=False, note="Paper state unreadable.")

        marks = _latest_marks(None)
        values = {
            str(i): float(p.market_value(marks.get(i, p.average_price)))
            for i, p in state.portfolio.positions.items()
            if not p.is_flat
        }
        if not values:
            return RiskResponse(present=False, note="The book holds nothing.")

        model = _cached_risk_model(sessions, _lake_fingerprint())
        if model is None:
            return RiskResponse(
                present=False,
                note=(
                    "Not enough history to estimate a covariance. Reporting "
                    "nothing beats reporting a matrix fitted to forty sessions."
                ),
            )

        total = sum(values.values())
        result = decompose(model, {k: v / total for k, v in values.items()})
        return RiskResponse(
            present=True,
            total_volatility=result.total_volatility,
            factor_volatility=result.factor_volatility,
            specific_volatility=result.specific_volatility,
            factor_share=result.factor_share,
            specific_share=result.specific_share,
            contributions=[
                RiskContribution(name=k, share=v)
                for k, v in sorted(result.contributions.items(), key=lambda kv: -abs(kv[1]))
            ],
            factor_volatilities=[
                RiskContribution(name=k, share=v) for k, v in model.factor_volatility.items()
            ],
            sessions=model.sessions,
            names=len(values),
            ill_conditioned=model.is_ill_conditioned,
            condition=model.condition,
        )

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
                BucketRow(
                    quantile=q.quantile,
                    forward_return=q.mean_forward_return,
                    median_forward_return=q.median_forward_return,
                    names=q.names,
                )
                for q in report.quantiles
            ],
            quantile_horizon=report.quantile_horizon,
            spread=report.spread,
            monotonic=report.is_monotonic,
            turnover=report.turnover,
            net_of_costs=net,
            survives_costs=net > 0,
            median_spread=report.median_spread,
            tail_driven=report.is_tail_driven,
        )

    return router
