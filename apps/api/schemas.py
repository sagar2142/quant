"""Analytics response models — MASTER_PLAN §12.6.

The wire shape, kept apart from the endpoints that build it. A response model
is a contract with the console; an endpoint is how one gets filled. Splitting
them means a field can be read without scrolling past the query it came from.
"""

from __future__ import annotations

from pydantic import BaseModel

__all__ = [
    "CrossSectionResponse",
    "HorizonReturn",
    "NameRow",
    "ScreenResponse",
    "ScreenRowResponse",
    "SecurityResponse",
]


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


class ScreenRowResponse(BaseModel):
    symbol: str
    adv: float
    bars: int
    last_close: float
    window_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    hurst: float
    verdict: str
    fadeable: bool
    is_implausible: bool
    fat_left_tail: bool


class ScreenResponse(BaseModel):
    rows: list[ScreenRowResponse]
    considered: int
    passed_filters: int
    profiled: int
    suspected_actions: int
    sort_by: str
