"""Factor risk model — MASTER_PLAN §6, §8.

**The question nothing in this system could answer.** The factor library scores
signals one at a time and the correlation matrix reports pairwise overlap, but
neither says how much of a *portfolio's* risk comes from a factor and how much
is specific to the names in it. That gap is not academic here: the four factors
that survive costs are all momentum variants, `momentum_12_1` and
`residual_momentum` correlate at 0.97, and a book built on them is closer to one
bet than to four. Nothing on any screen said so.

**What this computes.** Cross-sectional regressions, the Fama-MacBeth way:

    1. each session, regress that session's returns on the factor exposures
    2. the slopes are the factor returns for that session
    3. their covariance is the factor covariance matrix
    4. what the regression does not explain is specific risk

A portfolio's variance then splits into `wᵀ B Σ Bᵀ w` (factor) and `wᵀ D w`
(specific), and the first term decomposes per factor. That decomposition is the
output: not "your book is 18% volatile" but "14 of those 18 points are momentum
and you did not choose that."

**Exposures are standardised cross-sectionally, per session.** A raw momentum
score and a raw volatility score are not comparable units, and regressing on
both unscaled makes the slope of one an artefact of the other's dispersion.

**The market is a factor, and it has to be.** Style exposures are z-scored
cross-sectionally, so they sum to zero across the universe and an equal-weight
book of everything has literally no style exposure. Without a market column its
entire variance would be reported as name-specific, which for a long-only book
is the opposite of the truth: its dominant risk is that the market falls. The
market factor is a column of ones, whose slope each session is that session's
cross-sectional average return.

**Nothing here is a forecast.** These are realised covariances over a trailing
window, and a covariance estimated from 252 sessions is a statement about those
252 sessions. It is used to say what a book is exposed to, not what it will
earn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl

from quant.math.linalg.covariance import condition_number, correlation_from_covariance
from quant.research.composite import zscore_by_session
from quant.research.factors import Factor, FactorSpec, build_factor

__all__ = [
    "MARKET",
    "MAX_CONDITION",
    "MIN_NAMES_PER_REGRESSION",
    "MIN_SESSIONS",
    "RiskDecomposition",
    "RiskModel",
    "build_risk_model",
    "decompose",
]

FloatArray = npt.NDArray[np.float64]

#: A cross-sectional regression over fewer names than this is fitting noise.
#: Twenty is the same floor the IC uses for the same reason.
MIN_NAMES_PER_REGRESSION = 20

#: Sessions of factor returns needed before their covariance means anything.
#: A quarter is too few for a matrix this wide; a year is the usual choice and
#: keeps n well above the number of factors.
MIN_SESSIONS = 252

#: Trading sessions per year, for annualising.
SESSIONS_PER_YEAR = 252

#: The market factor's column name. A column of ones, so its slope each
#: session is the cross-sectional average return.
MARKET = "market"

#: Condition number above which the factor set is treated as near-collinear.
#: A hundred is the usual rule of thumb; this library clears it easily, which
#: is the point — sixteen factors worth about six independent bets cannot be
#: attributed to sixteen separate causes.
MAX_CONDITION = 100.0


@dataclass(frozen=True)
class RiskModel:
    """Estimated factor structure over a trailing window."""

    factors: tuple[Factor, ...]
    #: Factor return per session, shape (sessions, factors). The slopes of the
    #: cross-sectional regressions, which is what a factor "returned".
    factor_returns: FloatArray
    #: Covariance of those returns, annualised, shape (factors, factors).
    covariance: FloatArray
    #: Specific (residual) variance per instrument, annualised.
    specific_variance: dict[str, float]
    #: Latest exposure per instrument per factor, shape (names, factors).
    exposures: dict[str, FloatArray]
    sessions: int
    #: Condition number of the factor covariance. High means the factors are
    #: close to collinear and the *individual* attributions below are poorly
    #: identified even though the total is sound — the regression can trade one
    #: near-duplicate factor against another and land anywhere on that line.
    condition: float = 1.0

    @property
    def is_ill_conditioned(self) -> bool:
        """Whether per-factor attribution should be read with suspicion.

        This model's own library trips it: `momentum_12_1` and
        `residual_momentum` correlate at 0.97, so a book can show a large
        positive contribution from one and a large negative one from the other
        while the pair together contributes something modest and stable. The
        total decomposition is still right; the split between those two is not
        a fact about the portfolio.
        """
        return self.condition > MAX_CONDITION

    @property
    def factor_names(self) -> tuple[str, ...]:
        """Column order of the exposure matrix: market, then the styles."""
        return (MARKET, *(f.value for f in self.factors))

    @property
    def factor_volatility(self) -> dict[str, float]:
        """Annualised volatility of each factor's own return series."""
        return {
            name: float(np.sqrt(max(0.0, self.covariance[i, i])))
            for i, name in enumerate(self.factor_names)
        }

    def format(self) -> str:
        lines = [
            f"  {len(self.factors)} factors over {self.sessions:,} sessions,"
            f" {len(self.specific_variance):,} names",
            "",
            "  FACTOR VOLATILITY (annualised)",
        ]
        if self.is_ill_conditioned:
            lines.insert(
                1,
                f"  ILL-CONDITIONED: condition {self.condition:,.0f} — the factors are"
                " near-collinear, so per-factor attribution is unstable",
            )
        lines.extend(f"    {name:<24}{vol:>8.2%}" for name, vol in self.factor_volatility.items())
        return "\n".join(lines)


@dataclass(frozen=True)
class RiskDecomposition:
    """Where a portfolio's risk comes from."""

    total_volatility: float
    factor_volatility: float
    specific_volatility: float
    #: Variance contributed by each factor, as a share of total variance. These
    #: sum with `specific_share` to one, and may be negative: a factor can
    #: reduce portfolio risk by offsetting another.
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def specific_share(self) -> float:
        if self.total_volatility <= 0:
            return 0.0
        return float(self.specific_volatility**2 / self.total_volatility**2)

    @property
    def factor_share(self) -> float:
        return 1.0 - self.specific_share

    def format(self) -> str:
        lines = [
            f"  total volatility     {self.total_volatility:>8.2%}",
            f"    from factors       {self.factor_volatility:>8.2%}"
            f"   ({self.factor_share:>6.1%} of variance)",
            f"    specific           {self.specific_volatility:>8.2%}"
            f"   ({self.specific_share:>6.1%} of variance)",
            "",
            "  VARIANCE BY FACTOR",
        ]
        ranked = sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))
        for name, share in ranked:
            bar = "#" * max(0, int(abs(share) * 60))
            lines.append(f"    {name:<24}{share:>+8.1%}  {bar[:34]}")
        return "\n".join(lines)


def _exposure_frame(
    history: pl.DataFrame, factors: tuple[Factor, ...], window: int
) -> pl.DataFrame:
    """One column per factor, cross-sectionally standardised each session.

    Standardised because the regression slope of an unscaled factor absorbs its
    dispersion: momentum measured in return units and volatility measured in
    standard deviations would produce slopes that are not comparable, and the
    covariance of those slopes would be meaningless.
    """
    frames: list[pl.DataFrame] = []
    for factor in factors:
        scored = build_factor(history, FactorSpec(factor, window=window), (1,))
        if scored.is_empty():
            continue
        frames.append(
            scored.select(
                "event_time",
                "instrument_id",
                zscore_by_session("signal").alias(factor.value),
                pl.col("fwd_1").alias("_fwd"),
            )
        )

    if not frames:
        return pl.DataFrame()

    joined = frames[0]
    for frame in frames[1:]:
        joined = joined.join(frame.drop("_fwd"), on=["event_time", "instrument_id"], how="inner")
    # A column of ones, so the regression carries an intercept whose slope is
    # the session's average return. Named rather than implicit: it appears in
    # the decomposition alongside the styles, which is where a long-only book's
    # risk actually lives.
    return joined.drop_nulls().with_columns(pl.lit(1.0).alias(MARKET))


def build_risk_model(
    history: pl.DataFrame,
    factors: tuple[Factor, ...],
    window: int = 0,
) -> RiskModel | None:
    """Estimate factor returns, their covariance, and specific risk.

    Returns:
        None when the panel cannot support the estimate — too few sessions with
        enough names, or no factor produced exposures. Returning None rather
        than a degenerate model is deliberate: a covariance matrix estimated
        from forty sessions would be reported with the same confidence as one
        from a thousand, and nothing downstream could tell them apart.
    """
    frame = _exposure_frame(history, factors, window)
    if frame.is_empty():
        return None

    present = tuple(f for f in factors if f.value in frame.columns)
    if not present:
        return None

    # Market first: the intercept absorbs the level so the style slopes are
    # cross-sectional tilts rather than tilts plus whatever the market did.
    names = [MARKET, *[f.value for f in present]]
    slopes: list[FloatArray] = []
    residual_squares: dict[str, list[float]] = {}

    for (_session,), group in frame.group_by(["event_time"], maintain_order=True):
        if group.height < MIN_NAMES_PER_REGRESSION:
            continue
        exposures = group.select(names).to_numpy().astype(np.float64)
        returns = group["_fwd"].to_numpy().astype(np.float64)
        if not np.isfinite(exposures).all() or not np.isfinite(returns).all():
            continue

        # Least squares rather than a normal-equation inverse: the exposure
        # matrix is often near-collinear here — that collinearity is the very
        # thing being measured — and inverting it directly would amplify it.
        slope, *_ = np.linalg.lstsq(exposures, returns, rcond=None)
        slopes.append(slope)

        residuals = returns - exposures @ slope
        for instrument_id, residual in zip(group["instrument_id"], residuals, strict=True):
            residual_squares.setdefault(str(instrument_id), []).append(float(residual**2))

    if len(slopes) < MIN_SESSIONS:
        return None

    factor_returns = np.vstack(slopes)
    covariance = np.atleast_2d(
        np.asarray(np.cov(factor_returns, rowvar=False), dtype=np.float64)
    ) * float(SESSIONS_PER_YEAR)

    specific = {
        instrument_id: float(np.mean(squares) * SESSIONS_PER_YEAR)
        for instrument_id, squares in residual_squares.items()
        if squares
    }

    latest = frame.filter(pl.col("event_time") == frame["event_time"].max())
    latest_exposures = {
        str(row[0]): np.asarray(row[1:], dtype=np.float64)
        for row in latest.select("instrument_id", *names).iter_rows()
    }

    return RiskModel(
        # Of the *correlation* matrix, not the covariance. A covariance's
        # condition number conflates collinearity with scale: a factor at
        # 26% volatility beside one at 5% is 28x in variance before any
        # correlation exists, so covariance conditioning flags a genuinely
        # diversified set as ill-conditioned and trains the reader to
        # ignore the warning.
        condition=float(condition_number(correlation_from_covariance(covariance))),
        factors=present,
        factor_returns=factor_returns,
        covariance=covariance,
        specific_variance=specific,
        exposures=latest_exposures,
        sessions=len(slopes),
    )


def decompose(model: RiskModel, weights: dict[str, float]) -> RiskDecomposition:
    """Split a portfolio's variance into factor and specific parts.

    Args:
        weights: Portfolio weight per instrument_id. Names the model has no
            exposure for are dropped, and their weight with them — a position
            the model cannot see contributes no *estimated* risk, which is not
            the same as contributing none.
    """
    known = {k: v for k, v in weights.items() if k in model.exposures}
    if not known:
        return RiskDecomposition(0.0, 0.0, 0.0)

    w = np.array([known[k] for k in known], dtype=np.float64)
    exposures = np.vstack([model.exposures[k] for k in known])

    # Portfolio exposure to each factor: the weighted sum of its holdings'.
    portfolio_exposure = w @ exposures
    factor_variance = float(portfolio_exposure @ model.covariance @ portfolio_exposure)
    specific_variance = float(
        sum(model.specific_variance.get(k, 0.0) * known[k] ** 2 for k in known)
    )
    total_variance = max(0.0, factor_variance + specific_variance)

    # Each factor's share is its row of the quadratic form, so the parts sum to
    # the whole rather than being computed independently and left not to.
    contributions: dict[str, float] = {}
    if total_variance > 0:
        per_factor = portfolio_exposure * (model.covariance @ portfolio_exposure)
        for name, value in zip(model.factor_names, per_factor, strict=True):
            contributions[name] = float(value / total_variance)

    return RiskDecomposition(
        total_volatility=float(np.sqrt(total_variance)),
        factor_volatility=float(np.sqrt(max(0.0, factor_variance))),
        specific_volatility=float(np.sqrt(max(0.0, specific_variance))),
        contributions=contributions,
    )
