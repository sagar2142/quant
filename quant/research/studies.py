"""Questions a cross-sectional factor cannot ask — MASTER_PLAN §5.1, §6.

The factor lab answers one shape of question: rank every name by a signal
today, and see whether that ranking predicts a *forward* return. Four of the
pre-registered hypotheses are not that shape, and forcing them into it would
answer a different question than the one registered:

    gap reversion       a *same-session* claim. The gap opens and closes inside
                        one bar, so a forward horizon measures the day after
                        the effect rather than the effect.

    conditional IC      "reversal is stronger in high-turnover names" is a
                        claim about where an effect lives, not about a ranking.
                        It needs the same signal scored inside subsets.

    dispersion timing   "deploy when dispersion is high" is about *when* to
                        trade, not what to hold. Its unit of observation is a
                        session, not a name.

    double sort         "illiquidity survives controlling for size" asks
                        whether one effect is the other wearing a disguise,
                        which a single sort cannot separate by construction.

Each returns the statistic its hypothesis committed to, so the verdict is read
off the same way stage 3's is. None of them is a tradeable signal on its own —
they are tests of claims, and a claim surviving one is permission to build a
factor, not a factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from quant.research.factors import FactorSpec, build_factor
from quant.research.ic import MIN_NAMES_PER_SESSION

__all__ = [
    "MIN_SESSIONS",
    "MIN_T_STAT",
    "StudyResult",
    "conditional_ic",
    "dispersion_timing",
    "double_sorted_ic",
    "gap_reversion",
]

#: Sessions below which any of these is noise. The IC module's own floor.
MIN_SESSIONS = 30

#: Significance floor, matching every pre-registered hypothesis in the
#: catalogue. Uniform on purpose: a threshold tuned per study is a threshold
#: tuned to the answer.
MIN_T_STAT = 3.0

#: Terciles, for the conditioning studies. Three is the fewest that has a
#: middle to ignore, which is what makes top-versus-bottom meaningful.
BUCKETS = 3


@dataclass(frozen=True)
class StudyResult:
    """One study's answer, in the form its hypothesis committed to."""

    statistic: float
    t_stat: float
    observations: int
    detail: str

    @property
    def is_significant(self) -> bool:
        return abs(self.t_stat) >= MIN_T_STAT and self.observations >= MIN_SESSIONS


def _summarise(values: npt.NDArray[np.float64], detail: str) -> StudyResult:
    """Mean of a per-session series, with the t-statistic of that mean."""
    finite = values[np.isfinite(values)]
    if finite.size < MIN_SESSIONS:
        return StudyResult(0.0, 0.0, int(finite.size), f"{detail} — too few sessions")
    mean = float(np.mean(finite))
    deviation = float(np.std(finite, ddof=1))
    t_stat = mean / (deviation / np.sqrt(finite.size)) if deviation > 0 else 0.0
    return StudyResult(mean, t_stat, int(finite.size), detail)


def _with_column(panel: pl.DataFrame, scored: pl.DataFrame, column: str) -> pl.DataFrame:
    """Re-attach a raw panel column to a scored frame.

    `build_factor` returns identity, signal and forward returns and nothing
    else, so a study conditioning on volume or price has to fetch it back.
    Joined on `instrument_id` rather than symbol: a symbol is not identity
    (§3.3) and 344 of them map to more than one ISIN here.
    """
    if column in scored.columns:
        return scored
    if column not in panel.columns:
        return scored
    return scored.join(
        panel.select("event_time", "instrument_id", column),
        on=["event_time", "instrument_id"],
        how="inner",
    )


def gap_reversion(panel: pl.DataFrame) -> StudyResult:
    """Do overnight gaps reverse *within the same session*?

    The gap is `open / previous close - 1`; the intraday move is
    `close / open - 1`. Both belong to one bar, so nothing here reads a forward
    return — which is exactly why the factor lab cannot express it, and also
    why this is not a tradeable signal: acting on it requires trading intraday,
    and this system trades daily bars.

    A negative correlation means gaps fade, which is the registered prediction.

    **Read the result knowing this statistic is not identified from daily
    bars.** `open` appears in the gap with a plus sign and in the intraday
    return with a minus sign, so any pricing error in the recorded open — an
    opening-auction print, a wide spread on a mid-cap, a stale quote — lands in
    the two series with opposite signs and manufactures negative correlation
    out of a market that has none.

    It is not a small effect. Simulating a market with *zero* gap reversion and
    adding noise to the recorded open alone produces t = -15.6 at 0.5% noise
    and t = -54.6 at 1%. The measured value on the NSE panel is -0.2180 at
    t = -117.6, which is inside the range that pure microstructure noise
    reproduces, and Indian mid-caps carry that much spread at the open
    routinely.

    So a large negative number here is evidence of nothing on its own.
    Separating the two explanations needs intraday quotes, which this system
    does not have; the registered hypothesis was abandoned on exactly that
    ground rather than confirmed. `tests/test_studies.py` holds the
    demonstration, so the confound cannot be quietly forgotten.
    """
    ordered = panel.sort(["symbol", "event_time"]).with_columns(
        (pl.col("open") / pl.col("close").shift(1).over("symbol") - 1).alias("gap"),
        (pl.col("close") / pl.col("open") - 1).alias("intraday"),
    )
    per_session = (
        ordered.drop_nulls(["gap", "intraday"])
        .group_by("event_time")
        .agg(
            pl.corr(pl.col("gap").rank(), pl.col("intraday").rank()).alias("rho"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("rho").is_finite())
    )
    return _summarise(
        per_session["rho"].to_numpy(),
        "rank correlation of overnight gap with the same session's intraday return",
    )


def _tercile(column: str) -> pl.Expr:
    """Cross-sectional tercile of a column, per session."""
    return (
        (pl.col(column).rank("ordinal").over("event_time") - 1)
        * BUCKETS
        // pl.len().over("event_time")
    )


def conditional_ic(
    panel: pl.DataFrame, spec: FactorSpec, horizon: int, condition: str
) -> StudyResult:
    """Is a signal's IC larger inside the top bucket of `condition`?

    Scores the same factor separately within the top and bottom tercile of a
    conditioning variable, and reports the difference. Sorting on the signal
    across the whole cross-section — what the lab does — would average the two
    and answer neither.

    Returns the *difference* in IC, so a positive statistic means the effect is
    genuinely stronger where the hypothesis predicted.
    """
    scored = _with_column(panel, build_factor(panel, spec, (horizon,)), condition)
    if scored.is_empty() or condition not in scored.columns:
        return StudyResult(0.0, 0.0, 0, f"{condition} unavailable")

    column = f"fwd_{horizon}"
    bucketed = scored.drop_nulls(["signal", column, condition]).with_columns(
        _tercile(condition).alias("bucket")
    )
    per_session = (
        bucketed.filter(pl.col("bucket").is_in([0, BUCKETS - 1]))
        .group_by(["event_time", "bucket"])
        .agg(
            pl.corr(pl.col("signal").rank(), pl.col(column).rank()).alias("ic"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("ic").is_finite())
    )
    wide = per_session.pivot(on="bucket", index="event_time", values="ic").drop_nulls()
    top, bottom = str(BUCKETS - 1), "0"
    if top not in wide.columns or bottom not in wide.columns:
        return StudyResult(0.0, 0.0, 0, "one bucket never had enough names")

    difference = (wide[top] - wide[bottom]).to_numpy()
    return _summarise(difference, f"IC in the top {condition} tercile minus the bottom")


def double_sorted_ic(
    panel: pl.DataFrame, spec: FactorSpec, horizon: int, control: str
) -> StudyResult:
    """Does a signal still predict *within* buckets of a control variable?

    A single sort on illiquidity is also a sort on size, because the two are
    heavily confounded. Scoring inside size terciles asks whether anyone is
    paid for illiquidity itself, or whether the premium is the size effect
    under another name.

    Returns the mean IC across control buckets: the effect that survives.
    """
    scored = _with_column(panel, build_factor(panel, spec, (horizon,)), control)
    if scored.is_empty() or control not in scored.columns:
        return StudyResult(0.0, 0.0, 0, f"{control} unavailable")

    column = f"fwd_{horizon}"
    bucketed = scored.drop_nulls(["signal", column, control]).with_columns(
        _tercile(control).alias("bucket")
    )
    per_session = (
        bucketed.group_by(["event_time", "bucket"])
        .agg(
            pl.corr(pl.col("signal").rank(), pl.col(column).rank()).alias("ic"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("ic").is_finite())
        .group_by("event_time")
        .agg(pl.col("ic").mean().alias("ic"))
    )
    return _summarise(
        per_session["ic"].to_numpy(), f"mean IC within {control} terciles, not across them"
    )


def dispersion_timing(panel: pl.DataFrame, spec: FactorSpec, horizon: int) -> StudyResult:
    """Does high cross-sectional dispersion predict a *better factor* month?

    A timing claim, not a selection one: its unit of observation is a session
    rather than a name, so there is no ranking of instruments for the lab to
    score. Correlates each session's return dispersion with the factor IC that
    followed.

    Positive means the factor pays more after dispersed sessions, which is the
    registered prediction — an argument about when to deploy a book rather than
    what to hold in it.
    """
    scored = build_factor(panel, spec, (horizon,))
    if scored.is_empty():
        return StudyResult(0.0, 0.0, 0, "factor produced no signal")

    column = f"fwd_{horizon}"
    per_session = (
        scored.drop_nulls(["signal", column])
        .group_by("event_time")
        .agg(
            pl.corr(pl.col("signal").rank(), pl.col(column).rank()).alias("ic"),
            pl.col(column).std().alias("dispersion"),
            pl.len().alias("names"),
        )
        .filter((pl.col("names") >= MIN_NAMES_PER_SESSION) & pl.col("ic").is_finite())
        .sort("event_time")
        .drop_nulls(["dispersion"])
    )
    if per_session.height < MIN_SESSIONS:
        return StudyResult(0.0, 0.0, per_session.height, "too few sessions")

    # Dispersion leads the IC it is supposed to predict, so it is lagged. Using
    # the same session's dispersion would correlate a statistic with itself.
    lagged = per_session.with_columns(pl.col("dispersion").shift(1).alias("prior")).drop_nulls(
        ["prior"]
    )
    dispersion = lagged["prior"].to_numpy()
    ic = lagged["ic"].to_numpy()
    if dispersion.size < MIN_SESSIONS or np.std(dispersion) == 0:
        return StudyResult(0.0, 0.0, int(dispersion.size), "dispersion did not vary")

    rho = float(np.corrcoef(dispersion, ic)[0, 1])
    n = dispersion.size
    t_stat = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho**2)) if abs(rho) < 1 else 0.0
    return StudyResult(
        rho, float(t_stat), int(n), "correlation of prior-session dispersion with factor IC"
    )
