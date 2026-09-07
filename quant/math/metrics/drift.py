"""Paper against backtest — MASTER_PLAN §M9, §35.

**The truth metric, and the only one that can fail.** Every other number in this
system is computed from the same history twice: a backtest and its statistics
both read the panel, so a mistake in the panel is invisible to both. Paper
trading is the first measurement taken *forward*, against sessions nobody chose,
through the code path that will place real orders. Comparing it to what the
backtest said would happen over the same sessions is the one comparison in the
whole system where reality gets a vote.

**What a gap actually means.** The two curves are the same strategy over the
same sessions, so they should differ only by the things the backtest models
rather than observes: fees, slippage, the fill it assumed against the fill that
happened, and the order that was blocked by a risk limit and never placed. A
small negative gap is the expected cost of being real. A large one says the cost
model is wrong, and a *positive* gap is worse than a negative one — it means the
backtest was pessimistic in a way nobody designed, which is usually a sign the
two are not running the same strategy at all.

**Tracking error is the honest headline, not total return.** Two curves can end
at the same place having disagreed every single session, and that is a broken
model rather than a working one. The daily dispersion is what says whether the
backtest predicts the paper book day by day or merely happens to arrive nearby.

**Sessions are matched by date, never by position.** A paper cycle that did not
run — a holiday, a failed workflow — leaves a hole, and zipping two lists of
different lengths would silently compare Monday's paper to Tuesday's backtest
and every day after. That misalignment produces a plausible tracking error from
a comparison that is meaningless.

float64 here: these are statistics, not money (§14.1.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

from quant.math.metrics.performance import TRADING_DAYS, returns_from_equity

__all__ = ["MIN_SESSIONS", "DriftReport", "compare_curves"]

FloatArray = npt.NDArray[np.float64]

#: Sessions of overlap below which no drift number is reported at all.
#:
#: A tracking error from three points is not a small sample, it is noise wearing
#: a decimal point — and it would be read as evidence. The gauntlet refuses
#: short samples for the same reason.
MIN_SESSIONS = 10


@dataclass(frozen=True)
class DriftReport:
    """How far the paper book has run from what the backtest predicted."""

    #: Sessions present in both curves. Zero means they never overlapped.
    sessions: int
    #: Total return of each over the matched sessions.
    paper_return: float = 0.0
    backtest_return: float = 0.0
    #: Annualised standard deviation of the daily return differences. The
    #: headline: two curves can end together having disagreed throughout.
    tracking_error: float = 0.0
    #: Mean daily difference, annualised. Negative is the expected direction —
    #: paper pays costs the backtest only estimated.
    mean_drift: float = 0.0
    #: Largest single-session disagreement, and when it happened. A single
    #: outlier session is a specific incident to go and read, not a statistic.
    worst_session: date | None = None
    worst_gap: float = 0.0
    #: Why there is no number, when there is none.
    reason: str = ""

    @property
    def measurable(self) -> bool:
        """Whether a drift number was actually computed.

        Absence is not agreement. A report with too few sessions has not found
        the curves to be close; it has not looked.
        """
        return self.sessions >= MIN_SESSIONS and not self.reason

    @property
    def gap(self) -> float:
        """Total-return difference. Negative means paper trailed the model."""
        return self.paper_return - self.backtest_return


def compare_curves(
    paper: dict[date, float],
    backtest: dict[date, float],
    periods_per_year: int = TRADING_DAYS,
) -> DriftReport:
    """Compare two equity curves on the sessions they share.

    Args:
        paper: Session -> equity, from the paper cycle's own log.
        backtest: Session -> equity, from the same strategy over the same window.
        periods_per_year: For annualising. Daily NSE sessions by default.

    Returns:
        A report, which may be unmeasurable. A caller must check `measurable`
        rather than reading `tracking_error` as zero — an unmeasured drift and
        a drift of zero are opposite findings.

    Matching is on the dates themselves, so a missed cycle costs one session of
    overlap rather than misaligning every session after it.
    """
    shared = sorted(set(paper) & set(backtest))
    if len(shared) < MIN_SESSIONS:
        return DriftReport(
            sessions=len(shared),
            reason=(
                f"{len(shared)} overlapping session(s); {MIN_SESSIONS} needed before "
                "a tracking error means anything"
            ),
        )

    paper_equity = np.asarray([paper[d] for d in shared], dtype=np.float64)
    model_equity = np.asarray([backtest[d] for d in shared], dtype=np.float64)

    if paper_equity[0] <= 0 or model_equity[0] <= 0:
        return DriftReport(sessions=len(shared), reason="a curve starts at or below zero")

    paper_returns = returns_from_equity(paper_equity)
    model_returns = returns_from_equity(model_equity)
    difference = paper_returns - model_returns

    # ddof=1: this is a sample of sessions, not the population of them.
    dispersion = float(np.std(difference, ddof=1)) if difference.size > 1 else 0.0
    worst_index = int(np.argmax(np.abs(difference))) if difference.size else 0

    return DriftReport(
        sessions=len(shared),
        paper_return=float(paper_equity[-1] / paper_equity[0] - 1.0),
        backtest_return=float(model_equity[-1] / model_equity[0] - 1.0),
        tracking_error=dispersion * float(np.sqrt(periods_per_year)),
        mean_drift=float(np.mean(difference)) * periods_per_year,
        # `shared[0]` is the base of the first return, so differences are
        # indexed from the second session onward.
        worst_session=shared[worst_index + 1] if difference.size else None,
        worst_gap=float(difference[worst_index]) if difference.size else 0.0,
    )
