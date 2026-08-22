"""Risk limits as the console shows them — MASTER_PLAN §8, §12.7.

**Separated from the book because it answers a different question.** The book
is what you hold; this is what you are allowed to hold, and where you currently
sit against it. They share a data source and nothing else.

**The distinction this module is built around is null versus zero.** Only four
of the ten limits describe a state a portfolio can be *in*. The rest are
decided about an order as it is submitted, and a book at rest has no value for
them. The console previously hardcoded `observed: 0, passed: true` for all ten,
which reported every limit as measured, untouched and passing no matter what
the account held — including the one currently sitting above its cap.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel

from apps.api.snapshot import BookSnapshot
from trading.risk.limits import RiskLimits

__all__ = ["LimitRow", "limit_rows"]


class LimitRow(BaseModel):
    name: str
    threshold: float
    unit: str
    detail: str
    #: What the book currently shows for this limit, or None when the limit is
    #: checked per order and has no value at rest. The console renders None as
    #: an em dash. It previously hardcoded 0, which reads as "measured, and
    #: nothing is being used" — the opposite of "not measured".
    observed: float | None = None
    #: None travels with a null `observed`: a limit nobody measured has not
    #: passed, and colouring it green is the failure this whole module exists
    #: to prevent.
    passed: bool | None = None


def _observations(snapshot: BookSnapshot, limits: RiskLimits) -> dict[str, float | None]:
    """Current value per limit name, or None where the book cannot show one.

    Only four of the ten limits describe a state a portfolio can be *in*. The
    rest — order size, price band, order rate, open orders, ADV participation —
    are decided about an order as it is submitted, and no book at rest has a
    value for them. Reporting zero for those would say the budget is untouched
    when the truth is that the question does not apply.

    `cluster_concentration` is absent for a different reason: the paper state
    stores no cluster assignments, so it is unmeasured rather than
    inapplicable. Both render the same way, and both are honest.
    """
    del limits  # thresholds live on the rows; this maps observations only
    return {
        "position_size": _maybe_float(snapshot.largest_position_pct),
        "gross_exposure": _maybe_float(snapshot.gross_exposure),
        "net_exposure": _maybe_float(snapshot.net_exposure),
        "daily_loss": _maybe_float(snapshot.day_pnl_pct),
    }


def _maybe_float(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _verdict(name: str, observed: float | None, threshold: float) -> bool | None:
    """Whether the observation is inside the limit, or None if unmeasured.

    Compared strictly, and a marginal overshoot is reported as one. The engine
    enforces `max_position_pct` on the *resulting position at order time*; a
    holding that appreciates afterwards can sit slightly above the cap without
    any order having breached it. That is drift, not an engine failure, and the
    screen exists to show it — the real book currently carries one name at
    10.002% for exactly this reason. Widening the comparison to hide it would
    reintroduce the problem this module was written to fix.
    """
    if observed is None:
        return None
    if name == "daily_loss":
        # The only limit expressed as a negative floor: a session P&L of -2%
        # passes a -3% limit. Comparing it the same way as the ceilings would
        # invert the verdict on every losing day.
        return observed >= threshold
    if name == "net_exposure":
        # Directional tilt is bounded in both directions; a large short book
        # breaches it exactly as a large long one does.
        return abs(observed) <= threshold
    return observed <= threshold


def limit_rows(limits: RiskLimits, snapshot: BookSnapshot | None = None) -> list[LimitRow]:
    """Every enforced limit, in the order the engine checks them (§8).

    Args:
        snapshot: The current paper book. Omitted, every row reports its
            threshold with a null observation, which is what an unstarted
            account honestly looks like.
    """
    observations = (
        _observations(snapshot, limits) if snapshot is not None and snapshot.present else {}
    )
    rows = [
        LimitRow(
            name="order_notional",
            threshold=float(limits.max_order_notional),
            unit="INR",
            detail="single order size",
        ),
        LimitRow(
            name="position_size",
            threshold=float(limits.max_position_pct),
            unit="pct_nav",
            detail="resulting position as a fraction of NAV",
        ),
        LimitRow(
            name="price_band",
            threshold=float(limits.price_band_pct),
            unit="pct",
            detail="fat-finger guard: distance from last traded price",
        ),
        LimitRow(
            name="order_rate",
            threshold=float(limits.max_orders_per_minute),
            unit="count",
            detail="runaway-loop guard",
        ),
        LimitRow(
            name="open_orders",
            threshold=float(limits.max_open_orders),
            unit="count",
            detail="live orders resting at the venue",
        ),
        LimitRow(
            name="gross_exposure",
            threshold=float(limits.max_gross_exposure_pct),
            unit="pct_nav",
            detail="long plus short",
        ),
        LimitRow(
            name="net_exposure",
            threshold=float(limits.max_net_exposure_pct),
            unit="pct_nav",
            detail="directional tilt",
        ),
        LimitRow(
            name="cluster_concentration",
            threshold=float(limits.max_cluster_pct),
            unit="pct_nav",
            detail="correlated names count as one bet",
        ),
        LimitRow(
            name="daily_loss",
            threshold=float(limits.daily_loss_limit_pct),
            unit="pct",
            detail="session P&L halt threshold",
        ),
        LimitRow(
            name="liquidity",
            threshold=float(limits.max_adv_participation),
            unit="pct_adv",
            detail="order as a fraction of average daily value",
        ),
    ]
    return [
        row.model_copy(
            update={
                "observed": observations.get(row.name),
                "passed": _verdict(row.name, observations.get(row.name), row.threshold),
            }
        )
        for row in rows
    ]
