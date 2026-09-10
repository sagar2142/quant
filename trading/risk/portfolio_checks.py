"""Checks that read the whole book, not just the order — MASTER_PLAN §8.

Layer 2. A per-order check can be answered from the order alone; these cannot.
Whether a position may grow depends on what else is held in the same
correlation group, and whether it may be opened at all depends on what the
company has announced.

Free functions taking `limits` rather than methods, so the arithmetic can be
exercised without constructing an engine, and so the engine module stays short
enough to read in one sitting.
"""

from __future__ import annotations

from decimal import Decimal

from trading.risk.limits import PortfolioState, ProposedOrder, RiskCheck, RiskLimits

__all__ = ["cluster_check", "event_check"]


def event_check(order: ProposedOrder, state: PortfolioState, limits: RiskLimits) -> RiskCheck:
    """Do not add risk into a scheduled event.

    An earnings print is the one move whose *timing* is known in advance
    and whose size is not. Every volatility estimate this system makes is
    built from history, and history is mostly sessions where nothing was
    scheduled — so a forecast for the day a company reports understates a
    distribution known in advance to be bimodal. Sizing a fresh position
    against it is the mistake, and it is not one the strategy can see: a
    momentum score has no idea a name reports on Thursday.

    **Only increases are blocked.** Reducing or closing into an event is
    always allowed. A rule that blocked it would trap a book in the
    position it was trying to leave, which is the opposite of a control.
    """
    window = limits.event_blackout_days
    days = order.days_to_results
    current = state.positions.get(order.instrument_id, Decimal(0))
    # Reducing means the order moves the position toward zero. Opening from
    # flat counts as an increase; so does adding to either side.
    increasing = current == 0 or (current > 0) == (order.signed_notional > 0)

    if not increasing:
        return RiskCheck(
            "event_blackout",
            passed=True,
            observed=Decimal(days) if days is not None else None,
            threshold=Decimal(window),
            message="reducing into an event is always allowed",
        )
    if days is None:
        # Unmeasured, and reported as such. `observed=None` is what the
        # console renders as an em dash: this order was not checked against
        # a calendar, rather than checked and found clear.
        return RiskCheck(
            "event_blackout",
            passed=True,
            observed=None,
            threshold=Decimal(window),
            message="no calendar for this name; the event was not checked",
        )
    return RiskCheck(
        "event_blackout",
        passed=days > window,
        observed=Decimal(days),
        threshold=Decimal(window),
        message=f"sessions until announced results (blackout inside {window})",
    )


def cluster_check(order: ProposedOrder, state: PortfolioState, limits: RiskLimits) -> RiskCheck:
    """Correlated names count as one bet.

    Ten positions in correlated PSU banks is a single bet with ten tickers.
    A gross-exposure limit sees diversification that is not there; this
    check sees the bet.
    """
    current = state.clusters.get(order.cluster, Decimal(0))
    resulting = abs(current + order.signed_notional) / state.equity
    return RiskCheck(
        "cluster_concentration",
        passed=resulting <= limits.max_cluster_pct,
        observed=resulting,
        threshold=limits.max_cluster_pct,
        # The label carries its own namespace — "corr:" or "industry:" — so
        # a breach says which grouping decided rather than leaving the
        # reader to guess which of the two produced it.
        message=f"group '{order.cluster or 'none'}' as a fraction of NAV",
    )
