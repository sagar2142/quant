"""The independent risk engine — MASTER_PLAN §8, §21.

Every order passes through here before it reaches a venue, in backtest, paper
and live alike. The engine imports no strategy code and no backtest code
(§3.2), holds no opinion about whether a trade is *good*, and answers exactly
one question: **would this order breach a limit?**

**Fail closed.** An unrecognised instrument, a missing price, an unreadable
state — every one of those is a BLOCK, never an ALLOW. A risk engine that
permits what it cannot evaluate is not a risk engine.

**Every check runs, then the verdict is formed.** Short-circuiting at the first
breach would be marginally faster and would hide the other four things also
wrong with the order. Diagnosing an incident at 2am is much easier with the
complete list.

**The kill switch is deliberately trivial.** Halt new orders, record who and
why, require a manual release. The mechanism that stops everything must not
itself be complicated enough to fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from core.clock import utc_now
from trading.risk.limits import (
    DrawdownLadder,
    PortfolioState,
    ProposedOrder,
    RiskDecision,
    RiskLimits,
)

__all__ = ["KillSwitchEngagedError", "RiskCheck", "RiskEngine", "RiskVerdict"]


class KillSwitchEngagedError(RuntimeError):
    """The global halt is active. No order may pass while it is."""

    def __init__(self, reason: str, engaged_by: str, at: datetime) -> None:
        super().__init__(f"kill switch engaged by {engaged_by} at {at.isoformat()}: {reason}")


@dataclass(frozen=True)
class RiskCheck:
    """One limit, evaluated."""

    name: str
    passed: bool
    observed: Decimal | None = None
    threshold: Decimal | None = None
    message: str = ""

    def format(self) -> str:
        mark = "ok  " if self.passed else "FAIL"
        if self.observed is None:
            return f"    [{mark}] {self.name:<24} {self.message}"
        return (
            f"    [{mark}] {self.name:<24} {self.observed:>14,.2f} "
            f"vs {self.threshold:>14,.2f}  {self.message}"
        )


@dataclass(frozen=True)
class RiskVerdict:
    """The engine's answer, with the full reasoning attached."""

    decision: RiskDecision
    checks: tuple[RiskCheck, ...]
    #: Position scale imposed by the drawdown ladder, if any.
    scale: Decimal = Decimal(1)
    evaluated_at: datetime = field(default_factory=utc_now)

    @property
    def allowed(self) -> bool:
        return self.decision is RiskDecision.ALLOW

    @property
    def breaches(self) -> tuple[RiskCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)

    @property
    def reasons(self) -> list[str]:
        return [f"{c.name}: {c.message}" for c in self.breaches]

    def format(self) -> str:
        head = f"  {self.decision.value}"
        if self.scale != 1:
            head += f" (ladder scale {self.scale})"
        return "\n".join([head, *(c.format() for c in self.checks)])


class RiskEngine:
    """Evaluates orders against limits. Owns the kill switch.

    Args:
        limits: Thresholds. Owned here, never writable by a strategy.
        ladder: Pre-committed drawdown de-risking.
    """

    def __init__(
        self,
        limits: RiskLimits | None = None,
        ladder: DrawdownLadder | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.ladder = ladder or DrawdownLadder()
        self._killed = False
        self._kill_reason = ""
        self._kill_by = ""
        self._killed_at: datetime | None = None

    # ── kill switch (§8) ────────────────────────────────────────────────────

    @property
    def is_killed(self) -> bool:
        return self._killed

    def engage_kill(self, reason: str, engaged_by: str) -> None:
        """Halt all new orders.

        Both arguments are required: an unattributed halt with no stated cause
        is impossible to review afterwards, and the database CHECK enforces the
        same thing.
        """
        if not reason.strip():
            raise ValueError("kill switch requires a reason")
        if not engaged_by.strip():
            raise ValueError("kill switch requires an operator")
        self._killed = True
        self._kill_reason = reason
        self._kill_by = engaged_by
        self._killed_at = utc_now()

    def release_kill(self, released_by: str) -> None:
        """Resume trading. Manual only, deliberately.

        There is no automatic release and there should never be one: whatever
        engaged the switch needs a human to confirm it has been understood.
        """
        if not released_by.strip():
            raise ValueError("releasing the kill switch requires an operator")
        self._killed = False
        self._kill_reason = ""
        self._kill_by = ""
        self._killed_at = None

    def raise_if_killed(self) -> None:
        if self._killed:
            raise KillSwitchEngagedError(
                self._kill_reason, self._kill_by, self._killed_at or utc_now()
            )

    # ── evaluation ──────────────────────────────────────────────────────────

    def check(self, order: ProposedOrder, state: PortfolioState) -> RiskVerdict:
        """Evaluate one order against every limit.

        Returns a verdict rather than raising, so the caller can record the
        rejection and continue. Use `raise_if_killed` when a hard stop is
        wanted.
        """
        if self._killed:
            return RiskVerdict(
                decision=RiskDecision.BLOCK,
                checks=(
                    RiskCheck(
                        "kill_switch",
                        passed=False,
                        message=f"engaged by {self._kill_by}: {self._kill_reason}",
                    ),
                ),
                scale=Decimal(0),
            )

        checks = [
            *self._pre_trade_checks(order, state),
            *self._portfolio_checks(order, state),
            self._drawdown_check(state),
        ]
        scale = self.ladder.scale_for(state.drawdown)
        decision = RiskDecision.ALLOW if all(c.passed for c in checks) else RiskDecision.BLOCK
        return RiskVerdict(decision=decision, checks=tuple(checks), scale=scale)

    # ── layer 1 ─────────────────────────────────────────────────────────────

    def _pre_trade_checks(self, order: ProposedOrder, state: PortfolioState) -> list[RiskCheck]:
        limits = self.limits
        notional = order.notional
        checks = [
            RiskCheck(
                "order_notional",
                passed=notional <= limits.max_order_notional,
                observed=notional,
                threshold=limits.max_order_notional,
                message="single order size",
            ),
            RiskCheck(
                "order_rate",
                passed=state.orders_this_minute < limits.max_orders_per_minute,
                observed=Decimal(state.orders_this_minute),
                threshold=Decimal(limits.max_orders_per_minute),
                message="runaway-loop guard",
            ),
            RiskCheck(
                "open_orders",
                passed=state.open_orders < limits.max_open_orders,
                observed=Decimal(state.open_orders),
                threshold=Decimal(limits.max_open_orders),
                message="live order count",
            ),
        ]
        checks.append(self._price_band_check(order, state))
        checks.append(self._position_check(order, state))
        checks.append(self._liquidity_check(order, state))
        return checks

    def _price_band_check(self, order: ProposedOrder, state: PortfolioState) -> RiskCheck:
        """Fat-finger guard: reject a price far from the last trade.

        A missing last price is a BLOCK, not a pass. An order whose sanity
        cannot be established has not been established as sane.
        """
        last = state.last_prices.get(order.instrument_id)
        if last is None or last <= 0:
            return RiskCheck(
                "price_band",
                passed=False,
                message="no last price available; cannot verify the order is sane",
            )
        deviation = abs(order.price - last) / last
        return RiskCheck(
            "price_band",
            passed=deviation <= self.limits.price_band_pct,
            observed=deviation,
            threshold=self.limits.price_band_pct,
            message="distance from last traded price",
        )

    def _position_check(self, order: ProposedOrder, state: PortfolioState) -> RiskCheck:
        if state.equity <= 0:
            return RiskCheck("position_size", passed=False, message="equity is not positive")
        current = state.positions.get(order.instrument_id, Decimal(0))
        resulting = abs(current + order.signed_notional)
        pct = resulting / state.equity
        return RiskCheck(
            "position_size",
            passed=pct <= self.limits.max_position_pct,
            observed=pct,
            threshold=self.limits.max_position_pct,
            message="resulting position as a fraction of NAV",
        )

    def _liquidity_check(self, order: ProposedOrder, state: PortfolioState) -> RiskCheck:
        """Cap participation in the instrument's daily volume.

        Skipped, not failed, when ADV is unknown: a fresh listing has no
        history, and blocking every order in it would be wrong. The absence is
        reported so it is visible rather than assumed away.
        """
        adv = state.adv.get(order.instrument_id)
        if adv is None or adv <= 0:
            return RiskCheck(
                "liquidity", passed=True, message="ADV unknown; participation not checked"
            )
        participation = order.notional / adv
        return RiskCheck(
            "liquidity",
            passed=participation <= self.limits.max_adv_participation,
            observed=participation,
            threshold=self.limits.max_adv_participation,
            message="order as a fraction of average daily value",
        )

    # ── layer 2 ─────────────────────────────────────────────────────────────

    def _portfolio_checks(self, order: ProposedOrder, state: PortfolioState) -> list[RiskCheck]:
        if state.equity <= 0:
            return [RiskCheck("portfolio", passed=False, message="equity is not positive")]

        limits = self.limits
        gross = (state.gross_exposure + order.notional) / state.equity
        net = abs(state.net_exposure + order.signed_notional) / state.equity

        checks = [
            RiskCheck(
                "gross_exposure",
                passed=gross <= limits.max_gross_exposure_pct,
                observed=gross,
                threshold=limits.max_gross_exposure_pct,
                message="long plus short, as a fraction of NAV",
            ),
            RiskCheck(
                "net_exposure",
                passed=net <= limits.max_net_exposure_pct,
                observed=net,
                threshold=limits.max_net_exposure_pct,
                message="directional tilt",
            ),
            RiskCheck(
                "daily_loss",
                passed=state.day_pnl_pct > limits.daily_loss_limit_pct,
                observed=state.day_pnl_pct,
                threshold=limits.daily_loss_limit_pct,
                message="session P&L against the halt threshold",
            ),
        ]
        if order.cluster:
            checks.append(self._cluster_check(order, state))
        return checks

    def _cluster_check(self, order: ProposedOrder, state: PortfolioState) -> RiskCheck:
        """Correlated names count as one bet.

        Ten positions in correlated PSU banks is a single bet with ten tickers.
        A gross-exposure limit sees diversification that is not there; this
        check sees the bet.
        """
        current = state.clusters.get(order.cluster, Decimal(0))
        resulting = abs(current + order.signed_notional) / state.equity
        return RiskCheck(
            "cluster_concentration",
            passed=resulting <= self.limits.max_cluster_pct,
            observed=resulting,
            threshold=self.limits.max_cluster_pct,
            message=f"correlated group '{order.cluster}' as a fraction of NAV",
        )

    # ── layer 3 ─────────────────────────────────────────────────────────────

    def _drawdown_check(self, state: PortfolioState) -> RiskCheck:
        drawdown = state.drawdown
        halted = self.ladder.halts_at(drawdown)
        rung = self.ladder.rung_for(drawdown)
        detail = (
            f"ladder rung {rung.drawdown_pct} -> scale {rung.scale_to}"
            if rung
            else "above every rung"
        )
        return RiskCheck(
            "drawdown_ladder",
            passed=not halted,
            observed=drawdown,
            threshold=self.ladder.rungs[-1].drawdown_pct,
            message=detail,
        )
