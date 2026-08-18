"""One paper trading cycle — MASTER_PLAN §20, §M9.

The loop that was missing: every component below already existed — planner,
risk engine, paper broker, ledger, reconciliation — and nothing connected them.
This module is that connection, and only that. It holds no strategy logic (the
caller supplies target weights) and no persistence (the caller owns state),
because the same cycle must eventually run against a live adapter, where both
of those concerns move elsewhere.

**The order of operations is the contract:**

    1. mark the book and update the peak    (drawdown before decisions)
    2. plan deltas from target weights      (sells before buys, funded)
    3. risk-check every order               (fail closed, §8)
    4. submit survivors to the broker
    5. apply the broker's fills             (never our own assumptions)
    6. reconcile our book against theirs    (§9 — every cycle, no exceptions)

Step 5 matters more than it looks: the portfolio is updated from
`broker.fills_since(marker)`, not from the orders we submitted. In paper the
two coincide; against a real venue they will not — partial fills, rejects
after accept, fills arriving out of order — and a loop that trusts its own
submissions instead of the venue's fills reconciles clean right up until the
day it matters.

**A reconciliation break halts the account** (§9). The report says so via
`should_halt`, and the CLI translates that into refusing to run further cycles
until a human clears it. An unexplained break means the next order would be
sized against a book that is wrong by an unknown amount.

**Daily-cadence honesty:** with one cycle per session, `day_start_equity` is
the cycle's opening equity, so the intraday daily-loss check cannot trip in
paper. That protection is real only live; here the drawdown ladder does the
equivalent job across sessions. Written down so nobody mistakes the paper run
for evidence that the daily-loss limit works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from core.clock import utc_now
from core.instruments import Instrument, InstrumentId
from core.orders import OrderType, Side
from engine.accounting import Fill, Portfolio
from engine.backtest.sizing import OrderPlanner
from engine.costs.model import CostModel, TradeContext
from trading.execution.broker import BrokerAdapter, BrokerError
from trading.execution.orders import Order, TradingMode
from trading.reconcile.positions import ReconciliationReport, reconcile_positions
from trading.risk.engine import RiskEngine, RiskVerdict
from trading.risk.limits import PortfolioState, ProposedOrder

__all__ = ["CycleInputs", "CycleReport", "PaperSession"]


@dataclass(frozen=True)
class CycleInputs:
    """What one cycle needs from the outside world.

    Marks and ADV arrive as data, not as a feed handle, so the cycle is a pure
    function of its inputs and a test can replay any market it likes.
    """

    session: date
    #: Target weights from the strategy, as fractions of NAV.
    weights: dict[InstrumentId, Decimal]
    #: Latest observable close per instrument. Also the fill reference price.
    marks: dict[InstrumentId, Decimal]
    #: Trailing average daily traded value, for the liquidity limit.
    adv: dict[InstrumentId, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockedOrder:
    """An order the risk engine refused, kept for the record."""

    instrument_id: InstrumentId
    quantity: Decimal
    verdict: RiskVerdict

    def format(self) -> str:
        reasons = "; ".join(self.verdict.reasons) or "blocked"
        return f"  BLOCKED {self.instrument_id} qty {self.quantity}: {reasons}"


@dataclass
class CycleReport:
    """Everything one cycle did, in the order it did it."""

    session: date
    opening_equity: Decimal
    closing_equity: Decimal
    peak_equity: Decimal

    planned: int = 0
    blocked: list[BlockedOrder] = field(default_factory=list)
    submitted: int = 0
    fills_applied: int = 0
    fees_paid: Decimal = Decimal(0)
    #: Orders the broker refused. Distinct from risk blocks: these reached the
    #: venue and came back, which in paper signals a plumbing bug.
    broker_rejects: list[str] = field(default_factory=list)

    reconciliation: ReconciliationReport | None = None
    fill_marker: str | None = None

    @property
    def should_halt(self) -> bool:
        return self.reconciliation is not None and self.reconciliation.should_halt

    def format(self) -> str:
        lines = [
            f"cycle {self.session}: equity {self.opening_equity:,.2f} "
            f"-> {self.closing_equity:,.2f} (peak {self.peak_equity:,.2f})",
            f"  planned {self.planned}, blocked {len(self.blocked)}, "
            f"submitted {self.submitted}, fills {self.fills_applied}, "
            f"fees {self.fees_paid:,.2f}",
        ]
        lines.extend(b.format() for b in self.blocked)
        lines.extend(f"  BROKER REJECT {r}" for r in self.broker_rejects)
        if self.reconciliation is not None:
            state = "clean" if self.reconciliation.is_clean else "BREAKS FOUND"
            lines.append(f"  reconciliation: {state}")
            lines.extend(b.format() for b in self.reconciliation.breaks)
        if self.should_halt:
            lines.append("  ** HALT: unexplained break — no further cycles until cleared **")
        return "\n".join(lines)


class PaperSession:
    """Runs cycles for one paper account.

    Args:
        instruments: The tradeable universe, keyed by id.
        cost_model: Fees applied to paper fills. The broker itself reports zero
            fees (§20); costs come from the same model the backtest used, which
            is exactly what makes paper-versus-backtest drift attributable to
            *market* differences rather than accounting ones.
        risk: The independent risk engine. Owned by the caller so its kill
            switch outlives any single cycle.
        broker: Usually `PaperBroker`; anything satisfying the adapter works,
            which is the point — this same class must run live unchanged.
    """

    def __init__(
        self,
        instruments: dict[InstrumentId, Instrument],
        cost_model: CostModel,
        risk: RiskEngine,
        broker: BrokerAdapter,
        strategy_id: str = "paper",
    ) -> None:
        self.instruments = instruments
        self.cost_model = cost_model
        self.risk = risk
        self.broker = broker
        self.strategy_id = strategy_id
        self.planner = OrderPlanner(instruments)

    # ── the cycle ───────────────────────────────────────────────────────────

    def run_cycle(
        self,
        portfolio: Portfolio,
        peak_equity: Decimal,
        inputs: CycleInputs,
        fill_marker: str | None = None,
    ) -> CycleReport:
        """One complete decision-to-reconciliation pass. Mutates `portfolio`."""
        opening_equity = portfolio.equity(inputs.marks)
        peak = max(peak_equity, opening_equity)

        report = CycleReport(
            session=inputs.session,
            opening_equity=opening_equity,
            closing_equity=opening_equity,
            peak_equity=peak,
            fill_marker=fill_marker,
        )

        deltas = self.planner.plan(portfolio, inputs.weights, inputs.marks, opening_equity)
        report.planned = len(deltas)

        # Signed notional of orders already sent this cycle. The portfolio does
        # not move until fills are applied, which happens after this loop — so
        # without this, thirty orders are each judged against the same stale
        # book while their sum breaches gross exposure. An order in flight
        # consumes risk budget the moment it is sent, in paper and live alike.
        in_flight: dict[InstrumentId, Decimal] = {}
        for instrument_id, delta in deltas:
            price = inputs.marks[instrument_id]
            state = self._risk_state(portfolio, inputs, opening_equity, peak, in_flight)
            verdict = self.risk.check(self._proposed(instrument_id, delta, price), state)
            if not verdict.allowed:
                report.blocked.append(BlockedOrder(instrument_id, delta, verdict))
                continue
            if self._submit(instrument_id, delta, price, report):
                notional = delta * price * self.instruments[instrument_id].multiplier
                in_flight[instrument_id] = in_flight.get(instrument_id, Decimal(0)) + notional

        report.fill_marker = self._apply_fills(portfolio, inputs, report)
        report.closing_equity = portfolio.equity(inputs.marks)
        report.peak_equity = max(peak, report.closing_equity)
        report.reconciliation = reconcile_positions(portfolio, self.broker.positions())
        return report

    # ── steps ───────────────────────────────────────────────────────────────

    def _proposed(
        self, instrument_id: InstrumentId, delta: Decimal, price: Decimal
    ) -> ProposedOrder:
        instrument = self.instruments[instrument_id]
        return ProposedOrder(
            strategy_id=self.strategy_id,
            instrument_id=instrument_id,
            quantity=delta,
            price=price,
            multiplier=instrument.multiplier,
        )

    def _risk_state(
        self,
        portfolio: Portfolio,
        inputs: CycleInputs,
        opening_equity: Decimal,
        peak: Decimal,
        in_flight: dict[InstrumentId, Decimal],
    ) -> PortfolioState:
        """Snapshot for the risk engine, rebuilt before every order.

        Held positions and in-flight orders are merged: exposure the venue may
        create at any moment is exposure, whether or not the fill has landed.
        """
        notionals = {
            instrument_id: position.market_value(inputs.marks[instrument_id])
            for instrument_id, position in portfolio.positions.items()
            if not position.is_flat and instrument_id in inputs.marks
        }
        for instrument_id, notional in in_flight.items():
            notionals[instrument_id] = notionals.get(instrument_id, Decimal(0)) + notional
        return PortfolioState(
            equity=portfolio.equity(inputs.marks),
            cash=portfolio.cash,
            peak_equity=peak,
            day_start_equity=opening_equity,
            positions=notionals,
            open_orders=0,  # paper fills instantly; nothing rests at the venue
            orders_this_minute=len(in_flight),
            last_prices=inputs.marks,
            adv=inputs.adv,
        )

    def _submit(
        self,
        instrument_id: InstrumentId,
        delta: Decimal,
        price: Decimal,
        report: CycleReport,
    ) -> bool:
        order = Order(
            strategy_id=self.strategy_id,
            instrument_id=instrument_id,
            side=Side.BUY if delta > 0 else Side.SELL,
            quantity=abs(delta),
            order_type=OrderType.MARKET,
            mode=TradingMode.PAPER,
            decision_time=utc_now(),
        )
        try:
            self.broker.submit(order, reference_price=price)
        except BrokerError as exc:
            # Recorded, not raised: one refused order must not strand the rest
            # of the rebalance. In paper any entry here is a plumbing bug.
            report.broker_rejects.append(f"{instrument_id}: {exc}")
            return False
        report.submitted += 1
        return True

    def _apply_fills(
        self, portfolio: Portfolio, inputs: CycleInputs, report: CycleReport
    ) -> str | None:
        """Move the book by what the broker says happened, and only that."""
        marker = report.fill_marker
        for broker_fill in self.broker.fills_since(marker):
            instrument = self.instruments[broker_fill.instrument_id]
            costs = self.cost_model.cost(
                TradeContext(
                    instrument=instrument,
                    side=broker_fill.side,
                    quantity=broker_fill.quantity,
                    price=broker_fill.price,
                    adv_value=inputs.adv.get(broker_fill.instrument_id, Decimal(0)),
                )
            )
            portfolio.apply_fill(
                Fill(
                    instrument_id=broker_fill.instrument_id,
                    side=broker_fill.side,
                    quantity=broker_fill.quantity,
                    price=broker_fill.price,
                    costs=costs,
                    event_time=utc_now(),
                    multiplier=instrument.multiplier,
                )
            )
            report.fills_applied += 1
            report.fees_paid += costs.total
            marker = broker_fill.broker_fill_id
        return marker
