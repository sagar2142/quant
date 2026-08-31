"""The backtest event loop — MASTER_PLAN §14, PDF §14.

The scientific instrument. Everything else in the research plane exists to feed
it or to interrogate its output, which is why the plan budgets more time here
than anywhere else.

**The central invariant, enforced structurally:**

    decision on bar T  →  fill on bar T+1

The strategy is handed a `MarketView` built from `receive_time <= decision_time`
and nothing else. The fill simulator is handed bar T+1 and nothing else. There
is no code path by which the decision bar's prices can reach the fill, so the
commonest source of fake backtest profit (§7.6) is unreachable rather than
merely discouraged.

**Corporate actions apply to positions**, on their ex-date, before valuation —
a split multiplies your share count and halves your average price, exactly as
it does to a real holding (`data.corpactions`).

**Determinism.** No wall-clock reads, no unseeded randomness, no dict-ordering
dependence. Two runs over the same data version with the same spec produce
byte-identical equity curves, which is the M3 gate.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import numpy.typing as npt
import polars as pl

from core.clock import as_decision_time
from core.instruments import Instrument, InstrumentId
from data.corpactions.actions import ActionType, CorporateActionBook
from engine.accounting import Portfolio
from engine.backtest.context import (
    BacktestConfig,
    BacktestResult,
    MarketModel,
    RunState,
    validate_history,
)
from engine.backtest.execution import execute_order
from engine.backtest.fills import FillModel
from engine.backtest.sizing import OrderPlanner, SizingConfig
from engine.costs.model import CostModel
from quant.strategies.base import MarketView, Strategy, partition_by_instrument

__all__ = ["BacktestEngine"]


def _to_decimal(value: float) -> Decimal:
    """float64 price -> Decimal, without inheriting binary representation error.

    Via str, deliberately: Decimal(0.1) is 0.1000000000000000055511151231257827,
    while Decimal(str(0.1)) is exactly 0.1 (§14.1.2).
    """
    return Decimal(str(value))


class BacktestEngine:
    """Bar-driven simulator.

    Args:
        strategy: Produces target weights. Never sees the portfolio.
        market: Costs, fills, instrument master and corporate actions.
        config: Sizing and turnover controls.
    """

    def __init__(
        self,
        strategy: Strategy,
        market: MarketModel,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.market = market
        self.config = config or BacktestConfig()
        self.planner = OrderPlanner(
            market.instruments,
            SizingConfig(
                rebalance_threshold=self.config.rebalance_threshold,
                min_order_value=self.config.min_order_value,
                cost_headroom=self.config.cost_headroom,
            ),
        )

    @property
    def cost_model(self) -> CostModel:
        return self.market.cost_model

    @property
    def fill_model(self) -> FillModel:
        return self.market.fill_model

    @property
    def instruments(self) -> dict[InstrumentId, Instrument]:
        return self.market.instruments

    @property
    def actions(self) -> CorporateActionBook:
        return self.market.actions

    def run(
        self,
        history: pl.DataFrame,
        universe: tuple[InstrumentId, ...] | None = None,
    ) -> BacktestResult:
        """Replay `history` bar by bar.

        Args:
            history: Long format — event_time, receive_time, instrument_id,
                open, high, low, close, volume.
            universe: Tradable names. Defaults to everything in `history`.

        Raises:
            ValueError: if required columns are missing or history is empty.
        """
        validate_history(history)
        history = history.sort(["event_time", "instrument_id"])
        timestamps = history["event_time"].unique().sort().to_list()

        if universe is None:
            universe = tuple(sorted(history["instrument_id"].unique().to_list()))

        portfolio = Portfolio(
            cash=self.config.initial_cash,
            margin_allowance=self.config.margin_allowance,
        )
        result = BacktestResult(
            equity_curve=pl.DataFrame(),
            trades=pl.DataFrame(),
            final_portfolio=portfolio,
            config=self.config,
            strategy_fingerprint=self.strategy.spec.fingerprint(),
        )

        state = RunState(portfolio=portfolio, result=result)
        lookback = self.strategy.spec.lookback

        # Cumulative last-seen close per name. Valuation reads THIS, never the
        # single session's cross-section: a held name that stops trading — a
        # suspension, a rename, a delisting — keeps its last traded mark, which
        # is exactly what a real ledger does. Falling back to average price
        # instead silently erases the position's entire unrealised P&L from the
        # curve and re-books it the day the mistake is noticed.
        last_marks: dict[InstrumentId, Decimal] = {}

        # Split once for the whole run. `MarketView.series` is called once per
        # name per bar, and filtering the full frame each time made a backtest
        # quadratic in its own length — 78% of the runtime was polars `collect`.
        # The partition is unfiltered; each view applies its own cutoff.
        partition = partition_by_instrument(history)

        # Stop one short: the final bar can never be an execution bar, so it can
        # never be a decision bar either.
        for index in range(len(timestamps) - 1):
            decision_ts = timestamps[index]
            execution_ts = timestamps[index + 1]
            result.bars_processed += 1

            self._apply_corporate_actions(portfolio, decision_ts, execution_ts)

            marks = self._marks(history, decision_ts)
            last_marks.update(marks)

            # Charged before the bar is valued, so the cost of carrying a short
            # lands on the session it was carried through. One session per bar:
            # a book held without trading still pays, which is the whole point
            # of pricing time rather than transactions.
            self._charge_borrow(portfolio, last_marks)

            # Recorded *before* this bar's orders execute, and that ordering is
            # the whole point. Orders placed here fill on bar T+1, and appending
            # afterwards booked the resulting position into the row stamped T
            # while marking it at T's close — a price from before the trade. A
            # buy filled at an open of 130 against a close of 100 showed a 23%
            # loss on the bar *preceding* the trade. The fill now lands in the
            # row for T+1, valued at T+1's close, where it happened.
            state.equity.append(self._equity_row(decision_ts, portfolio, last_marks))
            if index + 1 < lookback:
                continue

            # Between rebalances the book is held, not re-decided. Skipping
            # the strategy entirely rather than planning and discarding: a
            # threshold-filtered no-op still pays the planner's rounding, and
            # over a long run those add up to a position the signal never
            # asked for.
            if (index - lookback) % max(1, self.config.rebalance_every) != 0:
                continue

            view = self._build_view(history, decision_ts, universe, partition)
            targets = self.strategy(view)

            equity = self._safe_equity(portfolio, last_marks)
            # Sizing still uses the session's own marks: a name with no bar
            # today cannot be traded today, and pricing an order off a stale
            # close would be an order at a price that does not exist.
            orders = self.planner.plan(portfolio, targets.weights, marks, equity)
            result.orders_generated += len(orders)

            execution_slice = history.filter(pl.col("event_time") == execution_ts)
            for instrument_id, quantity in orders:
                if execute_order(
                    self, state, instrument_id, quantity, execution_slice, execution_ts
                ):
                    result.orders_filled += 1

        # Value the book on the final bar so the curve ends where the data does.
        # This is also where the last execution bar's fills are recorded, since
        # the loop stops one short of it.
        if timestamps:
            last_marks.update(self._marks(history, timestamps[-1]))
            state.equity.append(self._equity_row(timestamps[-1], portfolio, last_marks))

        result.equity_curve = pl.DataFrame(state.equity) if state.equity else pl.DataFrame()
        result.trades = pl.DataFrame(state.trades) if state.trades else pl.DataFrame()
        result.final_portfolio = portfolio
        return result

    # ── internals ───────────────────────────────────────────────────────────

    def _build_view(
        self,
        history: pl.DataFrame,
        decision_ts: datetime,
        universe: tuple[InstrumentId, ...],
        partition: dict[InstrumentId, tuple[pl.DataFrame, npt.NDArray[np.datetime64]]]
        | None = None,
    ) -> MarketView:
        """Everything observable at the decision point, and nothing else.

        Filters on `receive_time`, not `event_time`: a bar that closed at 15:30
        but published at 18:00 is not observable at 15:30 (§3.3).

        Args:
            partition: The whole history split by instrument, built once for
                the run. Passed through to the view so `series` is a lookup
                rather than a scan of the entire frame; the view applies the
                same `receive_time` cutoff to it, so what a strategy can see is
                unchanged.
        """
        decision_time = as_decision_time(decision_ts)
        observable = history.filter(pl.col("receive_time") <= decision_time)
        return MarketView(
            as_of=decision_time,
            history=observable,
            universe=universe,
            partition=partition,
        )

    def _charge_borrow(self, portfolio: Portfolio, marks: dict[InstrumentId, Decimal]) -> None:
        """Deduct one session of borrow on every open short.

        Valued at last-seen marks, like everything else in the loop: a short in
        a name that stopped printing still has to be borrowed.

        A long-only run does no arithmetic here at all — the loop body never
        executes, because nothing is short — so this cannot change any result
        already measured.
        """
        shorts = {
            instrument_id: position.market_value(marks[instrument_id])
            for instrument_id, position in portfolio.open_positions().items()
            if position.is_short and instrument_id in marks
        }
        if not shorts:
            return
        charge = self.market.borrow.session_charge(shorts)
        if charge > 0:
            portfolio.apply_funding(charge)

    @staticmethod
    def _marks(history: pl.DataFrame, timestamp: datetime) -> dict[InstrumentId, Decimal]:
        rows = history.filter(pl.col("event_time") == timestamp)
        return {
            row["instrument_id"]: _to_decimal(row["close"])
            for row in rows.select("instrument_id", "close").to_dicts()
        }

    def _apply_corporate_actions(
        self, portfolio: Portfolio, previous_ts: datetime, current_ts: datetime
    ) -> None:
        """Apply anything with an ex-date in (previous, current]."""
        for instrument_id in list(portfolio.open_positions()):
            for action in self.actions.effective_between(instrument_id, previous_ts, current_ts):
                if action.action_type.changes_share_count:
                    portfolio.apply_split(instrument_id, action.ratio)
                elif action.action_type is ActionType.DIVIDEND:
                    portfolio.apply_dividend(instrument_id, action.cash_per_share)

    @staticmethod
    def _safe_equity(portfolio: Portfolio, marks: dict[InstrumentId, Decimal]) -> Decimal:
        """Equity from cumulative last-seen marks.

        `Portfolio.equity` deliberately raises on a missing mark; this
        tolerates one. The average-price fallback survives only as a guard for
        a position whose instrument never printed a bar — which cannot happen
        to a position acquired through this engine, since the fill itself came
        from a bar.
        """
        total = portfolio.cash
        for instrument_id, position in portfolio.open_positions().items():
            price = marks.get(instrument_id, position.average_price)
            total += position.market_value(price)
        return total

    @staticmethod
    def _equity_row(
        timestamp: datetime, portfolio: Portfolio, marks: dict[InstrumentId, Decimal]
    ) -> dict[str, object]:
        """One curve row, valued off cumulative last-seen marks."""
        position_value = Decimal(0)
        for instrument_id, position in portfolio.open_positions().items():
            price = marks.get(instrument_id, position.average_price)
            position_value += position.market_value(price)
        return {
            "event_time": timestamp,
            "equity": float(portfolio.cash + position_value),
            "cash": float(portfolio.cash),
            "positions": len(portfolio.open_positions()),
            "fees_paid": float(portfolio.fees_paid),
        }
