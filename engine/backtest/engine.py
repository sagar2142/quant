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

import polars as pl

from core.clock import as_decision_time
from core.instruments import Instrument, InstrumentId
from core.orders import Side
from data.corpactions.actions import ActionType, CorporateActionBook
from engine.accounting import Fill, Portfolio
from engine.backtest.context import (
    BacktestConfig,
    BacktestResult,
    MarketModel,
    RunState,
    validate_history,
)
from engine.backtest.fills import ExecutionBar, FillModel, NoLiquidityError
from engine.backtest.sizing import OrderPlanner, SizingConfig
from engine.costs.model import CostModel, TradeContext
from quant.strategies.base import MarketView, Strategy

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

        # Stop one short: the final bar can never be an execution bar, so it can
        # never be a decision bar either.
        for index in range(len(timestamps) - 1):
            decision_ts = timestamps[index]
            execution_ts = timestamps[index + 1]
            result.bars_processed += 1

            self._apply_corporate_actions(portfolio, decision_ts, execution_ts)

            marks = self._marks(history, decision_ts)
            last_marks.update(marks)

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

            view = self._build_view(history, decision_ts, universe)
            targets = self.strategy(view)

            equity = self._safe_equity(portfolio, last_marks)
            # Sizing still uses the session's own marks: a name with no bar
            # today cannot be traded today, and pricing an order off a stale
            # close would be an order at a price that does not exist.
            orders = self.planner.plan(portfolio, targets.weights, marks, equity)
            result.orders_generated += len(orders)

            execution_slice = history.filter(pl.col("event_time") == execution_ts)
            for instrument_id, quantity in orders:
                if self._execute(state, instrument_id, quantity, execution_slice, execution_ts):
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
    ) -> MarketView:
        """Everything observable at the decision point, and nothing else.

        Filters on `receive_time`, not `event_time`: a bar that closed at 15:30
        but published at 18:00 is not observable at 15:30 (§3.3).
        """
        decision_time = as_decision_time(decision_ts)
        observable = history.filter(pl.col("receive_time") <= decision_time)
        return MarketView(as_of=decision_time, history=observable, universe=universe)

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

    def _execute(
        self,
        state: RunState,
        instrument_id: InstrumentId,
        quantity: Decimal,
        execution_slice: pl.DataFrame,
        execution_ts: datetime,
    ) -> bool:
        """Fill one order into the execution bar. Returns whether it filled."""
        portfolio, result = state.portfolio, state.result
        rows = execution_slice.filter(pl.col("instrument_id") == instrument_id)
        if rows.is_empty():
            # The instrument did not trade this session. Counted separately: a
            # delisting is not a defect in our order logic.
            result.orders_no_market += 1
            return False

        row = rows.row(0, named=True)
        instrument = self.instruments[instrument_id]
        bar = ExecutionBar(
            instrument=instrument,
            open=_to_decimal(row["open"]),
            high=_to_decimal(row["high"]),
            low=_to_decimal(row["low"]),
            close=_to_decimal(row["close"]),
            volume=_to_decimal(row["volume"]),
        )
        side = Side.BUY if quantity > 0 else Side.SELL
        wanted = abs(quantity)

        if side is Side.BUY:
            wanted = self._affordable(portfolio, bar, self.fill_model.reference_price(bar), wanted)
            if wanted <= 0:
                result.orders_unfunded += 1
                return False

        try:
            simulated = self.fill_model.simulate(
                bar,
                side,
                wanted,
                allow_partial=self.config.allow_partial_fills,
            )
        except NoLiquidityError:
            # The bar could not absorb the order — zero volume, zero range, or
            # past the participation cap. Counted once, under liquidity. It is
            # not a rejection: nothing in our logic went wrong, the market was
            # simply not deep enough.
            result.liquidity_failures += 1
            return False

        quantity = simulated.quantity
        if side is Side.BUY:
            # Final trim against the *realised* fill price. The earlier check
            # used the fill model's reference price, and `simulate` then moved
            # it against us by the slippage. Without this the account overdraws
            # by exactly the slippage on the last order of a fully-invested
            # rebalance — which presents as a rejection rather than a bug.
            quantity = self._affordable(portfolio, bar, simulated.price, quantity)
            if quantity <= 0:
                result.orders_unfunded += 1
                return False

        costs = self.cost_model.cost(
            TradeContext(
                instrument=instrument,
                side=side,
                quantity=quantity,
                price=simulated.price,
                adv_value=bar.volume * bar.typical,
            )
        )
        fill = Fill(
            instrument_id=instrument_id,
            side=side,
            quantity=quantity,
            price=simulated.price,
            costs=costs,
            event_time=execution_ts,
            multiplier=instrument.multiplier,
        )

        try:
            realised = portfolio.apply_fill(fill)
        except Exception:  # noqa: BLE001 — insufficient cash is a rejection, not a crash
            result.orders_rejected += 1
            return False

        state.trades.append(
            {
                "event_time": execution_ts,
                "instrument_id": instrument_id,
                "side": side.value,
                "quantity": float(quantity),
                "price": float(simulated.price),
                "costs": float(costs.total),
                "realised_pnl": float(realised),
            }
        )
        return True

    def _affordable(
        self,
        portfolio: Portfolio,
        bar: ExecutionBar,
        price: Decimal,
        wanted: Decimal,
    ) -> Decimal:
        """Largest buy the account can fund at `price`.

        Delegates the arithmetic to the planner (§14.2). The important detail is
        that `cost_of` builds the *same* `TradeContext` the charge will use —
        including `adv_value`, which enables the square-root impact term. An
        estimate that omits impact under-charges by exactly the impact, and the
        order then overdraws by that amount.
        """
        instrument = bar.instrument
        adv_value = bar.volume * bar.typical

        def cost_of(quantity: Decimal, at_price: Decimal) -> Decimal:
            return self.cost_model.cost(
                TradeContext(
                    instrument=instrument,
                    side=Side.BUY,
                    quantity=quantity,
                    price=at_price,
                    adv_value=adv_value,
                )
            ).total

        return self.planner.affordable(portfolio, instrument, price, wanted, cost_of)

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
