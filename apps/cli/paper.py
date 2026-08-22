"""Paper trading cycle — MASTER_PLAN §20, §M9.

    # once per session, after the bhavcopy lands (a cron/scheduled task):
    python -m apps.cli.ingest_nse --start 2026-08-01
    python -m apps.cli.paper --top 30

One invocation is one cycle: read the latest session from the panel, produce
target weights, size, risk-check, submit to the paper broker, apply fills,
reconcile, persist. The process then *exits* — daily-frequency NSE trading
needs a scheduled job, not a daemon, and a job that exits cannot leak state,
wedge overnight, or need a supervisor.

**The six-week M9 clock is measured by `paper_equity.ndjson`**, one line per
cycle. Deleting the state directory restarts that clock; the state loader
refuses corrupted files for the same reason.

Exit codes: 0 cycle ran; 1 could not run (no data, no universe); 2 HALTED —
either a reconciliation break or a prior halt that has not been cleared.
A scheduler must treat 2 as "page the human", not "retry".
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

import polars as pl

from apps.cli.backtest import build_universe, load_panel, nse_instrument
from core.clock import UTC, as_decision_time, utc_now
from core.config import settings
from core.instruments import Instrument, InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.accounting import Portfolio
from engine.costs.india import NseEquityCostModel
from ops.alerts import AlertRouter
from ops.routing import build_router, describe_channels
from quant.strategies.base import MarketView, Strategy
from quant.strategies.baselines import CrossSectionalMomentum
from trading.execution.broker import BrokerPosition, PaperBroker
from trading.paper.session import CycleInputs, CycleReport, PaperSession
from trading.paper.state import PaperState, PaperStateStore, StateCorruptError
from trading.risk.engine import RiskEngine

#: Trailing window for average daily traded value, matching the risk limits.
ADV_WINDOW = 20

HALTED_EXIT = 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper trading cycle")
    parser.add_argument("--top", type=int, default=30, help="Universe size")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--skip", type=int, default=5)
    parser.add_argument("--cash", type=Decimal, default=Decimal(1_000_000))
    parser.add_argument(
        "--gross",
        type=Decimal,
        default=Decimal("0.9"),
        help=(
            "Gross exposure. Default 0.9, not 1.0: a 9-name book at gross 1.0 "
            "is 11.1%% per name, and the independent risk engine caps a single "
            "name at 10%% of NAV — the strategy must fit inside the limit, "
            "because the limit will not move for the strategy (§8)."
        ),
    )
    parser.add_argument("--lake", default=None)
    parser.add_argument("--state-dir", type=Path, default=Path("paper"))
    parser.add_argument(
        "--clear-halt",
        action="store_true",
        help="Acknowledge a reconciliation halt and continue. A human decision.",
    )
    return parser.parse_args(argv)


def latest_view(history: pl.DataFrame, universe: tuple[InstrumentId, ...]) -> MarketView:
    """Everything observable as of the latest session's publication."""
    last = history["event_time"].max()
    assert isinstance(last, datetime)
    as_of = as_decision_time(datetime.combine(last.date(), time(23, 59), tzinfo=UTC))
    return MarketView(as_of=as_of, history=history, universe=universe)


def latest_marks(history: pl.DataFrame) -> dict[InstrumentId, Decimal]:
    latest = history.sort("event_time").group_by("instrument_id").agg(pl.col("close").last())
    return {
        InstrumentId(i): Decimal(str(c))
        for i, c in zip(latest["instrument_id"], latest["close"], strict=True)
    }


def trailing_adv(history: pl.DataFrame) -> dict[InstrumentId, Decimal]:
    """Mean traded value over the trailing window, per name."""
    sessions = history["event_time"].unique().sort().tail(ADV_WINDOW)
    window = history.filter(pl.col("event_time").is_in(sessions.implode()))
    value = (
        window.with_columns((pl.col("close") * pl.col("volume")).alias("traded"))
        .group_by("instrument_id")
        .agg(pl.col("traded").mean())
    )
    return {
        InstrumentId(i): Decimal(str(round(v, 2)))
        for i, v in zip(value["instrument_id"], value["traded"], strict=True)
        if v is not None and v > 0
    }


def load_or_create_state(store: PaperStateStore, strategy_id: str, cash: Decimal) -> PaperState:
    if store.exists():
        return store.restore()
    return PaperState(
        strategy_id=strategy_id,
        portfolio=Portfolio(cash=cash),
        peak_equity=cash,
    )


def rebuild_broker(state: PaperState, instruments: dict[InstrumentId, Instrument]) -> PaperBroker:
    """A broker whose positions mirror the persisted book.

    `PaperBroker` is in-memory and this process is one cycle long, so the
    venue's positions are re-seeded from state. Injected via the sanctioned
    test hook: this is the one legitimate caller outside tests, and it exists
    precisely so reconciliation starts from agreement and any divergence within
    the cycle is the cycle's own doing.
    """
    broker = PaperBroker(instruments=instruments)
    for instrument_id, position in state.portfolio.positions.items():
        if position.is_flat:
            continue
        broker.inject_position(
            BrokerPosition(
                instrument_id=instrument_id,
                quantity=position.quantity,
                average_price=position.average_price,
            )
        )
    return broker


def build_strategy(args: argparse.Namespace) -> Strategy:
    return CrossSectionalMomentum(
        lookback_bars=args.lookback,
        skip_bars=args.skip,
        top_fraction=Decimal("0.3"),
        gross=args.gross,
    )


def already_traded(state: PaperState, session: date) -> bool:
    """One cycle per session, enforced.

    The scheduler will double-fire eventually — DST shifts, manual re-runs,
    retry logic. Trading the same bhavcopy twice would double the book's
    turnover for the day and quietly poison the drift analysis.
    """
    return state.last_session is not None and session <= state.last_session


def resume_state(store: PaperStateStore, args: argparse.Namespace) -> PaperState | None:
    """Load state and enforce the persisted halt. None means do not proceed.

    A persisted halt survives restarts — that is what makes it a halt. Only the
    explicit flag releases it, and the release is saved immediately so it is
    recorded even if this cycle then fails for other reasons.
    """
    try:
        state = load_or_create_state(store, "paper-momentum", args.cash)
    except StateCorruptError as exc:
        print(exc)
        return None

    if state.cycles and state.last_cycle_at is None:
        print("state carries cycles but no timestamp — inspect it before continuing")
        return None

    if state.halted:
        if not args.clear_halt:
            print(f"HALTED since a previous cycle: {state.halt_reason}")
            print("Investigate, then re-run with --clear-halt to acknowledge.")
            return None
        print(f"halt cleared by operator (was: {state.halt_reason})")
        state.clear_halt()
        store.save(state)
    return state


def run_cycle_for(
    state: PaperState,
    history: pl.DataFrame,
    universe: tuple[InstrumentId, ...],
    args: argparse.Namespace,
) -> CycleReport:
    """Assemble the market and run one cycle. Pure assembly, no persistence."""
    symbols = dict(history.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}

    strategy = build_strategy(args)
    weights = strategy(latest_view(history, universe))
    print(f"strategy      : {strategy.name} {strategy.spec.parameters}")

    marks = latest_marks(history)
    session_stamp = history["event_time"].max()
    assert isinstance(session_stamp, datetime)

    cycle = PaperSession(
        instruments=instruments,
        cost_model=NseEquityCostModel(),
        risk=RiskEngine(),
        broker=rebuild_broker(state, instruments),
        strategy_id=state.strategy_id,
    )
    return cycle.run_cycle(
        state.portfolio,
        state.peak_equity,
        CycleInputs(
            session=session_stamp.date(),
            weights=weights.weights,
            marks={k: v for k, v in marks.items() if k in instruments},
            adv=trailing_adv(history),
        ),
        # Broker is rebuilt fresh each run, so its fill log starts empty and
        # the persisted marker must not be applied to it.
        fill_marker=None,
    )


def persist_outcome(
    store: PaperStateStore,
    state: PaperState,
    report: CycleReport,
    alerts: AlertRouter,
) -> int:
    """Save everything the next cycle needs, then translate halt to exit code.

    **State is written before the alert is sent.** If the alerting API is
    unreachable the halt must still survive into the next run; the reverse
    order would let a network failure lose the halt entirely.
    """
    state.peak_equity = report.peak_equity
    state.cycles += 1
    state.last_cycle_at = utc_now()
    state.last_session = report.session
    state.fill_marker = report.fill_marker
    if report.should_halt and report.reconciliation is not None:
        state.engage_halt(
            f"reconciliation breaks on {report.session}: "
            + "; ".join(b.format().strip() for b in report.reconciliation.critical_breaks)
        )
    store.save(state)
    store.append_equity(
        report.session, report.closing_equity, state.portfolio.cash, state.portfolio.fees_paid
    )
    # After the state, for the same reason the state comes before the alert: a
    # failure here loses a blotter row, which is recoverable, while a failure
    # before `save` would lose the cycle's position changes, which is not.
    #
    # Written without a symbol on purpose. The `instrument_id` is the identity
    # (§3.3), and the console resolves today's ticker from the panel when it
    # renders — the same path positions take. Freezing a symbol into the log
    # would preserve whatever the name was called on the day it traded and
    # quietly disagree with every other screen after a rename.
    for fill in report.filled:
        store.append_fill(report.session, fill)
    print(f"state saved   : {store.state_path} (cycle {state.cycles})")

    announce(alerts, state, report)

    if state.halted:
        print("HALTED — investigate the break, then re-run with --clear-halt")
        return HALTED_EXIT
    return 0


def announce(alerts: AlertRouter, state: PaperState, report: CycleReport) -> None:
    """Send whatever this cycle earned.

    A halt pages; a blocked order warns; an ordinary session files a summary.
    §M9 gates live trading on having been woken by an alert at least once, and
    that cannot happen if the only record is a log line nobody reads.

    Alert delivery never affects the exit code. An unreachable Telegram must not
    turn a clean session into a failure, and it must not turn a halted one into
    a success — the halt is already persisted above.
    """
    if report.should_halt and report.reconciliation is not None:
        breaks = report.reconciliation.critical_breaks
        alerts.reconciliation_break(
            len(breaks),
            f"session {report.session}: "
            + "; ".join(b.format().strip() for b in breaks)
            + "\nAccount is HALTED. No further cycles until --clear-halt.",
        )
        return

    if report.blocked:
        first = report.blocked[0]
        alerts.risk_breach(
            first.verdict.reasons[0] if first.verdict.reasons else "risk",
            Decimal(len(report.blocked)),
            Decimal(0),
        )

    alerts.daily_summary(
        pnl=report.closing_equity - report.opening_equity,
        positions=len(state.portfolio.open_positions()),
        drift=None,  # needs a matched backtest run; M9-M10
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = PaperStateStore(args.state_dir)
    alerts = build_router()
    print(describe_channels())

    state = resume_state(store, args)
    if state is None:
        # resume_state printed why. Halt and corruption both land here: the
        # scheduler's job is to stop, not to distinguish them.
        return HALTED_EXIT if store.exists() else 1

    panel = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    try:
        history = load_panel(panel)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2026-01-01")
        return 1
    if history.is_empty():
        print("panel is empty")
        return 1

    session_stamp = history["event_time"].max()
    assert isinstance(session_stamp, datetime)
    if already_traded(state, session_stamp.date()):
        print(
            f"session {session_stamp.date()} already traded "
            f"(last was {state.last_session}) — nothing to do"
        )
        return 0

    universe = build_universe(panel, args.top)
    if not universe:
        print("universe is empty — ingest more sessions")
        return 1

    report = run_cycle_for(state, history, universe, args)
    print(report.format())
    return persist_outcome(store, state, report, alerts)


if __name__ == "__main__":
    sys.exit(run())
