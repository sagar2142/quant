"""Resolve pre-registered hypotheses against their own criteria — §5.1, §5.5.

    python -m apps.cli.resolve --dry-run
    python -m apps.cli.resolve --commit

**A hypothesis is closed by the criteria it was registered with, not by
judgement afterwards.** That is the entire point of writing kill criteria down
before the answer is known, and it only means something if closing is
mechanical. This applies each registered hypothesis's own `kill_criteria` and
`success_criteria` to the measured result and writes the verdict.

**Cheapest question first (§5.5).** Stage 3 — the factor lab — answers most of
them: a signal whose IC is below its kill threshold, or whose spread does not
survive its own turnover, is rejected without a backtest ever running. Only what
clears stage 3 is worth the gauntlet, which is the ordering the whole research
loop is built around.

**Stage 3 can reject and cannot confirm.** Clearing it is permission to build a
strategy, not evidence one works, and the gauntlet disagrees often enough to
matter: `ma200_distance` met every stage-3 threshold in this catalogue and was
then rejected at walk-forward. Confirmation is recorded by whoever runs the
gauntlet, against the run that produced it.

**Some questions this cannot answer yet**, and it says so rather than guessing.
Four of the fifteen need machinery the factor lab does not have — a same-session
gap study, a conditional double sort, a timing rather than selection claim.
Those stay OPEN. An unresolved hypothesis is honest; a hypothesis resolved on a
test that was never run is not, and it is exactly the corruption the M6/M7 gate
exists to detect.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import dataclass
from datetime import date

import polars as pl

from apps.cli.backtest import build_universe, load_panel, nse_instrument
from apps.cli.factor import ROUND_TRIP_COST
from apps.cli.runs import Panel, run_cadence
from core.config import settings
from core.instruments import InstrumentId
from data.store.bars import NoDataError
from data.store.panel import PanelStore
from engine.experiments.registry import HypothesisStatus
from engine.experiments.repository import ExperimentRepository
from ops.db import optional_connection
from quant.math.metrics.performance import cagr, sharpe_ratio
from quant.research.factors import (
    FORWARD_HORIZONS,
    Factor,
    FactorSpec,
    build_factor,
    prepare_panel,
)
from quant.research.ic import analyse_factor

RULE = "─" * 78

#: Which factor tests which registered question, keyed by the opening words of
#: the statement so a reworded mechanism does not silently unmap it.
#:
#: A `None` means the question is real and the factor lab cannot answer it.
#: Those are left OPEN: the alternative is resolving a hypothesis against a test
#: that was never run, which would put a verdict in the record with nothing
#: behind it.
#: The four questions the factor lab cannot express, and the study that can.
#: Each returns the statistic its hypothesis committed to, plus the direction
#: the hypothesis predicted, so the verdict is read off the same way.
STUDIED_BY: dict[str, tuple[str, str]] = {
    "Opening gaps mean-revert": ("gap_reversion", "negative"),
    "Short-term reversal is stronger": ("conditional_ic", "positive"),
    "Cross-sectional return dispersion times": ("dispersion_timing", "positive"),
    "The illiquidity premium survives": ("double_sorted_ic", "positive"),
}

#: The two questions that are about *portfolio construction* rather than about
#: ranking, and the cadence each one committed to.
#:
#: Neither can be answered by a cross-sectional IC, because neither is a claim
#: about which names to hold — both hold exactly the same names as a daily book
#: and differ only in how often that book is re-decided. Judging them on a
#: factor score would return the same number for both arms and resolve nothing.
#:
#: The window matters and is part of the mapping. The 63-session question was
#: registered before anything had been measured, so development data is
#: legitimate evidence for it. The 21-session question was not: it came out of
#: an exploratory sweep, its dev result is already known, and it was registered
#: as confirmatory on that basis. Judging it on dev data would be scoring an
#: answer against the data that produced it.
CADENCE_BY: dict[str, tuple[int, str]] = {
    "A signal is best traded on the horizon": (63, "dev"),
    "A monthly rebalance is the horizon": (21, "val"),
}

TESTED_BY: dict[str, Factor | None] = {
    "Overnight returns and intraday returns": Factor.OVERNIGHT_MOMENTUM,
    "Average trade size predicts": Factor.AVG_TRADE_SIZE,
    "Trade-count shocks signal": Factor.TRADE_COUNT_SHOCK,
    "Divergence between range volatility": Factor.RANGE_VOL_RATIO,
    "Opening gaps mean-revert": None,  # same-session; the lab scores forward horizons
    "Momentum consistency predicts": Factor.MOMENTUM_CONSISTENCY,
    "Price acceleration predicts": Factor.MOMENTUM_ACCELERATION,
    "Distance from the 200-day": Factor.MA200_DISTANCE,
    "Volatility of volatility carries": Factor.VOL_OF_VOL,
    "Short-term reversal computed on residual": Factor.RESIDUAL_REVERSAL,
    "Short-term reversal is stronger": None,  # conditional on turnover, not a sort
    "Cross-sectional return dispersion times": None,  # a timing claim, not selection
    "The illiquidity premium survives": None,  # double sort on size
    "Downside-to-upside volatility asymmetry": Factor.SEMI_DEVIATION_RATIO,
    "Proximity to the 52-week low": Factor.LOW_52W_PROXIMITY,
    "A signal is best traded on the horizon": None,  # cadence, not selection
    "A monthly rebalance is the horizon": None,  # cadence, not selection
}

#: The momentum configuration both cadence questions were registered about:
#: twelve-month lookback, one-month skip. Fixed rather than swept, because a
#: cadence question that also swept the lookback would be two questions.
LOOKBACK = 252
SKIP = 21

#: Below this, a window is too short for a 21- or 63-session cadence to have
#: rebalanced enough times to mean anything. Reported as OPEN rather than
#: rejected: too little data is not evidence against a claim.
MIN_CADENCE_SESSIONS = 250

#: How many names the cadence comparison trades. Matched to the gauntlet's
#: default so a hypothesis that clears here is handed to a stage that measures
#: the same book, not a wider or narrower one.
CADENCE_UNIVERSE = 200

#: Significance floor every hypothesis in the catalogue committed to. Uniform
#: on purpose: a threshold tuned per idea is a threshold tuned to the answer.
MIN_T_STAT = 3.0

#: The horizon each hypothesis committed to, read from its own success criteria.
#: A 5-day claim judged on a 21-day IC is unfalsifiable either way.
HORIZON_KEYS = {"ic_1d": 1, "ic_5d": 5, "ic_21d": 21, "ic_63d": 63}


def factor_for(statement: str) -> tuple[Factor | None, bool]:
    """The factor testing this statement, and whether it is mapped at all."""
    for prefix, factor in TESTED_BY.items():
        if statement.startswith(prefix):
            return factor, True
    return None, False


def committed_horizon(criteria: dict[str, object]) -> int:
    """The forward horizon this hypothesis committed to being judged on."""
    for key, horizon in HORIZON_KEYS.items():
        if key in criteria:
            return horizon
    return 21


def threshold(criteria: dict[str, object], key: str, fallback: float) -> float:
    """Numeric bound from a criterion written as text, e.g. `">= 0.02"`."""
    raw = str(criteria.get(key, "")).replace(">=", "").replace("<", "").strip()
    try:
        return float(raw)
    except ValueError:
        return fallback


@dataclass(frozen=True)
class Verdict:
    """One hypothesis, judged against the criteria it carries."""

    status: HypothesisStatus
    reason: str


def judge(  # noqa: PLR0913, PLR0917 - a hypothesis, its criteria, and the data to test it
    history: pl.DataFrame,
    factor: Factor,
    success: dict[str, object],
    kill: dict[str, object],
    min_adv: float,
    sessions: int,
) -> Verdict:
    """Measure the factor and apply this hypothesis's own thresholds.

    Kill criteria are checked as written: below the IC floor, or a spread that
    does not survive its own turnover, or a sign opposite to the prediction.
    Any one of them closes the question — which is what makes closing
    mechanical rather than a matter of opinion after the fact.
    """
    horizon = committed_horizon(success)
    horizons = tuple(sorted({*FORWARD_HORIZONS, horizon}))
    scored = build_factor(history, FactorSpec(factor, min_adv=min_adv, window=sessions), horizons)
    if scored.is_empty():
        return Verdict(HypothesisStatus.OPEN, "no names survived the filters")

    report = analyse_factor(scored, factor.value, horizons, horizon, 5)
    measured = next((h for h in report.horizons if h.horizon == horizon), None)
    if measured is None:
        return Verdict(HypothesisStatus.OPEN, f"horizon {horizon}d not measured")

    ic = measured.mean
    per_rebalance = min(1.0, report.turnover * report.quantile_horizon)
    net = report.spread - per_rebalance * ROUND_TRIP_COST
    reason = f"{factor.value}: IC({horizon}d) {ic:+.4f} t {measured.t_stat:+.2f} net {net:+.3%}"

    if abs(ic) < threshold(kill, f"ic_{horizon}d", 0.005) or net <= 0 or ic < 0:
        return Verdict(HypothesisStatus.REJECTED, reason)

    # **Stage 3 rejects; it does not confirm.** Meeting the success criteria
    # here means the signal predicts and clears its own costs, which is
    # permission to build a strategy rather than evidence one works (§5.5).
    # The gauntlet disagrees often enough to matter: `ma200_distance` cleared
    # every stage-3 threshold in this catalogue and was then rejected at
    # walk-forward, in-sample 2.77 collapsing to 0.69 out-of-sample. Confirming
    # here would have written a verdict the next stage contradicts.
    if (
        ic >= threshold(success, f"ic_{horizon}d", 0.02)
        and abs(measured.t_stat) >= MIN_T_STAT
        and net > 0
    ):
        return Verdict(HypothesisStatus.OPEN, f"{reason} — clears stage 3, needs the gauntlet")
    return Verdict(HypothesisStatus.OPEN, reason)


#: The registered split (§5.3). Duplicated from the pre-registration catalogue
#: rather than imported, because the windows a hypothesis was registered under
#: must not follow a later edit to that file — the whole point of §5.1 is that
#: the goalposts cannot move after the fact.
CADENCE_WINDOWS: dict[str, tuple[date, date]] = {
    "dev": (date(2019, 1, 1), date(2023, 12, 31)),
    "val": (date(2024, 1, 1), date(2025, 6, 30)),
}


def cadence_panel(span: pl.DataFrame, args: argparse.Namespace) -> Panel | None:
    """A tradable panel over one registered window.

    The universe is built from the whole store rather than from the window, so
    membership stays point-in-time correct: restricting it to names that traded
    inside the window would select on having survived it.
    """
    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    universe = build_universe(store, CADENCE_UNIVERSE)
    if not universe:
        return None
    symbols = dict(span.select("instrument_id", "symbol").unique().iter_rows())
    instruments = {InstrumentId(i): nse_instrument(i, symbols.get(i, i)) for i in universe}
    return Panel(history=span, instruments=instruments, universe=universe)


def run_cadence_study(  # noqa: PLR0913, PLR0917 - a hypothesis, its cadence, and the data
    history: pl.DataFrame,
    every: int,
    window: str,
    success: dict[str, object],
    kill: dict[str, object],
    args: argparse.Namespace,
) -> Verdict:
    """Two backtests, identical but for how often the book is re-decided.

    The comparison is the measurement. Sharpe alone would not resolve either
    hypothesis, because both predicted a fee saving *and* an improvement — a
    cadence that halved fees while halving return has not beaten a daily book,
    and one that improved Sharpe while paying the same fees did not do it for
    the reason claimed.

    Rejection is the default. A cadence that fails either arm of its own
    prediction is closed, and one that merely ties is closed too: "no worse
    than daily" was not what was registered, and treating a tie as support is
    how a null result becomes a finding.
    """
    start, end = CADENCE_WINDOWS[window]
    span = history.filter(
        (pl.col("event_time").dt.date() >= start) & (pl.col("event_time").dt.date() <= end)
    )
    sessions = span["event_time"].n_unique()
    if sessions < MIN_CADENCE_SESSIONS:
        return Verdict(HypothesisStatus.OPEN, f"{window} window has only {sessions} sessions")

    panel = cadence_panel(span, args)
    if panel is None:
        return Verdict(HypothesisStatus.OPEN, "universe is empty in this window")

    daily = run_cadence(panel, LOOKBACK, SKIP, 1)
    tested = run_cadence(panel, LOOKBACK, SKIP, every)
    if daily.returns.size == 0 or tested.returns.size == 0:
        return Verdict(HypothesisStatus.OPEN, "no equity curve — nothing traded")

    d_sharpe, t_sharpe = sharpe_ratio(daily.returns), sharpe_ratio(tested.returns)
    d_cagr, t_cagr = cagr(daily.returns), cagr(tested.returns)
    reason = (
        f"{every}d vs daily on {window} ({sessions} sessions): "
        f"sharpe {t_sharpe:+.2f} vs {d_sharpe:+.2f}, "
        f"cagr {t_cagr:+.2%} vs {d_cagr:+.2%}, "
        f"fees {tested.fees:,.0f} vs {daily.fees:,.0f}"
    )

    beat_sharpe = t_sharpe > d_sharpe
    beat_cagr = t_cagr > d_cagr
    halved_fees = tested.fees < daily.fees / 2

    if not (beat_sharpe and beat_cagr):
        return Verdict(HypothesisStatus.REJECTED, f"{reason} — did not beat daily")
    if not halved_fees:
        return Verdict(
            HypothesisStatus.REJECTED, f"{reason} — beat daily, but not by saving fees as claimed"
        )

    # Beating a daily book on one window is not permission to trade it. The
    # gauntlet is the next stage, exactly as for a factor that clears stage 3.
    _ = (success, kill)
    return Verdict(HypothesisStatus.OPEN, f"{reason} — clears its own criteria, needs the gauntlet")


def run_study(history: pl.DataFrame, name: str, predicted: str) -> Verdict:
    """Run one of the four studies and judge it against its own prediction.

    A study answers a claim rather than producing a signal, so the verdict is
    read off the direction and significance the hypothesis committed to. A
    result that is significant in the *opposite* direction is a rejection, not
    a near miss — the prediction was wrong, and saying so is the point.
    """
    from quant.research.studies import (  # noqa: PLC0415 - keeps the import local
        conditional_ic,
        dispersion_timing,
        double_sorted_ic,
        gap_reversion,
    )

    if name == "gap_reversion":
        prepped = prepare_panel(history, FactorSpec(Factor.REVERSAL_5D))
        result = gap_reversion(prepped)
    elif name == "conditional_ic":
        result = conditional_ic(history, FactorSpec(Factor.REVERSAL_5D), 5, "volume")
    elif name == "dispersion_timing":
        result = dispersion_timing(history, FactorSpec(Factor.MOMENTUM_12_1), 21)
    else:
        result = double_sorted_ic(history, FactorSpec(Factor.ILLIQUIDITY), 21, "volume")

    reason = f"{name}: {result.statistic:+.4f} t {result.t_stat:+.2f} n {result.observations}"
    if not result.is_significant:
        return Verdict(HypothesisStatus.REJECTED, f"{reason} — not significant")

    right_way = result.statistic < 0 if predicted == "negative" else result.statistic > 0
    if not right_way:
        return Verdict(HypothesisStatus.REJECTED, f"{reason} — sign opposite to the prediction")

    # A claim that survives its own test is not yet a strategy: none of these
    # produces a tradeable signal on its own, and confirming here would skip
    # every stage between a finding and a book.
    return Verdict(HypothesisStatus.OPEN, f"{reason} — holds, but is not yet a signal")


def resolve_one(
    history: pl.DataFrame,
    statement: str,
    success: dict[str, object],
    kill: dict[str, object],
    args: argparse.Namespace,
) -> Verdict | None:
    """The verdict for one statement, or None if it is not a catalogue entry.

    Two routes, because the catalogue asks two shapes of question: eleven are
    cross-sectional signals the factor lab scores, and four are claims that
    need a study of their own. Both are judged against what the hypothesis
    committed to; neither is judged by reading the number and deciding.
    """
    factor, mapped = factor_for(statement)
    if not mapped:
        return None
    if factor is not None:
        return judge(history, factor, success, kill, args.min_adv, args.sessions)

    cadence = next((v for k, v in CADENCE_BY.items() if statement.startswith(k)), None)
    if cadence is not None:
        return run_cadence_study(history, *cadence, success, kill, args)

    study = next((v for k, v in STUDIED_BY.items() if statement.startswith(k)), None)
    if study is None:
        return Verdict(HypothesisStatus.OPEN, "no test exists for this")
    return run_study(history, *study)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close hypotheses against their own criteria")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Write the verdicts. Without it, nothing is recorded.",
    )
    parser.add_argument("--min-adv", type=float, default=1e7)
    parser.add_argument("--sessions", type=int, default=0)
    parser.add_argument("--lake", default=None)
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    store = PanelStore(args.lake if args.lake is not None else settings.lake, venue="NSE")
    try:
        history = load_panel(store)
    except NoDataError as exc:
        print(f"{exc}\nRun: python -m apps.cli.ingest_nse --start 2019-01-01")
        return 1

    with optional_connection() as connection:
        if connection is None:
            print("database unreachable — nothing to resolve against")
            return 1
        repository = ExperimentRepository(connection)
        with connection.cursor() as cur:
            cur.execute(
                "SELECT hypothesis_id, statement, success_criteria, kill_criteria "
                "FROM hypotheses WHERE status = 'OPEN' ORDER BY created_at"
            )
            open_rows = cur.fetchall()

        print()
        print(RULE)
        print(f"RESOLVING {len(open_rows)} OPEN HYPOTHESES against their own criteria")
        print(RULE)
        print("\n  Stage 3 first. A signal below its kill threshold is rejected without")
        print("  a backtest ever running — which is the ordering the loop is built on.\n")

        verdicts: list[tuple[uuid.UUID, str, HypothesisStatus, str]] = []
        for hypothesis_id, statement, success, kill in open_rows:
            verdict = resolve_one(history, str(statement), success, kill, args)
            if verdict is None:
                print(f"  [ SKIP  ] {str(statement)[:52]:<54} not a pre-registered question")
                continue
            mark = {
                HypothesisStatus.CONFIRMED: "CONFIRM",
                HypothesisStatus.REJECTED: "REJECT ",
                HypothesisStatus.OPEN: " OPEN  ",
            }[verdict.status]
            print(f"  [{mark}] {str(statement)[:52]:<54} {verdict.reason}")
            if verdict.status is not HypothesisStatus.OPEN:
                verdicts.append((hypothesis_id, str(statement), verdict.status, verdict.reason))

        print()
        print(RULE)
        rejected = sum(1 for _, _, s, _ in verdicts if s is HypothesisStatus.REJECTED)
        confirmed = sum(1 for _, _, s, _ in verdicts if s is HypothesisStatus.CONFIRMED)
        print(
            f"  {rejected} rejected, {confirmed} confirmed, {len(open_rows) - len(verdicts)} left open"
        )
        if verdicts:
            rate = rejected / len(verdicts)
            print(f"  rejection rate {rate:.0%} of those resolved — the M6/M7 gate wants 90%")

        if not args.commit:
            print("\n  Nothing written. Re-run with --commit to record these.")
            print()
            return 0

        for hypothesis_id, _, status, _ in verdicts:
            repository.resolve_hypothesis(hypothesis_id, status)
        connection.commit()
        print(f"\n  recorded {len(verdicts)} verdicts")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(run())
