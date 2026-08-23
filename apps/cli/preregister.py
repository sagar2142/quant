"""Pre-registered research questions — MASTER_PLAN §5.1, M6/M7.

    python -m apps.cli.preregister --list
    python -m apps.cli.preregister --register

**Read this file before registering anything in it.** These are claims about
how markets work, made in your name, and §5.1 exists precisely so that they
cannot be edited after the results arrive. Registration writes them to the
database with a CHECK constraint on the mechanism and a trigger on the trial
counter; changing your mind afterwards means abandoning the hypothesis on the
record, not quietly rewording it.

**Why none of these are the sixteen factors already in the library.** Every
factor in `quant.research.factors` has been scored: its IC, spread, turnover
and cost verdict are known. Registering those now with "pre-committed" success
criteria would be back-registration wearing pre-registration's clothes — the
counter would read fifteen and the evidence behind it would be worthless. §5.1
says *before any backtest runs*, and that is not a formality: the whole value
of the M6/M7 gate is that the criteria were fixed while the answer was still
unknown.

So every question below is one the system has **not** tested. Each is
computable from the daily OHLCV panel already in the lake — several use the
`trades` column, which no existing factor touches — so none of them is blocked
on data that does not exist yet.

**The mechanisms are drafts.** They are written to the 80-character floor with
a real causal story and a named counterparty, but they are my reasoning, not
yours. Anything you do not actually believe should be rewritten or deleted
before it is registered. A hypothesis you cannot defend is worse than one fewer
hypothesis.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from engine.experiments.registry import Hypothesis
from engine.experiments.repository import ExperimentRepository
from ops.db import optional_connection

RULE = "─" * 78

#: The panel runs 2019-01-01 to 2026-07-31. Development takes the first five
#: years, validation the next eighteen months, and the final year is locked
#: (§5.3) — touched once per strategy, ever, through the repository's UNIQUE
#: constraint rather than through good intentions.
DEV = (date(2019, 1, 1), date(2023, 12, 31))
VAL = (date(2024, 1, 1), date(2025, 6, 30))
TEST = (date(2025, 7, 1), date(2026, 7, 31))


def question(
    statement: str,
    mechanism: str,
    prediction: str,
    success: dict[str, object],
    kill: dict[str, object],
) -> Hypothesis:
    """One pre-registered question, on the standard three-way split."""
    return Hypothesis(
        statement=statement,
        economic_mechanism=mechanism,
        prediction=prediction,
        success_criteria=success,
        kill_criteria=kill,
        dev_start=DEV[0],
        dev_end=DEV[1],
        val_start=VAL[0],
        val_end=VAL[1],
        test_start=TEST[0],
        test_end=TEST[1],
    )


#: Success criteria are deliberately uniform and modest: a 21-day IC of 0.02
#: with a t-statistic above 3, and a quantile spread that survives the round
#: trip the cost model actually charges. Uniformity is the point — a threshold
#: tuned per idea is a threshold tuned to the answer.
def standard_success(ic: float = 0.02, horizon: int = 21) -> dict[str, object]:
    return {
        f"ic_{horizon}d": f">= {ic}",
        "t_stat": ">= 3.0",
        "net_spread_after_costs": "> 0",
    }


def standard_kill(ic: float = 0.005, horizon: int = 21) -> dict[str, object]:
    return {
        f"ic_{horizon}d": f"< {ic}",
        "or_net_spread": "<= 0",
        "or_sign": "opposite to prediction",
    }


CATALOGUE: tuple[Hypothesis, ...] = (
    question(
        "Overnight returns and intraday returns carry opposite cross-sectional signals on NSE",
        "Overnight moves price information that arrived while the exchange was shut, set by a "
        "thin book of participants willing to hold risk to the open; intraday moves are largely "
        "liquidity provision to that flow. The counterparty to overnight momentum is the market "
        "maker who must clear the opening imbalance and demands compensation for it.",
        "Overnight-return momentum has positive 21d IC; intraday-return momentum has negative IC",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Average trade size predicts cross-sectional returns on NSE equities",
        "Volume divided by trade count separates institutional accumulation from retail churn. "
        "Large average tickets indicate a patient buyer working an order over days, and the "
        "counterparty is the retail seller who does not observe that footprint and prices the "
        "stock as though the flow were transient.",
        "Top-quintile average trade size outperforms bottom by a positive net spread at 21 days",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Trade-count shocks signal retail attention and precede short-horizon reversal",
        "A spike in transactions without a matching spike in traded value is many small orders, "
        "which is retail attention rather than institutional repositioning. Attention-driven "
        "buying pushes price beyond fundamentals and the counterparty is the liquidity provider "
        "who absorbs it and is paid as attention fades.",
        "High trade-count shock predicts negative 5-day forward returns",
        standard_success(horizon=5),
        standard_kill(horizon=5),
    ),
    question(
        "Divergence between range volatility and close-to-close volatility predicts reversal",
        "When the Parkinson high-low estimator greatly exceeds close-to-close volatility, the "
        "session churned intraday and closed near where it opened. That churn is uninformed "
        "two-sided flow, and the counterparty harvesting it is whoever provides depth through "
        "the day rather than whoever holds the position overnight.",
        "High range-to-close volatility ratio predicts positive forward returns over 5 days",
        standard_success(horizon=5),
        standard_kill(horizon=5),
    ),
    question(
        "Opening gaps mean-revert intraday on NSE equities",
        "Gaps are formed by orders queued overnight from participants who cannot monitor the "
        "market, concentrating imbalance into the opening auction. Liquidity providers who take "
        "the other side require a price concession that unwinds through the session as patient "
        "flow arrives, which is a payment for immediacy rather than a forecast.",
        "Larger overnight gaps predict opposite-signed intraday returns on the same session",
        standard_success(horizon=1),
        standard_kill(horizon=1),
    ),
    question(
        "Momentum consistency predicts returns beyond total momentum on NSE",
        "The same twelve-month return earned smoothly and earned in one jump are different "
        "signals. A smooth path attracts less arbitrage attention because it never triggers a "
        "screen, so the underreaction persists; the counterparty is the investor who rebalances "
        "on headline moves and never notices the steady one.",
        "Fraction of positive months within the lookback adds IC over raw 12-1 momentum",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Price acceleration predicts returns before the momentum level does",
        "Acceleration is the change in trailing return, so it turns before the level does. If "
        "underreaction to news is gradual, the inflection in flow appears in the second "
        "derivative first, and the counterparty is the trend follower whose entry rule fires "
        "only once the level itself has moved.",
        "Momentum acceleration has positive 21d IC after controlling for momentum level",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Distance from the 200-day moving average is a distinct anchor from the 52-week high",
        "Both are anchoring effects but the reference point differs and so does who watches it. "
        "The 200-day average is the institutional trend filter; the 52-week high is the retail "
        "salience anchor. If the two are separately priced, the counterparty differs and the "
        "signals should not be redundant.",
        "Distance from the 200d MA has positive IC and correlates below 0.6 with 52w proximity",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Volatility of volatility carries a cross-sectional risk premium on NSE",
        "Investors dislike not knowing how risky a position will be, distinct from disliking "
        "risk itself. A name whose volatility is itself unstable is harder to size, so holders "
        "demand compensation; the counterparty is the investor with a fixed risk budget who must "
        "shed it precisely when the estimate moves against them.",
        "High vol-of-vol names earn a positive premium over low vol-of-vol names at 21 days",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Short-term reversal computed on residual returns beats reversal on raw returns",
        "Raw reversal mixes name-specific liquidity provision with market-wide reversal, and "
        "only the first is a payment for absorbing an imbalance. Stripping beta isolates the "
        "part where the counterparty is a seller needing immediacy in that specific name rather "
        "than an index-level flow.",
        "Residual 5-day reversal has higher IC than raw 5-day reversal on the same universe",
        standard_success(horizon=5),
        standard_kill(horizon=5),
    ),
    question(
        "Short-term reversal is stronger in high-turnover names",
        "Reversal is compensation for supplying liquidity, so it should scale with how much "
        "liquidity was demanded. In names where turnover is high the imbalance being absorbed is "
        "larger, and the counterparty paying for immediacy is more numerous, so the payment "
        "should be visibly larger.",
        "Reversal IC within the top turnover tercile exceeds that in the bottom tercile",
        standard_success(horizon=5),
        standard_kill(horizon=5),
    ),
    question(
        "Cross-sectional return dispersion times the profitability of cross-sectional strategies",
        "Dispersion measures how much difference there is between names to harvest at all. When "
        "every stock moves together the cross-section holds no information and any long-short "
        "book is paying costs for noise. This is a timing claim about when to deploy, not a "
        "claim about which names to hold.",
        "Factor IC in the following month rises with this month's cross-sectional dispersion",
        {"ic_of_ic": ">= 0.15", "t_stat": ">= 2.5", "regime_split": "monotonic across terciles"},
        {"ic_of_ic": "< 0.05", "or_sign": "opposite to prediction"},
    ),
    question(
        "The illiquidity premium survives a double sort on size",
        "Illiquidity and small size are heavily confounded, and a raw illiquidity sort may be "
        "paying the size premium under another name. Double-sorting asks whether anyone is "
        "actually compensated for bearing illiquidity itself, where the counterparty is the "
        "investor who needs to exit a thin name in a hurry.",
        "Illiquidity retains positive IC within size terciles, not only across them",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Downside-to-upside volatility asymmetry is priced on NSE equities",
        "Investors pay for protection against losses more than they pay for exposure to gains, "
        "so a name whose volatility is concentrated on the downside should trade at a discount. "
        "The counterparty is the investor selling that protection implicitly by holding the "
        "asymmetric name through the drawdown.",
        "High downside-to-upside semi-deviation ratio predicts positive forward returns",
        standard_success(),
        standard_kill(),
    ),
    question(
        "Proximity to the 52-week low is a distinct effect from proximity to the 52-week high",
        "The disposition effect makes holders reluctant to realise losses near lows, which is a "
        "different behaviour from the anchoring that slows adjustment near highs. If the two are "
        "separate biases with separate counterparties, the low-proximity signal should not be "
        "the mirror image of the high-proximity one.",
        "52-week-low proximity has independent IC, correlating below 0.5 with high proximity",
        standard_success(),
        standard_kill(),
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-register research questions (§5.1)")
    parser.add_argument("--list", action="store_true", help="Print the catalogue and exit")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Write them to the database. Read them first — this is a commitment.",
    )
    return parser.parse_args(argv)


def show(hypothesis: Hypothesis, index: int) -> None:
    print(f"\n  {index}. {hypothesis.statement}")
    print(f"     mechanism : {hypothesis.economic_mechanism[:150]}...")
    print(f"     predicts  : {hypothesis.prediction}")
    print(f"     succeeds  : {hypothesis.success_criteria}")
    print(f"     killed by : {hypothesis.kill_criteria}")


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not (args.list or args.register):
        print("give --list to read them, or --register to commit them")
        return 1

    print()
    print(RULE)
    print(f"PRE-REGISTERED QUESTIONS — {len(CATALOGUE)} of the 15 the M6/M7 gate wants")
    print(RULE)
    print("\n  None of these has been tested. The sixteen factors already in the")
    print("  library are deliberately absent: their results are known, and a")
    print("  hypothesis registered after the answer is not a hypothesis.")

    for index, hypothesis in enumerate(CATALOGUE, start=1):
        show(hypothesis, index)

    if not args.register:
        print(f"\n{RULE}")
        print("  Read them, edit what you do not believe, then: --register")
        print()
        return 0

    with optional_connection() as connection:
        if connection is None:
            print("\n  database unreachable — nothing registered")
            return 1
        repository = ExperimentRepository(connection)
        for hypothesis in CATALOGUE:
            repository.ensure_hypothesis(hypothesis)
        connection.commit()

    print(f"\n  registered {len(CATALOGUE)} hypotheses")
    print("  Each is OPEN until a gauntlet run resolves it. The gate wants 90%")
    print("  of them rejected — that is the expected outcome, not a failure.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(run())
