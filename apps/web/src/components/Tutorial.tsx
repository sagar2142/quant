/**
 * Tutorial screen — MASTER_PLAN §12.6.
 *
 * **What a new operator is missing is not button locations — it is the order
 * of operations.** This system is built to reject ideas, and someone who
 * starts at the backtester will read a rising equity curve as a discovery
 * rather than as the first of twelve questions. The tutorial therefore teaches
 * the *loop*, cheapest stage first, and says at each step what a failure there
 * means.
 *
 * Shown automatically on first visit and reachable afterwards from the nav, so
 * it is a reference rather than a one-time gate.
 */

import { useState } from "react";

interface Step {
  n: number;
  title: string;
  cost: string;
  what: string;
  command?: string;
  screen?: string;
  failure: string;
}

const STEPS: Step[] = [
  {
    n: 1,
    title: "Get data",
    cost: "~40 min cold, seconds daily",
    what:
      "NSE bhavcopy: every security that traded, every session. Because a delisted " +
      "stock stays in the files up to its last day, unioning the archive gives a " +
      "universe that includes the failures — survivorship-bias-free, from free files.",
    command: "python -m apps.cli.ingest_nse --start 2019-01-01",
    failure:
      "Interrupted? Re-run it. Sessions already stored are skipped, so a backfill resumes.",
  },
  {
    n: 2,
    title: "Check the data",
    cost: "seconds",
    what:
      "Gaps, stale prices, impossible bars, unexplained jumps. Run it after every " +
      "ingest, before trusting any number computed from it.",
    command: "python -m apps.cli.quality",
    failure:
      "A CRITICAL finding exits non-zero. Bad data does not raise an exception later — " +
      "it produces a plausible result, which is worse.",
  },
  {
    n: 3,
    title: "Score a signal",
    cost: "~3 min for all 16",
    what:
      "Does this idea predict anything at all? Information Coefficient at four " +
      "horizons, quintile buckets, turnover, and whether the spread survives the " +
      "22bp NSE round trip. Still the cheapest question, so it goes first.",
    screen: "Factors  (f)",
    command: "python -m apps.cli.factor --all",
    failure:
      "DIES ON COSTS is the normal outcome — 12 of 16 do. Kill it here and lose three " +
      "minutes instead of an afternoon.",
  },
  {
    n: 4,
    title: "Check for duplicates",
    cost: "~2 seconds",
    what:
      "Combining two factors that measure the same thing counts one effect twice and " +
      "calls it diversification. The 16-factor library is worth about 6 independent " +
      "bets; momentum and 52-week-high correlate at 0.65.",
    command: "python -m apps.cli.factor --overlap",
    failure:
      "High overlap is not fatal — it tells you which factor to drop before combining.",
  },
  {
    n: 5,
    title: "Combine what survived",
    cost: "~3 seconds",
    what:
      "Z-score, remove the overlap, weight by historical IC. A good composite scores " +
      "higher than any of its parts — that is the whole reason to combine. Pair a " +
      "survivor with a different effect: all four survivors are momentum variants, " +
      "and combining two of those is the stage-4 trap, not diversification.",
    command: "python -m apps.cli.factor --combine residual_momentum,reversal_5d",
    failure:
      "Weights are fitted on the sample they are scored against. In-sample by " +
      "construction, which is why this is a candidate and not evidence.",
  },
  {
    n: 6,
    title: "Backtest it",
    cost: "minutes",
    what:
      "Now ask what it earns, with real Indian costs: STT both legs, DP charges, " +
      "market impact. A decision on bar T fills on bar T+1, structurally — the " +
      "engine cannot be told otherwise.",
    command: "python -m apps.cli.backtest --strategy momentum --top 30",
    failure:
      "A strong IC routinely loses money. Fees on a 30-name book ran 5.3% of capital " +
      "over three years — the gap between predicting and earning is mostly costs.",
  },
  {
    n: 7,
    title: "Run the gauntlet",
    cost: "~48 backtests",
    what:
      "Twelve checks: deflated Sharpe against every trial you have ever run, " +
      "probability of backtest overfitting, walk-forward, universe dropout, a " +
      "placebo of random entries. All twelve must pass.",
    command: "python -m apps.cli.validate --top 30",
    failure:
      "Expect rejection. A 90% rejection rate means the gauntlet works; a strategy " +
      "that sails through on the first attempt usually means it is broken.",
  },
  {
    n: 8,
    title: "Paper trade it",
    cost: "6+ weeks of calendar",
    what:
      "One cycle per session against real prices, simulated fills. Measures drift " +
      "between what the backtest promised and what happens — the single most " +
      "informative number the system produces.",
    command: "python -m apps.cli.paper --top 30",
    failure:
      "Exit code 2 means HALTED on a reconciliation break. It survives restarts and " +
      "only you can clear it: --clear-halt.",
  },
  {
    n: 9,
    title: "Only then, consider capital",
    cost: "human judgement",
    what:
      "An executable version of the pre-live checklist: legal and tax position, " +
      "kill switch tested this month, drawdown ladder written down while calm, " +
      "capital small enough that losing all of it changes nothing.",
    command: "python -m apps.cli.readiness",
    failure:
      "Nothing trades by default. Four independent guards stand between this code " +
      "and a real order, and NEUTRON_LIVE_ENABLED is false.",
  },
];

const PRINCIPLES = [
  {
    title: "The system exists to reject ideas",
    body:
      "Its own momentum strategy fails the gauntlet — dropout Sharpe −0.51, parameter " +
      "retention 0.44 against a 0.6 floor. A validation suite that passes everything " +
      "is not a validation suite.",
  },
  {
    title: "Cheapest question first",
    body:
      "Six seconds, then minutes, then 48 backtests, then six weeks. Most ideas should " +
      "die at the first stage. Running them in the wrong order is how a month " +
      "disappears into one bad idea.",
  },
  {
    title: "Wrong numbers look plausible",
    body:
      "A missed corporate action turned a +1.2% three-year return into −49.4%. Cash " +
      "ETFs contaminated a low-volatility factor. Neither raised an exception. That is " +
      "why the data checks and the gauntlet exist.",
  },
  {
    title: "Decimal for money, float for statistics",
    body:
      "If a broker could disagree with you about a number, it is Decimal. A custom lint " +
      "rejects float anywhere in the ledger, costs, risk or execution paths.",
  },
];

function StepCard({ step }: { step: Step }) {
  return (
    <div className="tut-step">
      <div className="tut-step-head">
        <span className="tut-num">{step.n}</span>
        <span className="tut-title">{step.title}</span>
        <span className="tut-cost">{step.cost}</span>
      </div>
      <p className="tut-what">{step.what}</p>
      {step.screen ? (
        <p className="tut-screen">
          Console: <strong>{step.screen}</strong>
        </p>
      ) : null}
      {step.command ? <code className="tut-cmd">{step.command}</code> : null}
      <p className="tut-failure">{step.failure}</p>
    </div>
  );
}

export function Tutorial({ onDismiss }: { onDismiss?: () => void }) {
  const [tab, setTab] = useState<"loop" | "principles" | "screens">("loop");

  return (
    <div className="analytics">
      <div className="analytics-bar">
        {(["loop", "principles", "screens"] as const).map((id) => (
          <button
            key={id}
            type="button"
            className={`analytics-run ${tab === id ? "tut-on" : ""}`}
            onClick={() => setTab(id)}
          >
            {id === "loop" ? "The loop" : id === "principles" ? "Why" : "Screens"}
          </button>
        ))}
        <span className="analytics-spacer" />
        {onDismiss ? (
          <button type="button" className="analytics-run" onClick={onDismiss}>
            Start using it
          </button>
        ) : null}
      </div>

      <div className="analytics-body">
        {tab === "loop" ? (
          <>
            <p className="tut-lede">
              Nine stages, cheapest first. Each one is a filter, and most ideas
              should die at stage 3 — six seconds of work rather than six weeks.
            </p>
            {STEPS.map((s) => (
              <StepCard key={s.n} step={s} />
            ))}
          </>
        ) : null}

        {tab === "principles" ? (
          <>
            <p className="tut-lede">
              Four ideas the whole system is built around. They explain most of
              what looks unusual in it.
            </p>
            {PRINCIPLES.map((p) => (
              <div className="tut-step" key={p.title}>
                <div className="tut-step-head">
                  <span className="tut-title">{p.title}</span>
                </div>
                <p className="tut-what">{p.body}</p>
              </div>
            ))}
          </>
        ) : null}

        {tab === "screens" ? (
          <>
            <p className="tut-lede">
              Analysis screens answer research questions. Operations screens
              monitor a running book, and stay quiet until you paper trade.
            </p>
            <table className="grid">
              <thead>
                <tr>
                  <th>screen</th>
                  <th>key</th>
                  <th>answers</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["Factors", "f", "Does this signal predict anything? Does it survive costs?"],
                  ["Screener", "s", "Which of 3,268 names? By momentum, reversal, liquidity, volatility."],
                  ["Analytics", "a", "What is this name? Returns, risk, stationarity, correlation, clusters."],
                  ["Overview", "o", "What is the book doing right now?"],
                  ["Positions", "p", "What am I holding, and at what weight?"],
                  ["Blotter", "b", "What did I trade?"],
                  ["Risk", "r", "The ten limits the engine enforces on every order."],
                  ["Reconcile", "c", "Does the broker agree with me? A break halts trading."],
                ].map(([name, key, answers]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="mono">{key}</td>
                    <td className="text-secondary">{answers}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="analytics-note text-secondary">
              Operations screens are near-empty until the paper loop has run for
              a while. That is correct — they describe a running book, and one
              cycle of history is not one.
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
