/**
 * What every column means — MASTER_PLAN §12.3.
 *
 * **One definition per term, used by every table.** A column headed `maxDD`
 * appears on the screener, the analytics screen and the report; if each spelled
 * out its own explanation they would drift, and two screens disagreeing about
 * what a number means is worse than neither explaining it.
 *
 * The definitions say what the number *is* and, where it matters, what it does
 * not tell you. A Sharpe ratio explained as "risk-adjusted return" teaches
 * nothing; explained as "assumes returns are symmetric, which trading
 * strategies routinely are not" it tells the reader when to distrust it.
 */

export const GLOSSARY: Record<string, string> = {
  // ── identity ───────────────────────────────────────────────────────────
  symbol:
    "Current NSE ticker. Not identity — symbols are reassigned after delisting, " +
    "so the system keys everything on the ISIN-based instrument_id underneath.",
  cluster:
    "Correlation group. Names that move together count as one bet against the " +
    "concentration limit. Blank means no cluster data exists yet.",

  // ── liquidity ──────────────────────────────────────────────────────────
  "adv cr":
    "Average daily traded value in crore rupees, median over the window. The " +
    "constraint on position size: an order that is a large share of this moves " +
    "the price against you before it fills.",
  adv: "Average daily traded value. Sets how much you can trade without moving the price.",

  // ── return and risk ────────────────────────────────────────────────────
  return:
    "Total return over the window, uncompounded across names. Says nothing " +
    "about the path — a +700% return with a −90% drawdown on the way is in here.",
  cagr: "Compound annual growth rate: the constant yearly rate that reaches the same end value.",
  vol:
    "Annualised standard deviation of daily returns. Symmetric by construction, " +
    "so it treats a 5% gain and a 5% loss as equally bad.",
  maxdd:
    "Worst peak-to-trough decline in the window. The number that decides whether " +
    "a strategy is survivable: a 2.0 Sharpe with a 60% drawdown gets abandoned at " +
    "the bottom by every human who has ever run one.",
  "current dd": "How far below the running high-water mark the price sits right now.",
  "% nav": "This position's market value as a share of total account equity.",

  // ── ratios ─────────────────────────────────────────────────────────────
  sharpe:
    "Mean excess return divided by volatility, annualised. Assumes returns are " +
    "roughly symmetric and independent — trading strategies routinely violate " +
    "both, which is why §5.4 requires the Deflated Sharpe instead.",
  sortino:
    "Sharpe, but dividing only by downside deviation. The fairer measure when a " +
    "strategy's upside is deliberately asymmetric, such as trend following.",
  calmar: "CAGR divided by the absolute maximum drawdown. Return per unit of worst pain.",

  // ── distribution ───────────────────────────────────────────────────────
  skew:
    "Asymmetry of the return distribution. Negative means many small gains and " +
    "rare large losses — the shape a Sharpe ratio hides.",
  kurtosis:
    "Fatness of the tails. A normal distribution is 3; higher means extreme days " +
    "happen far more often than a volatility figure implies.",
  var5: "The daily loss exceeded on the worst 5% of days. Says nothing about how bad the rest are.",
  cvar5: "Average loss on those worst 5% of days. This is the number VaR leaves out.",
  "tail ratio": "Size of the right tail against the left. Below 1 means losses reach further than gains.",
  hit: "Share of periods with a positive return. Not a quality measure — trend following wins under half the time and pays anyway.",

  // ── process ────────────────────────────────────────────────────────────
  hurst:
    "Long-memory exponent. 0.5 is a random walk; below is mean-reverting, above " +
    "is trending. Computed from variance of lagged differences, because the " +
    "classical R/S estimator returns ~1.0 for any random walk and is wrong.",
  process:
    "Whether the price series is stationary. UNIT_ROOT means it wanders with no " +
    "level to return to, so a mean-reversion strategy has nothing to fade. " +
    "STATIONARY means it does revert.",
  flags:
    "Warnings on this name: an implausible Sharpe usually means a missed corporate " +
    "action, and a fat left tail means small gains until a large loss.",

  // ── factor research ────────────────────────────────────────────────────
  ic:
    "Information Coefficient: rank correlation between the signal today and the " +
    "return that followed. 0.03 is a real edge in equities; 0.10 usually means a leak.",
  ir: "IC divided by its own standard deviation — how consistent the edge is, not how large.",
  t: "How many standard errors the mean IC sits from zero. Above 3 is hard to explain as luck.",
  horizon: "How far forward the return is measured, in trading sessions.",
  sessions: "Number of trading sessions contributing to the statistic.",
  names: "Number of instruments in the cross-section after liquidity and history filters.",
  turnover: "Share of the book that changes per session. Multiplied by the round trip, it is the cost of the edge.",
  spread: "Return difference between the top and bottom quintile, before costs.",
  monotonic: "Whether the quantile returns rise in order. One lucky extreme bucket is not a signal.",

  // ── risk limits ────────────────────────────────────────────────────────
  observed:
    "What the book currently shows for this limit. An em dash means the limit is " +
    "checked per order and has no value at rest, not that it is zero.",
  threshold: "The level the risk engine enforces on every order.",
  used: "Observed as a share of the threshold. Red when the limit is breached.",

  // ── book ───────────────────────────────────────────────────────────────
  qty: "Shares held. Negative is short.",
  avg: "Average price paid, excluding costs.",
  last: "Most recent close from the panel. Falls back to average price when the name did not trade.",
  unrealised: "Mark-to-market profit on open positions, not yet realised by a sale.",
  costs: "Total charges on the fill: STT, exchange and SEBI fees, stamp duty, GST, depository charges and slippage.",
  price: "Price the fill executed at, before costs. A decision on bar T fills on bar T+1, so this is not the price the signal saw.",
  side: "BUY or SELL.",
  time: "When the fill was applied, in UTC. The system stores every timestamp in UTC and converts only for display.",
  limit: "Which risk rule this row reports. The engine checks all ten on every order.",

  // ── reconciliation ─────────────────────────────────────────────────────
  instrument:
    "The instrument the two records disagree about, by instrument_id rather " +
    "than ticker — a rename must never look like a break.",
  kind: "What kind of disagreement: a quantity, a price, or a position the other side does not have.",
  internal: "What this system believes it holds.",
  broker: "What the broker says you hold. When these differ and nobody can explain why, trading halts (§9).",
  diff: "Internal minus broker. Any non-zero value is unexplained until a human explains it.",
  state: "Where the order reached in its lifecycle.",
};

/**
 * Definition for a column heading, or undefined when there is none.
 *
 * Matched case-insensitively on the visible text so a header renders as it
 * always did and the lookup is not a second thing to keep in sync.
 */
export function describe(heading: string): string | undefined {
  return GLOSSARY[heading.trim().toLowerCase()];
}
