/**
 * Backtest and gauntlet, on screen — MASTER_PLAN §5, §M3, §12.9.
 *
 * **The console could show research and start none of it.** Every other tab
 * displays something already computed; this is the first one that runs the two
 * things the system is actually for. Both take minutes, so both are jobs: fire
 * and poll, with the run list kept beside the result so a morning's work is
 * still readable at lunchtime.
 *
 * **The gauntlet's verdict is not a score.** Twelve checks, and all twelve must
 * pass — a candidate that clears eleven has not nearly passed, it has failed.
 * So the banner says PASSED or names the check that rejected it, and the table
 * below shows every check rather than stopping at the first failure: "fix this"
 * and "abandon this" look identical if you only ever see one red line.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { formatCount, formatPercent } from "../format";

type JobState = "queued" | "running" | "done" | "failed" | "cancelled";

interface JobRow {
  id: string;
  kind: string;
  label: string;
  state: JobState;
  progress: string;
  submittedAt: string;
  elapsedSeconds: number;
  error: string;
  result: Record<string, unknown> | null;
}

interface CurvePoint {
  t: string;
  equity: number;
}

interface BacktestResult {
  strategy: string;
  costModel: string;
  costMultiple: number;
  venue: string;
  universe: number;
  sessions: number;
  start: string;
  end: string;
  startEquity: number;
  endEquity: number;
  totalReturn: number;
  maxDrawdown: number;
  feesPaid: number;
  ordersGenerated: number;
  ordersFilled: number;
  ordersRejected: number;
  ordersNoMarket: number;
  ordersUnfunded: number;
  liquidityFailures: number;
  trades: number;
  corporateActions: boolean;
  actionsNote: string;
  curve: CurvePoint[];
}

interface CheckRow {
  test: string;
  passed: boolean;
  skipped: boolean;
  statistic: number | null;
  threshold: number | null;
  reason: string;
}

interface GauntletResult {
  label: string;
  venue: string;
  universe: number;
  periods: number;
  passed: boolean;
  firstFailure: string | null;
  sharpe: number;
  implausible: boolean;
  hypothesis: string | null;
  results: CheckRow[];
  neighbourhood: { label: string; sharpe: number }[];
  log: string;
}

function camel<T>(input: unknown): T {
  if (Array.isArray(input)) return input.map((v) => camel(v)) as unknown as T;
  if (input === null || typeof input !== "object") return input as T;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    out[key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase())] = camel(value);
  }
  return out as T;
}

/** An equity curve as a single path, scaled to its own range. */
function curvePath(points: CurvePoint[], width: number, height: number): string {
  if (points.length < 2) return "";
  const values = points.map((p) => p.equity);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  return points
    .map((point, index) => {
      const x = (index / (points.length - 1)) * width;
      const y = height - ((point.equity - low) / span) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
}

function elapsed(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

export function Lab({ apiBase = "/api" }: { apiBase?: string }) {
  const [mode, setMode] = useState<"backtest" | "gauntlet">("backtest");
  const [strategy, setStrategy] = useState("momentum");
  const [factor, setFactor] = useState("");
  const [factors, setFactors] = useState<string[]>([]);
  const [top, setTop] = useState(30);
  const [lookback, setLookback] = useState(60);
  const [costMultiple, setCostMultiple] = useState(1);
  const [actions, setActions] = useState(false);
  const [venue, setVenue] = useState("NSE");
  const [hypothesis, setHypothesis] = useState("");

  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<JobRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  //: Held in a ref so the poll interval reads the current selection without
  //: being torn down and rebuilt every time it changes.
  const selectedRef = useRef<string | null>(null);
  selectedRef.current = selected;

  useEffect(() => {
    void (async () => {
      const response = await fetch(`${apiBase}/lab/strategies`);
      if (response.ok) {
        const body = (await response.json()) as { factors: string[] };
        setFactors(body.factors);
      }
    })();
  }, [apiBase]);

  const refresh = useCallback(async () => {
    const response = await fetch(`${apiBase}/lab/jobs`);
    if (!response.ok) return;
    setJobs(camel<JobRow[]>(await response.json()));

    const id = selectedRef.current;
    if (!id) return;
    const one = await fetch(`${apiBase}/lab/jobs/${id}`);
    if (one.ok) setDetail(camel<JobRow>(await one.json()));
  }, [apiBase]);

  //: Two seconds. A backtest is minutes, so polling faster buys nothing but
  //: request noise; polling slower makes the progress line look frozen.
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const start = useCallback(async () => {
    setError(null);
    const path = mode === "backtest" ? "backtest" : "gauntlet";
    const body =
      mode === "backtest"
        ? {
            strategy,
            top,
            lookback,
            cost_multiple: costMultiple,
            venue,
            actions,
          }
        : {
            factor: factor || null,
            top,
            lookback,
            venue,
            hypothesis: hypothesis.trim() || null,
          };

    const response = await fetch(`${apiBase}/lab/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detailBody = (await response.json().catch(() => ({}))) as { detail?: unknown };
      setError(
        typeof detailBody.detail === "string" ? detailBody.detail : `HTTP ${response.status}`,
      );
      return;
    }
    const job = camel<JobRow>(await response.json());
    setSelected(job.id);
    setDetail(job);
    void refresh();
  }, [
    apiBase,
    mode,
    strategy,
    factor,
    top,
    lookback,
    costMultiple,
    venue,
    actions,
    hypothesis,
    refresh,
  ]);

  const cancel = useCallback(
    async (id: string) => {
      const response = await fetch(`${apiBase}/lab/jobs/${id}`, { method: "DELETE" });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
        setError(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      }
      void refresh();
    },
    [apiBase, refresh],
  );

  const backtest =
    detail?.kind === "backtest" && detail.result
      ? (camel<BacktestResult>(detail.result) as BacktestResult)
      : null;
  const gauntlet =
    detail?.kind === "gauntlet" && detail.result
      ? (camel<GauntletResult>(detail.result) as GauntletResult)
      : null;

  return (
    <div className="analytics">
      <div className="analytics-bar">
        <select
          className="analytics-select"
          value={mode}
          onChange={(event) => setMode(event.target.value as "backtest" | "gauntlet")}
          aria-label="run type"
        >
          <option value="backtest">Backtest</option>
          <option value="gauntlet">Gauntlet</option>
        </select>

        {mode === "backtest" ? (
          <select
            className="analytics-select"
            value={strategy}
            onChange={(event) => setStrategy(event.target.value)}
            aria-label="strategy"
          >
            <option value="momentum">momentum</option>
            <option value="sma">sma crossover</option>
            <option value="hold">buy and hold</option>
          </select>
        ) : (
          <select
            className="analytics-select factor-pick"
            value={factor}
            onChange={(event) => setFactor(event.target.value)}
            aria-label="factor"
          >
            <option value="">momentum strategy</option>
            {factors.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        )}

        <select
          className="analytics-select"
          value={top}
          onChange={(event) => setTop(Number(event.target.value))}
          aria-label="universe size"
        >
          {[10, 20, 30, 50, 100].map((n) => (
            <option key={n} value={n}>
              top {n}
            </option>
          ))}
        </select>

        <select
          className="analytics-select"
          value={lookback}
          onChange={(event) => setLookback(Number(event.target.value))}
          aria-label="lookback"
        >
          {[20, 60, 120, 252].map((n) => (
            <option key={n} value={n}>
              {n}d lookback
            </option>
          ))}
        </select>

        <select
          className="analytics-select"
          value={venue}
          onChange={(event) => setVenue(event.target.value)}
          aria-label="venue"
        >
          <option value="NSE">NSE</option>
          <option value="BSE">BSE</option>
        </select>

        {mode === "backtest" ? (
          <>
            <select
              className="analytics-select"
              value={costMultiple}
              onChange={(event) => setCostMultiple(Number(event.target.value))}
              aria-label="cost multiple"
            >
              <option value={1}>1x costs</option>
              <option value={3}>3x costs</option>
              <option value={5}>5x costs</option>
            </select>
            <label className="analytics-check" title="Fetches splits and dividends per name">
              <input
                type="checkbox"
                checked={actions}
                onChange={(event) => setActions(event.target.checked)}
              />
              corp. actions
            </label>
          </>
        ) : (
          <input
            className="analytics-input"
            value={hypothesis}
            onChange={(event) => setHypothesis(event.target.value)}
            placeholder="hypothesis id (required to pass DSR)"
            aria-label="hypothesis"
          />
        )}

        <span className="analytics-spacer" />
        <button type="button" className="analytics-run" onClick={() => void start()}>
          Run
        </button>
      </div>

      <div className="analytics-body lab">
        {error ? <div className="analytics-note text-critical">{error}</div> : null}

        {/* Corporate actions off means a split reads as a -50% day (§9). Said
            here rather than buried in the result, because it changes what the
            drawdown below means. */}
        {mode === "backtest" && !actions ? (
          <div className="analytics-note">
            Corporate actions are off: a split in any holding will read as a price crash.
          </div>
        ) : null}

        <div className="lab-split">
          <div className="lab-runs">
            <div className="lab-heading">Runs</div>
            {jobs.length === 0 ? (
              <div className="text-secondary">Nothing has been run yet.</div>
            ) : null}
            {/* Two sibling buttons, not one nested in the other. A
                `<span role="button">` inside a `<button>` is invalid DOM: the
                browser may reparent it, and the cancel click then lands on
                whatever the fixup produced. */}
            {jobs.map((job) => (
              <div key={job.id} className={`lab-run ${selected === job.id ? "on" : ""}`}>
                <button
                  type="button"
                  className="lab-run-open"
                  onClick={() => {
                    setSelected(job.id);
                    void refresh();
                  }}
                >
                  <span className={`lab-state lab-state-${job.state}`}>{job.state}</span>
                  <span className="lab-run-label">{job.label}</span>
                  <span className="text-secondary">
                    {job.state === "running" ? job.progress : elapsed(job.elapsedSeconds)}
                  </span>
                </button>
                {job.state === "queued" ? (
                  <button
                    type="button"
                    className="lab-cancel"
                    onClick={() => void cancel(job.id)}
                  >
                    cancel
                  </button>
                ) : null}
              </div>
            ))}
          </div>

          <div className="lab-detail">
            {detail && detail.state === "failed" ? (
              <div className="verdict-banner verdict-bad">
                <span className="verdict-word">FAILED</span>
                <span>{detail.error}</span>
              </div>
            ) : null}

            {detail && !detail.result && detail.state !== "failed" ? (
              <div className="analytics-note">
                {detail.state} — {detail.progress}
              </div>
            ) : null}

            {backtest ? (
              <>
                <div
                  className={`verdict-banner ${
                    backtest.totalReturn > 0 ? "verdict-ok" : "verdict-bad"
                  }`}
                >
                  <span className="verdict-word">
                    {formatPercent(backtest.totalReturn)}
                  </span>
                  <span>
                    {backtest.strategy} · {formatCount(backtest.sessions)} sessions ·{" "}
                    {backtest.costModel}
                    {backtest.costMultiple === 1 ? "" : ` at ${backtest.costMultiple}x`}
                  </span>
                </div>

                <svg className="lab-curve" viewBox="0 0 600 140" preserveAspectRatio="none">
                  <path d={curvePath(backtest.curve, 600, 140)} className="lab-curve-line" />
                </svg>

                <div className="stat-row">
                  <div className="stat">
                    <div className="stat-label">max drawdown</div>
                    <div className="stat-value text-loss">
                      {formatPercent(backtest.maxDrawdown)}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">fees</div>
                    <div className="stat-value">
                      {formatPercent(backtest.feesPaid / backtest.startEquity)}
                    </div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">trades</div>
                    <div className="stat-value">{formatCount(backtest.trades)}</div>
                  </div>
                  <div className="stat">
                    <div className="stat-label">filled</div>
                    <div className="stat-value">
                      {formatCount(backtest.ordersFilled)} /{" "}
                      {formatCount(backtest.ordersGenerated)}
                    </div>
                  </div>
                </div>

                {/* Rejections are a bug; unfunded and no-market are the market
                    and the budget, and lumping them together would make a
                    healthy fully-invested book look broken. */}
                {backtest.ordersRejected > 0 ? (
                  <div className="analytics-note text-critical">
                    {backtest.ordersRejected} order(s) rejected — that is an accounting refusal,
                    not a market condition.
                  </div>
                ) : null}
                {backtest.actionsNote ? (
                  <pre className="lab-log">{backtest.actionsNote}</pre>
                ) : null}
              </>
            ) : null}

            {gauntlet ? (
              <>
                <div
                  className={`verdict-banner ${gauntlet.passed ? "verdict-ok" : "verdict-bad"}`}
                >
                  <span className="verdict-word">
                    {gauntlet.passed ? "PASSED" : `REJECTED at ${gauntlet.firstFailure}`}
                  </span>
                  <span>
                    {gauntlet.label} · Sharpe {gauntlet.sharpe.toFixed(2)} ·{" "}
                    {formatCount(gauntlet.periods)} periods
                  </span>
                </div>

                {gauntlet.implausible ? (
                  <div className="verdict-banner verdict-warn">
                    <span className="verdict-word">IMPLAUSIBLE</span>
                    <span>
                      Sharpe above the 2.5 smell test (§2.1). Suspect a leak before believing it.
                    </span>
                  </div>
                ) : null}

                {!gauntlet.hypothesis ? (
                  <div className="analytics-note">
                    No pre-registered hypothesis, so the Deflated Sharpe check reports its number
                    but cannot pass: an unverified trial count moves a strategy only ever toward
                    accept.
                  </div>
                ) : null}

                <table className="grid">
                  <thead>
                    <tr>
                      <th>check</th>
                      <th className="num">statistic</th>
                      <th className="num">limit</th>
                      <th>why</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gauntlet.results.map((row) => (
                      <tr key={row.test}>
                        <td>
                          <span
                            className={
                              row.skipped
                                ? "lab-check skip"
                                : row.passed
                                  ? "lab-check pass"
                                  : "lab-check fail"
                            }
                          >
                            {row.skipped ? "SKIP" : row.passed ? "PASS" : "FAIL"}
                          </span>{" "}
                          {row.test}
                        </td>
                        <td className="num">
                          {row.statistic === null ? "—" : row.statistic.toFixed(4)}
                        </td>
                        <td className="num">{row.threshold === null ? "—" : row.threshold}</td>
                        <td className="text-secondary">{row.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
