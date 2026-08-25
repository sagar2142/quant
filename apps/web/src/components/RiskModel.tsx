/**
 * Risk decomposition — MASTER_PLAN §6, §8.
 *
 * **The question every other screen begged.** Factors shows whether a signal
 * predicts; Positions shows what is held; Risk shows the limits. None of them
 * said how much of *this book's* risk is a factor and how much is the names in
 * it — so a book whose variance is two-thirds market looked, from every angle,
 * like a portfolio of carefully chosen factor bets.
 *
 * The market row is usually the answer and is meant to be. A long-only book's
 * dominant risk is that the market falls, and presenting style tilts without
 * it would flatter the research that produced them.
 */

import { useCallback, useEffect, useState } from "react";
import { formatPercent, formatRatio, signClass } from "../format";

interface Contribution {
  name: string;
  share: number;
}

interface RiskModelResult {
  present: boolean;
  totalVolatility: number;
  factorVolatility: number;
  specificVolatility: number;
  factorShare: number;
  specificShare: number;
  contributions: Contribution[];
  factorVolatilities: Contribution[];
  sessions: number;
  names: number;
  illConditioned: boolean;
  condition: number;
  note: string;
}

function camel<T>(value: unknown): T {
  if (Array.isArray(value)) return value.map((v) => camel(v)) as unknown as T;
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([k, v]) => [
        k.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()),
        camel(v),
      ]),
    ) as T;
  }
  return value as T;
}

export function RiskModel({ apiBase = "/api" }: { apiBase?: string }) {
  const [sessions, setSessions] = useState(756);
  const [result, setResult] = useState<RiskModelResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/risk/model?sessions=${sessions}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setResult(camel<RiskModelResult>(await response.json()));
    } catch (cause) {
      setError(String(cause));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [apiBase, sessions]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="analytics">
      <div className="analytics-bar">
        <select
          className="analytics-select"
          value={sessions}
          onChange={(event) => setSessions(Number(event.target.value))}
          aria-label="estimation window"
        >
          <option value={504}>2y window</option>
          <option value={756}>3y window</option>
          <option value={1260}>5y window</option>
        </select>
        <span className="analytics-spacer" />
        {loading ? <span className="text-secondary">estimating…</span> : null}
      </div>

      <div className="analytics-body">
        {error ? <div className="analytics-note text-critical">{error}</div> : null}

        {result && !result.present ? (
          <div className="empty">{result.note || "Nothing to decompose."}</div>
        ) : null}

        {result?.present ? (
          <>
            {/* The headline is the split, not the level. A 17% volatility means
                little; 17% of which two thirds is market means a great deal. */}
            <div className="stat-row">
              <div className="stat">
                <div className="stat-label">total volatility</div>
                <div className="stat-value">{formatPercent(result.totalVolatility)}</div>
              </div>
              <div className="stat">
                <div className="stat-label">from factors</div>
                <div className="stat-value">
                  {formatPercent(result.factorVolatility)}{" "}
                  <span className="text-secondary">
                    ({formatPercent(result.factorShare)} of variance)
                  </span>
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">specific</div>
                <div className="stat-value">
                  {formatPercent(result.specificVolatility)}{" "}
                  <span className="text-secondary">
                    ({formatPercent(result.specificShare)} of variance)
                  </span>
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">names</div>
                <div className="stat-value">{result.names}</div>
              </div>
            </div>

            {result.illConditioned ? (
              <div className="verdict-banner verdict-warn">
                <span className="verdict-word">ILL-CONDITIONED</span>
                <span>
                  condition {formatRatio(result.condition)} — the factors are
                  near-collinear, so the split between them below is unstable.
                  The total is still sound; which factor gets the credit is not.
                </span>
              </div>
            ) : null}

            <div className="factor-split">
              <section>
                <h3 className="analytics-subhead">
                  Variance by factor
                  <span className="subhead-note">share of total; negative offsets</span>
                </h3>
                <div className="buckets">
                  {result.contributions.map((row) => (
                    <div className="bucket" key={row.name}>
                      <span className="bucket-label">{row.name}</span>
                      <div className="bucket-track">
                        <div
                          className={`bucket-fill ${row.share >= 0 ? "up" : "down"}`}
                          style={{ width: `${Math.min(100, Math.abs(row.share) * 100)}%` }}
                        />
                      </div>
                      <span className={`bucket-value ${signClass(row.share)}`}>
                        {formatPercent(row.share)}
                      </span>
                    </div>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="analytics-subhead">
                  Factor volatility
                  <span className="subhead-note">annualised, own return series</span>
                </h3>
                <table className="grid">
                  <thead>
                    <tr>
                      <th>factor</th>
                      <th className="right">volatility</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.factorVolatilities.map((row) => (
                      <tr key={row.name}>
                        <td>{row.name}</td>
                        <td className="right">{formatPercent(row.share)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="analytics-note text-secondary">
                  Estimated over {result.sessions.toLocaleString()} sessions of
                  cross-sectional regressions. These are realised covariances,
                  not forecasts — they say what the book is exposed to, not what
                  it will earn.
                </div>
              </section>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
