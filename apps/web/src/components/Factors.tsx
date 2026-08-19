/**
 * Factor research screen — MASTER_PLAN §6.
 *
 * The fast loop, on screen. Pick a signal, get its Information Coefficient at
 * four horizons, its quantile buckets and — the line that matters most — whether
 * the spread survives the round-trip cost of trading it at its own turnover.
 *
 * **The cost verdict is rendered before the pretty numbers, and coloured.** A
 * signal with a strong IC and an edge smaller than its transaction costs is the
 * single most common false positive in factor research, and a screen that buries
 * that under a monotonic bar chart is helping you fool yourself.
 */

import { useCallback, useEffect, useState } from "react";
import { formatPercent, formatRatio, signClass } from "../format";

export interface HorizonRow {
  horizon: number;
  ic: number;
  informationRatio: number;
  tStat: number;
  hitRate: number;
  sessions: number;
  significant: boolean;
}

export interface BucketRow {
  quantile: number;
  forwardReturn: number;
  names: number;
}

export interface FactorResult {
  factor: string;
  description: string;
  names: number;
  sessions: number;
  horizons: HorizonRow[];
  buckets: BucketRow[];
  quantileHorizon: number;
  spread: number;
  monotonic: boolean;
  turnover: number;
  netOfCosts: number;
  survivesCosts: boolean;
}

interface FactorOption {
  name: string;
  description: string;
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

/** Bucket bars scaled to the largest magnitude, so shape is comparable. */
function barWidth(value: number, buckets: BucketRow[]): string {
  const peak = Math.max(...buckets.map((b) => Math.abs(b.forwardReturn)), 1e-9);
  return `${(Math.abs(value) / peak) * 100}%`;
}

export function Factors({ apiBase = "/api" }: { apiBase?: string }) {
  const [options, setOptions] = useState<FactorOption[]>([]);
  const [factor, setFactor] = useState("momentum_12_1");
  const [horizon, setHorizon] = useState(21);
  const [sessions, setSessions] = useState(0);
  const [result, setResult] = useState<FactorResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void (async () => {
      const response = await fetch(`${apiBase}/factors`);
      if (response.ok) setOptions(camel<FactorOption[]>(await response.json()));
    })();
  }, [apiBase]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${apiBase}/factor/${factor}?horizon=${horizon}&sessions=${sessions}`,
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(body.detail ?? `HTTP ${response.status}`);
      }
      setResult(camel<FactorResult>(await response.json()));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [apiBase, factor, horizon, sessions]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="analytics">
      <div className="analytics-bar">
        <select
          className="analytics-select factor-pick"
          value={factor}
          onChange={(event) => setFactor(event.target.value)}
          aria-label="factor"
        >
          {options.map((o) => (
            <option key={o.name} value={o.name}>{o.name}</option>
          ))}
        </select>
        <select
          className="analytics-select"
          value={horizon}
          onChange={(event) => setHorizon(Number(event.target.value))}
          aria-label="quantile horizon"
        >
          {[1, 5, 21, 63].map((h) => (
            <option key={h} value={h}>{h}d forward</option>
          ))}
        </select>
        <select
          className="analytics-select"
          value={sessions}
          onChange={(event) => setSessions(Number(event.target.value))}
          aria-label="sessions"
        >
          <option value={0}>all history</option>
          <option value={1250}>5y</option>
          <option value={750}>3y</option>
          <option value={500}>2y</option>
        </select>
        <span className="analytics-spacer" />
        {loading ? <span className="text-secondary">scoring…</span> : null}
      </div>

      <div className="analytics-body">
        {error ? <div className="analytics-note text-critical">{error}</div> : null}

        {result ? (
          <>
            {/* The verdict first. A monotonic bar chart above the cost line
                would let a signal that cannot be traded look like a find. */}
            <div
              className={`verdict-banner ${result.survivesCosts ? "verdict-ok" : "verdict-bad"}`}
            >
              <span className="verdict-word">
                {result.survivesCosts ? "SURVIVES COSTS" : "DIES ON COSTS"}
              </span>
              <span>
                spread {formatPercent(result.spread)} · turnover{" "}
                {formatPercent(result.turnover)}/session · net{" "}
                {formatPercent(result.netOfCosts)}
              </span>
            </div>

            <div className="stat-row">
              <div className="stat">
                <div className="stat-label">names</div>
                <div className="stat-value">{result.names.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="stat-label">sessions</div>
                <div className="stat-value">{result.sessions.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="stat-label">monotonic</div>
                <div className={`stat-value ${result.monotonic ? "text-profit" : "text-warn"}`}>
                  {result.monotonic ? "yes" : "no"}
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">spread Q{result.buckets.length}-Q1</div>
                <div className={`stat-value ${signClass(result.spread)}`}>
                  {formatPercent(result.spread)}
                </div>
              </div>
            </div>

            <h3 className="analytics-subhead">
              Information coefficient — rank, per session
            </h3>
            <table className="grid">
              <thead>
                <tr>
                  <th>horizon</th>
                  <th className="right">IC</th>
                  <th className="right">IR</th>
                  <th className="right">t</th>
                  <th className="right">hit</th>
                  <th className="right">sessions</th>
                </tr>
              </thead>
              <tbody>
                {result.horizons.map((h) => (
                  <tr key={h.horizon}>
                    <td>{h.horizon}d</td>
                    <td className={`right ${signClass(h.ic)}`}>{h.ic.toFixed(4)}</td>
                    <td className="right">{formatRatio(h.informationRatio)}</td>
                    <td className={`right ${h.significant ? "text-profit" : "text-secondary"}`}>
                      {h.tStat.toFixed(2)}
                    </td>
                    <td className="right">{formatPercent(h.hitRate)}</td>
                    <td className="right text-secondary">{h.sessions.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            <h3 className="analytics-subhead">
              Quantile forward return — {result.quantileHorizon}d
            </h3>
            <div className="buckets">
              {result.buckets.map((b) => (
                <div className="bucket" key={b.quantile}>
                  <span className="bucket-label">Q{b.quantile}</span>
                  <div className="bucket-track">
                    <div
                      className={`bucket-fill ${b.forwardReturn >= 0 ? "up" : "down"}`}
                      style={{ width: barWidth(b.forwardReturn, result.buckets) }}
                    />
                  </div>
                  <span className={`bucket-value ${signClass(b.forwardReturn)}`}>
                    {formatPercent(b.forwardReturn)}
                  </span>
                </div>
              ))}
            </div>

            <div className="analytics-note text-secondary">{result.description}</div>
            <div className="analytics-note text-secondary">
              A strong IC is permission to build a strategy, not evidence one
              works. Send survivors to the gauntlet.
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
