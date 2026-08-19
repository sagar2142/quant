/**
 * Screener screen — MASTER_PLAN §6, §253.
 *
 * Answers *which names*, rather than describing one already named. The system
 * could do the second from the start; this is the question research begins
 * from.
 *
 * Flags are shown, never hidden: a Sharpe above 2.5 on a single name usually
 * means a missed corporate action, and a screen that presents it as a
 * discovery is worse than no screen.
 */

import { useCallback, useEffect, useState } from "react";
import { formatPercent, formatRatio, signClass } from "../format";

export interface ScreenRow {
  symbol: string;
  adv: number;
  bars: number;
  lastClose: number;
  windowReturn: number;
  annualVolatility: number;
  sharpe: number;
  maxDrawdown: number;
  hurst: number;
  verdict: string;
  fadeable: boolean;
  isImplausible: boolean;
  fatLeftTail: boolean;
}

export interface ScreenResult {
  rows: ScreenRow[];
  considered: number;
  passedFilters: number;
  profiled: number;
  suspectedActions: number;
  sortBy: string;
}

const SORTS = [
  { id: "liquidity", label: "Liquidity" },
  { id: "momentum", label: "Momentum" },
  { id: "reversal", label: "Reversal" },
  { id: "volatility", label: "Volatility" },
];

function camel<T>(input: unknown): T {
  if (Array.isArray(input)) return input.map((v) => camel(v)) as unknown as T;
  if (input === null || typeof input !== "object") return input as T;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    out[key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase())] = camel(value);
  }
  return out as T;
}

export function Screener({
  apiBase = "/api",
  onPick,
}: {
  apiBase?: string;
  onPick?: (symbol: string) => void;
}) {
  const [sort, setSort] = useState("liquidity");
  const [limit, setLimit] = useState(25);
  const [stationaryOnly, setStationaryOnly] = useState(false);
  const [result, setResult] = useState<ScreenResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const url =
        `${apiBase}/screen?sort=${sort}&limit=${limit}` +
        `&stationary_only=${stationaryOnly}`;
      const response = await fetch(url);
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(body.detail ?? `HTTP ${response.status}`);
      }
      setResult(camel<ScreenResult>(await response.json()));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [apiBase, sort, limit, stationaryOnly]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="analytics">
      <div className="analytics-bar">
        <select
          className="analytics-select"
          value={sort}
          onChange={(event) => setSort(event.target.value)}
          aria-label="sort by"
        >
          {SORTS.map((s) => (
            <option key={s.id} value={s.id}>{s.label}</option>
          ))}
        </select>
        <select
          className="analytics-select"
          value={limit}
          onChange={(event) => setLimit(Number(event.target.value))}
          aria-label="limit"
        >
          {[10, 25, 50, 100].map((n) => (
            <option key={n} value={n}>top {n}</option>
          ))}
        </select>
        <label className="analytics-check">
          <input
            type="checkbox"
            checked={stationaryOnly}
            onChange={(event) => setStationaryOnly(event.target.checked)}
          />
          fadeable only
        </label>
        <span className="analytics-spacer" />
        {loading ? <span className="text-secondary">screening…</span> : null}
      </div>

      <div className="analytics-body">
        {error ? <div className="analytics-note text-critical">{error}</div> : null}

        {result ? (
          <>
            <div className="stat-row">
              <div className="stat">
                <div className="stat-label">considered</div>
                <div className="stat-value">{result.considered.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="stat-label">passed filters</div>
                <div className="stat-value">{result.passedFilters.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="stat-label">profiled</div>
                <div className="stat-value">{result.profiled}</div>
              </div>
              <div className="stat">
                <div className="stat-label">action-excluded</div>
                <div className="stat-value text-warn">{result.suspectedActions}</div>
              </div>
            </div>

            {result.suspectedActions > 0 ? (
              <div className="analytics-note text-secondary">
                {result.suspectedActions} name(s) excluded for a session move above
                35%. The panel holds raw prices, so a bonus reads as a crash and
                would otherwise dominate a reversal screen (§9).
              </div>
            ) : null}

            {result.rows.length === 0 ? (
              <div className="analytics-note text-warn">
                Nothing met the criteria. With “fadeable only” that is a result,
                not an error: no liquid name is currently stationary enough for a
                mean-reversion strategy to have a level to revert to (§253).
              </div>
            ) : (
              <table className="grid">
                <thead>
                  <tr>
                    <th>symbol</th>
                    <th className="right">ADV Cr</th>
                    <th className="right">return</th>
                    <th className="right">vol</th>
                    <th className="right">sharpe</th>
                    <th className="right">maxDD</th>
                    <th className="right">hurst</th>
                    <th>process</th>
                    <th>flags</th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row) => (
                    <tr
                      key={row.symbol}
                      className={onPick ? "clickable" : undefined}
                      onClick={onPick ? () => onPick(row.symbol) : undefined}
                    >
                      <td>{row.symbol}</td>
                      <td className="right">{(row.adv / 1e7).toFixed(0)}</td>
                      <td className={`right ${signClass(row.windowReturn)}`}>
                        {formatPercent(row.windowReturn)}
                      </td>
                      <td className="right">{formatPercent(row.annualVolatility)}</td>
                      <td className={`right ${signClass(row.sharpe)}`}>
                        {formatRatio(row.sharpe)}
                      </td>
                      <td className="right text-loss">{formatPercent(row.maxDrawdown)}</td>
                      <td className="right">{row.hurst.toFixed(3)}</td>
                      <td className={row.fadeable ? "text-profit" : "text-secondary"}>
                        {row.verdict}
                      </td>
                      <td className="flags">
                        {row.fadeable ? <span className="tag tag-ok">FADEABLE</span> : null}
                        {row.isImplausible ? (
                          <span className="tag tag-warn">SR&gt;2.5</span>
                        ) : null}
                        {row.fatLeftTail ? (
                          <span className="tag tag-loss">FAT TAIL</span>
                        ) : null}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
