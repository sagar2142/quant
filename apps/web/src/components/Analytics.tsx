/**
 * Analytics screens — MASTER_PLAN §12.6.
 *
 * The same numbers `apps.cli.terminal` prints, served by `apps/api/analytics`
 * so there is one implementation of "what is this security". Two would
 * eventually disagree, and the one on screen would be the wrong one.
 *
 * Dense by design (§12.10): a quant screen is read by scanning columns. Every
 * figure that fits without scrolling is one compared against its neighbours
 * for free.
 *
 * Nothing here recommends a trade. The profile describes price behaviour; the
 * gauntlet decides whether a strategy built on it is evidence.
 */

import { useCallback, useEffect, useState } from "react";
import { formatPercent, formatPrice, formatRatio, signClass } from "../format";
import { DrawdownChart, PriceChart } from "./Sparkline";

export interface Horizon {
  label: string;
  value: number | null;
}

export interface Security {
  symbol: string;
  observations: number;
  lastClose: number;
  horizons: Horizon[];
  cagr: number;
  high52w: number;
  low52w: number;
  offHigh: number;
  annualVolatility: number;
  maxDrawdown: number;
  currentDrawdown: number;
  advValue: number | null;
  sharpe: number;
  sortino: number;
  calmar: number;
  hitRate: number;
  skewness: number;
  kurtosis: number;
  var5: number;
  cvar5: number;
  tailRatio: number;
  verdict: string;
  adfPvalue: number;
  kpssPvalue: number;
  hurst: number;
  tradableAsMeanReversion: boolean;
  autocorrelation: Record<string, number>;
  realisedVol: number;
  ewmaVol: number;
  volRegime: string;
  isImplausible: boolean;
  fatLeftTail: boolean;
}

export interface NameRow {
  symbol: string;
  totalReturn: number;
  annualVolatility: number;
  sharpe: number;
  beta: number;
  correlationToMarket: number;
  weightHrp: number;
  weightErc: number;
  cluster: number;
}

export interface CrossSection {
  names: NameRow[];
  sessions: number;
  meanCorrelation: number;
  clusters: number;
  effectiveBets: number;
  diversificationRatio: number;
  conditionNumber: number;
  shrinkage: number;
  marketReturn: number;
  marketVolatility: number;
  isIllConditioned: boolean;
  concentrationWarning: string | null;
  correlationLabels: string[];
  correlation: number[][];
}

/** snake_case from FastAPI to camelCase, so the API stays idiomatic Python. */
function camel<T>(input: unknown): T {
  if (Array.isArray(input)) return input.map((v) => camel(v)) as unknown as T;
  if (input === null || typeof input !== "object") return input as T;
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input as Record<string, unknown>)) {
    const name = key.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());
    // Autocorrelation is keyed by lag ("1", "2"); those are data, not names.
    out[name] = key === "autocorrelation" ? value : camel(value);
  }
  return out as T;
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string };
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return camel<T>(await response.json());
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

function SecurityScreen({ data, closes }: { data: Security; closes: number[] }) {
  const ac = data.autocorrelation;
  return (
    <>
      <div className="analytics-head">
        <span className="analytics-symbol">{data.symbol}</span>
        <span className="text-secondary">
          {data.observations.toLocaleString()} sessions
        </span>
        <span className="analytics-last">{formatPrice(data.lastClose)}</span>
      </div>

      <div className="stat-row">
        {data.horizons.map((h) => (
          <Stat
            key={h.label}
            label={h.label}
            value={h.value === null ? "—" : formatPercent(h.value)}
            tone={h.value === null ? "text-secondary" : signClass(h.value)}
          />
        ))}
        <Stat label="CAGR" value={formatPercent(data.cagr)} tone={signClass(data.cagr)} />
      </div>

      {closes.length > 1 ? (
        <div className="chart-stack">
          <div className="chart-wrap">
            <span className="chart-label">price (back-adjusted)</span>
            <PriceChart closes={closes} label={data.symbol} />
          </div>
          <div className="chart-wrap">
            <span className="chart-label">drawdown from peak</span>
            <DrawdownChart closes={closes} />
          </div>
        </div>
      ) : null}

      <div className="analytics-grid">
        <section className="analytics-block">
          <h3>Risk</h3>
          <dl>
            <dt>volatility</dt><dd>{formatPercent(data.annualVolatility)}</dd>
            <dt>max drawdown</dt><dd className="text-loss">{formatPercent(data.maxDrawdown)}</dd>
            <dt>current drawdown</dt><dd className="text-loss">{formatPercent(data.currentDrawdown)}</dd>
            <dt>VaR 5%</dt><dd className="text-loss">{formatPercent(data.var5)}</dd>
            <dt>CVaR 5%</dt><dd className="text-loss">{formatPercent(data.cvar5)}</dd>
            <dt>52w high / low</dt>
            <dd>{formatPrice(data.high52w)} / {formatPrice(data.low52w)}</dd>
            <dt>off high</dt><dd className={signClass(data.offHigh)}>{formatPercent(data.offHigh)}</dd>
          </dl>
        </section>

        <section className="analytics-block">
          <h3>Ratios &amp; shape</h3>
          <dl>
            <dt>Sharpe</dt><dd className={signClass(data.sharpe)}>{formatRatio(data.sharpe)}</dd>
            <dt>Sortino</dt><dd className={signClass(data.sortino)}>{formatRatio(data.sortino)}</dd>
            <dt>Calmar</dt><dd className={signClass(data.calmar)}>{formatRatio(data.calmar)}</dd>
            <dt>hit rate</dt><dd>{formatPercent(data.hitRate)}</dd>
            <dt>skewness</dt><dd className={signClass(data.skewness)}>{formatRatio(data.skewness)}</dd>
            <dt>kurtosis</dt><dd>{formatRatio(data.kurtosis)}</dd>
            <dt>tail ratio</dt><dd>{formatRatio(data.tailRatio)}</dd>
          </dl>
        </section>

        <section className="analytics-block">
          <h3>Process</h3>
          <dl>
            <dt>verdict</dt>
            <dd className={data.tradableAsMeanReversion ? "text-profit" : "text-warn"}>
              {data.verdict}
            </dd>
            <dt>ADF p</dt><dd>{data.adfPvalue.toFixed(4)}</dd>
            <dt>KPSS p</dt><dd>{data.kpssPvalue.toFixed(4)}</dd>
            <dt>Hurst</dt><dd>{data.hurst.toFixed(3)}</dd>
            {Object.entries(ac).map(([lag, value]) => (
              <>
                <dt key={`k${lag}`}>autocorr lag {lag}</dt>
                <dd key={`v${lag}`} className={signClass(value)}>{value.toFixed(3)}</dd>
              </>
            ))}
          </dl>
        </section>

        <section className="analytics-block">
          <h3>Volatility</h3>
          <dl>
            <dt>realised</dt><dd>{formatPercent(data.realisedVol)}</dd>
            <dt>EWMA</dt><dd>{formatPercent(data.ewmaVol)}</dd>
            <dt>regime</dt>
            <dd className={data.volRegime === "elevated" ? "text-warn" : ""}>
              {data.volRegime}
            </dd>
            <dt>ADV</dt>
            <dd>{data.advValue === null ? "—" : `${(data.advValue / 1e7).toFixed(1)} Cr`}</dd>
          </dl>
        </section>
      </div>

      {data.isImplausible ? (
        <div className="analytics-note text-warn">
          Sharpe above the 2.5 smell test (§2.1) — suspect a missed corporate
          action before believing this number.
        </div>
      ) : null}
      {data.fatLeftTail ? (
        <div className="analytics-note text-loss">
          Fat left tail: negative skew with excess kurtosis. Small gains until
          they are not.
        </div>
      ) : null}
      {!data.tradableAsMeanReversion ? (
        <div className="analytics-note text-secondary">
          Not fadeable — {data.verdict.toLowerCase()} process. Mean reversion has
          no level to revert to here (§253).
        </div>
      ) : null}
    </>
  );
}

/** Correlation cell shading. Colour is semantic here, not decoration. */
function correlationTone(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 0.999) return "corr-self";
  if (magnitude >= 0.7) return "corr-high";
  if (magnitude >= 0.4) return "corr-mid";
  return "corr-low";
}

function CrossSectionScreen({ data }: { data: CrossSection }) {
  return (
    <>
      <div className="stat-row">
        <Stat label="names" value={String(data.names.length)} />
        <Stat label="sessions" value={data.sessions.toLocaleString()} />
        <Stat
          label="market"
          value={formatPercent(data.marketReturn)}
          tone={signClass(data.marketReturn)}
        />
        <Stat label="mean corr" value={data.meanCorrelation.toFixed(3)} />
        <Stat label="clusters" value={String(data.clusters)} />
        <Stat
          label="effective bets"
          value={data.effectiveBets.toFixed(2)}
          tone={data.concentrationWarning ? "text-warn" : ""}
        />
        <Stat label="div ratio" value={data.diversificationRatio.toFixed(2)} />
        <Stat
          label="condition"
          value={data.conditionNumber.toFixed(0)}
          tone={data.isIllConditioned ? "text-critical" : ""}
        />
      </div>

      {data.concentrationWarning ? (
        <div className="analytics-note text-warn">{data.concentrationWarning}</div>
      ) : null}
      {data.isIllConditioned ? (
        <div className="analytics-note text-critical">
          Covariance is near-singular: treat every weight below as arbitrary.
          Add history or drop names (§268).
        </div>
      ) : null}

      <table className="grid">
        <thead>
          <tr>
            <th>symbol</th>
            <th className="right">return</th>
            <th className="right">vol</th>
            <th className="right">sharpe</th>
            <th className="right">beta</th>
            <th className="right">corr</th>
            <th className="right">HRP</th>
            <th className="right">ERC</th>
            <th className="right">clu</th>
          </tr>
        </thead>
        <tbody>
          {data.names.map((n) => (
            <tr key={n.symbol}>
              <td>{n.symbol}</td>
              <td className={`right ${signClass(n.totalReturn)}`}>
                {formatPercent(n.totalReturn)}
              </td>
              <td className="right">{formatPercent(n.annualVolatility)}</td>
              <td className={`right ${signClass(n.sharpe)}`}>{formatRatio(n.sharpe)}</td>
              <td className="right">{n.beta.toFixed(2)}</td>
              <td className="right">{n.correlationToMarket.toFixed(2)}</td>
              <td className="right">{formatPercent(n.weightHrp)}</td>
              <td className="right">{formatPercent(n.weightErc)}</td>
              <td className="right">{n.cluster}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 className="analytics-subhead">Correlation</h3>
      <table className="grid corr">
        <thead>
          <tr>
            <th />
            {data.correlationLabels.map((s) => (
              <th key={s} className="right">{s.slice(0, 6)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.correlation.map((row, i) => (
            <tr key={data.correlationLabels[i]}>
              <td>{data.correlationLabels[i]}</td>
              {row.map((v, j) => (
                <td
                  key={`${i}-${j}`}
                  className={`right ${correlationTone(v)}`}
                >
                  {v.toFixed(2)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function Analytics({
  apiBase = "/api",
  initialSymbols,
}: {
  apiBase?: string;
  initialSymbols?: string;
}) {
  const [input, setInput] = useState(
    initialSymbols ?? "RELIANCE TCS INFY HDFCBANK ICICIBANK SBIN",
  );
  const [sessions, setSessions] = useState(750);
  const [security, setSecurity] = useState<Security | null>(null);
  const [closes, setCloses] = useState<number[]>([]);
  const [matches, setMatches] = useState<{ symbol: string; adv: number }[]>([]);
  const [section, setSection] = useState<CrossSection | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Suggestions for the token under the cursor — the last word, since the
  // field takes several symbols. Without this the field demands you already
  // know the exact ticker, and "TATA" is not one: TATASTEEL, TATAPOWER and
  // eight others are.
  const typing = input.trim().split(/[\s,]+/).pop() ?? "";
  useEffect(() => {
    if (typing.length < 2) {
      setMatches([]);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(async () => {
      const response = await fetch(`${apiBase}/symbols?q=${encodeURIComponent(typing)}`);
      if (!response.ok || cancelled) return;
      const rows = (await response.json()) as { symbol: string; adv: number }[];
      // Hide the list once the token is already an exact ticker.
      setMatches(rows.length === 1 && rows[0]?.symbol === typing.toUpperCase() ? [] : rows.slice(0, 12));
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [apiBase, typing]);

  const pick = useCallback(
    (symbol: string) => {
      const parts = input.trim().split(/[\s,]+/).filter(Boolean);
      parts[parts.length === 0 ? 0 : parts.length - 1] = symbol;
      setInput(parts.join(" ") + " ");
      setMatches([]);
    },
    [input],
  );

  const load = useCallback(async () => {
    const symbols = input.trim().split(/[\s,]+/).filter(Boolean).map((s) => s.toUpperCase());
    if (symbols.length === 0) return;

    setLoading(true);
    setError(null);
    try {
      const profile = await fetchJson<Security>(
        `${apiBase}/security/${symbols[0]}?sessions=${sessions}`,
      );
      setSecurity(profile);

      const curve = await fetchJson<{ closes: number[] }>(
        `${apiBase}/security/${symbols[0]}/series?sessions=${sessions}`,
      );
      setCloses(curve.closes);

      if (symbols.length > 1) {
        setSection(
          await fetchJson<CrossSection>(
            `${apiBase}/crosssection?symbols=${symbols.join(",")}&sessions=${sessions}`,
          ),
        );
      } else {
        setSection(null);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      setSecurity(null);
      setSection(null);
      setCloses([]);
    } finally {
      setLoading(false);
    }
  }, [apiBase, input, sessions]);

  useEffect(() => {
    void load();
    // Deliberately once on mount. Re-running on every keystroke would fire a
    // multi-second analytics request per character.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="analytics">
      <form
        className="analytics-bar"
        onSubmit={(event) => {
          event.preventDefault();
          void load();
        }}
      >
        <input
          className="analytics-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="RELIANCE TCS INFY"
          aria-label="symbols"
          spellCheck={false}
        />
        <select
          className="analytics-select"
          value={sessions}
          onChange={(event) => setSessions(Number(event.target.value))}
          aria-label="sessions"
        >
          <option value={250}>1y</option>
          <option value={500}>2y</option>
          <option value={750}>3y</option>
          <option value={1250}>5y</option>
          <option value={0}>all</option>
        </select>
        <button className="analytics-run" type="submit" disabled={loading}>
          {loading ? "…" : "Run"}
        </button>
      </form>

      {matches.length > 0 ? (
        <div className="suggestions">
          {matches.map((m) => (
            <button
              key={m.symbol}
              type="button"
              className="suggestion"
              onClick={() => pick(m.symbol)}
            >
              <span>{m.symbol}</span>
              <span className="suggestion-adv">{(m.adv / 1e7).toFixed(0)} Cr</span>
            </button>
          ))}
        </div>
      ) : null}

      {error ? <div className="analytics-note text-critical">{error}</div> : null}

      <div className="analytics-body">
        {security ? <SecurityScreen data={security} closes={closes} /> : null}
        {section ? (
          <>
            <h3 className="analytics-subhead">
              Cross-section — {section.names.length} names
            </h3>
            <CrossSectionScreen data={section} />
          </>
        ) : null}
        {!security && !error && !loading ? (
          <div className="empty">Enter one or more symbols.</div>
        ) : null}
      </div>
    </div>
  );
}
