/**
 * System health — MASTER_PLAN §12.7, §5.1.
 *
 * **This replaces four screens that could not have shown anything.** Overview,
 * Positions, Blotter and Reconcile all read the paper-trading state file,
 * which was deliberately removed: `/book` returns an empty book, `/equity` an
 * empty curve, `/fills` nothing and `/reconciliation` `checked: false`. Four of
 * fourteen nav entries were permanently blank, which is worse than absent —
 * a screen that is always empty teaches you to stop looking at screens.
 *
 * What replaces them is what can actually be known without a broker connected:
 * how much data there is and how fresh it is, which limits are armed, and what
 * the research protocol has decided so far.
 */

import { useCallback, useEffect, useState } from "react";
import { Icon } from "./Icon";

interface Venue {
  venue: string;
  sessions: number;
  first: string | null;
  last: string | null;
}

interface PaperStatus {
  started: boolean;
  cycles: number;
  first_session: string | null;
  last_session: string | null;
  days_since_last: number | null;
  halted: boolean;
  halt_reason: string;
  sessions_required: number;
  strategies: string[];
  drift_reason: string;
}

interface Gate {
  name: string;
  ready: boolean;
  detail: string;
}

interface TradeStatus {
  can_trade: boolean;
  mode: string;
  gates: Gate[];
}

interface LimitRow {
  name: string;
  observed: number | null;
  threshold: number;
  passed: boolean | null;
}

interface Health {
  status: string;
  environment: string;
  live_enabled: boolean;
  kill_engaged: boolean;
  as_of: string;
}

function fmt(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Math.abs(value) < 1 ? value.toFixed(4) : value.toFixed(2);
}

export function SystemPanel() {
  const [venues, setVenues] = useState<Venue[]>([]);
  const [trade, setTrade] = useState<TradeStatus | null>(null);
  const [limits, setLimits] = useState<LimitRow[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [options, setOptions] = useState<number | null>(null);
  const [paper, setPaper] = useState<PaperStatus | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setBusy(true);
    void Promise.all([
      fetch("/api/venues").then((r) => (r.ok ? r.json() : [])),
      fetch("/api/trade/status").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/risk/limits").then((r) => (r.ok ? r.json() : [])),
      fetch("/api/health").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/options/underlyings").then((r) => (r.ok ? r.json() : [])),
      fetch("/api/paper/status").then((r) => (r.ok ? r.json() : null)),
    ])
      .then(([v, t, l, h, o, p]) => {
        setVenues(Array.isArray(v) ? v : []);
        setTrade(t);
        setLimits(Array.isArray(l) ? l : (l?.limits ?? []));
        setHealth(h);
        setOptions(Array.isArray(o) ? o.length : null);
        setPaper(p);
      })
      .catch(() => undefined)
      .finally(() => setBusy(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="system">
      <div className="charts-bar">
        <span className="charts-label">System</span>
        <button type="button" className="ghost" onClick={load} disabled={busy}>
          <Icon name="refresh" /> {busy ? "…" : "Refresh"}
        </button>
        <span className="charts-hint">Data coverage, gates and limits</span>
      </div>

      <div className="system-grid">
        <article className="block">
          <header className="panel-header">Environment</header>
          {health ? (
            <dl className="stats">
              <div className="stat">
                <dt>status</dt>
                <dd className="mono">{health.status}</dd>
              </div>
              <div className="stat">
                <dt>environment</dt>
                <dd className="mono">{health.environment}</dd>
              </div>
              <div className="stat">
                <dt>live enabled</dt>
                <dd className={health.live_enabled ? "mono up" : "mono"}>
                  {health.live_enabled ? "yes" : "no"}
                </dd>
              </div>
              <div className="stat">
                <dt>kill switch</dt>
                <dd className={health.kill_engaged ? "mono down" : "mono up"}>
                  {health.kill_engaged ? "ENGAGED" : "clear"}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="empty">API unavailable.</p>
          )}
        </article>

        <article className="block">
          <header className="panel-header">Data coverage</header>
          <table className="grid">
            <thead>
              <tr>
                <th>Venue</th>
                <th className="num">Sessions</th>
                <th>From</th>
                <th>To</th>
              </tr>
            </thead>
            <tbody>
              {venues.map((v) => (
                <tr key={v.venue}>
                  <td className="mono">{v.venue}</td>
                  <td className="num mono">{v.sessions.toLocaleString()}</td>
                  <td className="mono">{v.first ?? "—"}</td>
                  <td className="mono">{v.last ?? "—"}</td>
                </tr>
              ))}
              {options != null && (
                <tr>
                  <td className="mono">NFO</td>
                  <td className="num mono">—</td>
                  <td className="mono" colSpan={2}>
                    {options} underlyings
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </article>

        <article className="block">
          <header className="panel-header">
            Paper cycle
            {paper && (
              <span className={paper.started ? "badge live" : "badge blocked"}>
                {paper.halted ? "HALTED" : paper.started ? "RUNNING" : "NOT STARTED"}
              </span>
            )}
          </header>
          {paper ? (
            <>
              {/* Not-started is a state, not an empty screen. Showing zeros here
                  would read as "ran and did nothing", which is the opposite of
                  what a stalled scheduler means. */}
              {!paper.started ? (
                <p className="empty">
                  No cycle has ever written state. The M9 clock has not started — the
                  scheduled workflow runs from the default branch, so it only fires once
                  this is merged there.
                </p>
              ) : (
                <dl className="stats">
                  <div className="stat">
                    <dt>cycles</dt>
                    <dd className="mono">
                      {paper.cycles} / {paper.sessions_required}
                    </dd>
                  </div>
                  <div className="stat">
                    <dt>first session</dt>
                    <dd className="mono">{paper.first_session ?? "—"}</dd>
                  </div>
                  <div className="stat">
                    <dt>last session</dt>
                    <dd className="mono">{paper.last_session ?? "—"}</dd>
                  </div>
                  <div className="stat">
                    <dt>days since</dt>
                    {/* The number that says a scheduler stopped firing. A stalled
                        clock is indistinguishable from a quiet market otherwise. */}
                    <dd
                      className={
                        paper.days_since_last !== null && paper.days_since_last > 4
                          ? "mono down"
                          : "mono"
                      }
                    >
                      {paper.days_since_last ?? "—"}
                    </dd>
                  </div>
                  <div className="stat">
                    <dt>strategy</dt>
                    <dd className="mono">
                      {paper.strategies.length ? paper.strategies.join(", ") : "—"}
                    </dd>
                  </div>
                </dl>
              )}
              {paper.halted && (
                <p className="analytics-note text-critical">{paper.halt_reason}</p>
              )}
              {paper.drift_reason && (
                <p className="block-note">Drift vs backtest: {paper.drift_reason}</p>
              )}
            </>
          ) : (
            <p className="empty">API unavailable.</p>
          )}
        </article>

        <article className="block block-wide">
          <header className="panel-header">
            Live trading gates
            {trade && (
              <span className={trade.can_trade ? "badge live" : "badge blocked"}>{trade.mode}</span>
            )}
          </header>
          <ul className="check-list">
            {(trade?.gates ?? []).map((gate) => (
              <li key={gate.name} className={gate.ready ? "pass" : "fail"}>
                <span className="check-mark">
                  <Icon name={gate.ready ? "check" : "cross"} />
                </span>
                <span className="check-name">{gate.name}</span>
                <span className="check-detail">{gate.detail}</span>
              </li>
            ))}
          </ul>
        </article>

        <article className="block block-wide">
          <header className="panel-header">Risk limits</header>
          <table className="grid">
            <thead>
              <tr>
                <th>Limit</th>
                <th className="num">Observed</th>
                <th className="num">Threshold</th>
                <th>State</th>
              </tr>
            </thead>
            <tbody>
              {limits.map((row) => (
                <tr key={row.name}>
                  <td className="mono">{row.name}</td>
                  <td className="num mono">{fmt(row.observed)}</td>
                  <td className="num mono">{fmt(row.threshold)}</td>
                  <td className={row.passed === false ? "down" : row.passed ? "up" : ""}>
                    {/* Null is "not measured", not "passing". A book at rest has
                        no value for a limit checked per order, and rendering
                        that as a pass would claim a check that never ran. */}
                    {row.passed == null ? "Not measured" : row.passed ? "Within limit" : "BREACH"}
                  </td>
                </tr>
              ))}
              {limits.length === 0 && (
                <tr>
                  <td colSpan={4} className="empty">
                    No limits reported
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </article>
      </div>
    </div>
  );
}
