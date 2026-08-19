/**
 * Ops console shell — MASTER_PLAN §12.6, §12.9.
 *
 * An instrument panel, not a website. Fixed viewport, panels scroll internally,
 * vitals always visible, keyboard-first.
 *
 * Screens are deliberately few. §13.8 is explicit that the polished UI belongs
 * at M8+, *after* the research loop has been operated, because you do not know
 * which screens matter until then. These are the ones paper trading actually
 * needs: what am I holding, what did I trade, what is the risk, does the broker
 * agree with me.
 */

import { useCallback, useEffect, useState } from "react";
import { Analytics } from "./components/Analytics";
import { Screener } from "./components/Screener";
import { VitalsBar, type Vitals } from "./components/VitalsBar";
import {
  directionGlyph,
  formatCount,
  formatLevel,
  formatPercent,
  formatPnL,
  formatPrice,
  signClass,
} from "./format";
import "./tokens.css";
import "./shell.css";

type Screen = "overview" | "screener" | "analytics" | "positions" | "blotter" | "risk" | "reconcile";

const SCREENS: { id: Screen; label: string; icon: string; key: string }[] = [
  { id: "overview", label: "Overview", icon: "◧", key: "o" },
  { id: "screener", label: "Screener", icon: "⌗", key: "s" },
  { id: "analytics", label: "Analytics", icon: "∿", key: "a" },
  { id: "positions", label: "Positions", icon: "▤", key: "p" },
  { id: "blotter", label: "Blotter", icon: "▦", key: "b" },
  { id: "risk", label: "Risk", icon: "▲", key: "r" },
  { id: "reconcile", label: "Reconcile", icon: "⇄", key: "c" },
];

export interface Position {
  instrumentId: string;
  symbol: string;
  quantity: number;
  averagePrice: number;
  lastPrice: number;
  unrealisedPnl: number;
  weightPct: number;
  cluster: string;
}

export interface Trade {
  eventTime: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  costs: number;
  state: string;
}

export interface RiskRow {
  name: string;
  observed: number;
  threshold: number;
  passed: boolean;
}

export interface Break {
  instrumentId: string;
  kind: string;
  internal: number;
  broker: number;
}

export interface ConsoleState {
  vitals: Vitals;
  positions: Position[];
  trades: Trade[];
  risk: RiskRow[];
  breaks: Break[];
  environment: "dev" | "paper" | "live";
  gitSha: string;
  latencyMs: number;
}

function Panel({
  title,
  children,
  flush = false,
  footer,
}: {
  title: string;
  children: React.ReactNode;
  flush?: boolean;
  footer?: string;
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <span>{title}</span>
      </div>
      <div className={flush ? "panel-body flush" : "panel-body"}>{children}</div>
      {footer ? <div className="panel-footer">{footer}</div> : null}
    </section>
  );
}

function PositionsTable({ positions }: { positions: Position[] }) {
  if (positions.length === 0) {
    return <div className="empty">No open positions.</div>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Symbol</th>
          <th className="num">Qty</th>
          <th className="num">Avg</th>
          <th className="num">Last</th>
          <th className="num">Unrealised</th>
          <th className="num">% NAV</th>
          <th>Cluster</th>
        </tr>
      </thead>
      <tbody>
        {positions.map((position) => (
          <tr key={position.instrumentId}>
            <td>{position.symbol}</td>
            <td className={`num ${signClass(position.quantity)}`}>
              {formatCount(position.quantity)}
            </td>
            <td className="num">{formatPrice(position.averagePrice)}</td>
            <td className="num">{formatPrice(position.lastPrice)}</td>
            <td className={`num ${signClass(position.unrealisedPnl)}`}>
              {formatPnL(position.unrealisedPnl)} {directionGlyph(position.unrealisedPnl)}
            </td>
            <td className="num">{formatLevel(position.weightPct)}</td>
            <td className="text-secondary">{position.cluster || "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Blotter({ trades }: { trades: Trade[] }) {
  if (trades.length === 0) {
    return <div className="empty">No trades today.</div>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Symbol</th>
          <th>Side</th>
          <th className="num">Qty</th>
          <th className="num">Price</th>
          <th className="num">Costs</th>
          <th>State</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((trade, index) => (
          <tr key={`${trade.eventTime}-${index}`}>
            <td className="mono">{trade.eventTime}</td>
            <td>{trade.symbol}</td>
            <td className={trade.side === "BUY" ? "text-profit" : "text-loss"}>
              {trade.side}
            </td>
            <td className="num">{formatCount(trade.quantity)}</td>
            <td className="num">{formatPrice(trade.price)}</td>
            <td className="num text-secondary">{formatPrice(trade.costs)}</td>
            <td className="text-secondary">{trade.state}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RiskTable({ rows }: { rows: RiskRow[] }) {
  if (rows.length === 0) {
    return <div className="empty">No limits configured.</div>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Limit</th>
          <th className="num">Observed</th>
          <th className="num">Threshold</th>
          <th className="num">Used</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const used = row.threshold === 0 ? 0 : row.observed / row.threshold;
          return (
            <tr key={row.name}>
              <td>{row.name}</td>
              <td className="num">{formatPrice(row.observed, 4)}</td>
              <td className="num text-secondary">{formatPrice(row.threshold, 4)}</td>
              <td className={`num ${row.passed ? "" : "text-critical"}`}>
                {formatLevel(used)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Any non-empty diff is a red banner. An unexplained break halts trading (§9). */
function Reconciliation({ breaks }: { breaks: Break[] }) {
  if (breaks.length === 0) {
    return <div className="empty">Broker and internal records agree.</div>;
  }
  return (
    <>
      <div className="panel-body" style={{ paddingBottom: 0 }}>
        <strong className="text-critical">
          {breaks.length} unexplained break(s) — halt new orders and find the cause.
        </strong>
      </div>
      <table>
        <thead>
          <tr>
            <th>Instrument</th>
            <th>Kind</th>
            <th className="num">Internal</th>
            <th className="num">Broker</th>
            <th className="num">Diff</th>
          </tr>
        </thead>
        <tbody>
          {breaks.map((row) => (
            <tr key={`${row.instrumentId}-${row.kind}`}>
              <td>{row.instrumentId}</td>
              <td className="text-critical">{row.kind}</td>
              <td className="num">{formatPrice(row.internal, 4)}</td>
              <td className="num">{formatPrice(row.broker, 4)}</td>
              <td className="num text-critical">
                {formatPrice(row.internal - row.broker, 4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function App({
  state,
  onKill,
}: {
  state: ConsoleState;
  onKill: (reason: string) => void;
}) {
  const [screen, setScreen] = useState<Screen>("overview");
  // Set when a screener row is clicked, so the analytics screen opens on that
  // name. Keyed on the component so it remounts and refetches.
  const [picked, setPicked] = useState<string | null>(null);

  // Keyboard-first: `g` then a letter. The mouse is optional (§12.8).
  const handleKey = useCallback((event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement) return;
    const match = SCREENS.find((s) => s.key === event.key);
    if (match) setScreen(match.id);
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  const totalUnrealised = state.positions.reduce((sum, p) => sum + p.unrealisedPnl, 0);

  return (
    <div className="shell">
      <VitalsBar vitals={state.vitals} onKill={onKill} />

      <div className="shell-body">
        <nav className="nav" aria-label="Screens">
          {SCREENS.map((item) => (
            <button
              key={item.id}
              type="button"
              title={`${item.label} (${item.key})`}
              aria-current={screen === item.id ? "page" : undefined}
              onClick={() => setScreen(item.id)}
            >
              {item.icon}
            </button>
          ))}
        </nav>

        {screen === "overview" ? (
          <div className="workspace grid-2x2">
            <Panel
              title="Positions"
              flush
              footer={`${state.positions.length} open · unrealised ${formatPnL(totalUnrealised)}`}
            >
              <PositionsTable positions={state.positions} />
            </Panel>
            <Panel title="Risk" flush>
              <RiskTable rows={state.risk} />
            </Panel>
            <Panel title="Today's fills" flush footer={`${state.trades.length} fill(s)`}>
              <Blotter trades={state.trades} />
            </Panel>
            <Panel title="Reconciliation" flush>
              <Reconciliation breaks={state.breaks} />
            </Panel>
          </div>
        ) : (
          <div className="workspace">
            {screen === "screener" ? (
              <Panel title="Screener" flush>
                <Screener
                  onPick={(symbol) => {
                    setPicked(symbol);
                    setScreen("analytics");
                  }}
                />
              </Panel>
            ) : null}
            {screen === "analytics" ? (
              <Panel title="Analytics" flush>
                <Analytics key={picked ?? "default"} initialSymbols={picked ?? undefined} />
              </Panel>
            ) : null}
            {screen === "positions" ? (
              <Panel title="Positions" flush footer={`${state.positions.length} open`}>
                <PositionsTable positions={state.positions} />
              </Panel>
            ) : null}
            {screen === "blotter" ? (
              <Panel title="Blotter" flush footer={`${state.trades.length} fill(s)`}>
                <Blotter trades={state.trades} />
              </Panel>
            ) : null}
            {screen === "risk" ? (
              <Panel title="Risk limits" flush>
                <RiskTable rows={state.risk} />
              </Panel>
            ) : null}
            {screen === "reconcile" ? (
              <Panel title="Reconciliation" flush>
                <Reconciliation breaks={state.breaks} />
              </Panel>
            ) : null}
          </div>
        )}
      </div>

      <footer className="status">
        {/* Not cosmetic: this is what stops a test order reaching production. */}
        <span className={`env-badge env-${state.environment}`}>
          {state.environment.toUpperCase()}
        </span>
        <span className="mono">{state.gitSha.slice(0, 7)}</span>
        <span>latency {state.latencyMs.toFixed(0)}ms</span>
        <span className="text-secondary">
          drawdown {formatPercent(state.vitals.drawdown)}
        </span>
      </footer>
    </div>
  );
}
