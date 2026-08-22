/**
 * Console entry point — MASTER_PLAN §13.6.
 *
 * Polls the API rather than holding trading state. The console is a *client*:
 * closing it must never affect trading, and it must be safe to rebuild without
 * touching a line of trading logic.
 */

import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { App, type ConsoleState } from "./App";

/** Frequent enough to feel live, slow enough to be free. */
const POLL_INTERVAL_MS = 1000;

/**
 * A JSON number, or null when the API could not measure it.
 *
 * `Number(null)` is `0`, so mapping these fields with `Number` silently turned
 * "not measured" into "measured, and it is zero" — a flat P&L, a healthy
 * drawdown, a fresh feed. The whole point of the nullable API is lost at the
 * first coercion, so it happens here and nowhere else.
 */
function nullable(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

const EMPTY: ConsoleState = {
  equity: [],
  // Null, not zero. Before the first poll lands nothing has been measured,
  // and a bar reading "P&L +₹0.00, DD 0.00%, fresh" is a claim about a book
  // this process has never seen.
  vitals: {
    feeds: [],
    stalenessSeconds: null,
    dayPnl: null,
    dayPnlPct: null,
    drawdown: null,
    ladderRungs: [-0.05, -0.08, -0.1],
    riskUtilisation: null,
    killEngaged: false,
  },
  positions: [],
  trades: [],
  risk: [],
  breaks: [],
  reconciliation: { checked: false, halted: false, haltReason: "", cycles: 0 },
  environment: "dev",
  gitSha: "unknown",
  latencyMs: 0,
};

function Console() {
  const [state, setState] = useState<ConsoleState>(EMPTY);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      const started = performance.now();
      try {
        // Vitals, the book and the limits together: three cheap reads, and a
        // console showing a stale position next to a fresh P&L is worse than
        // one that refreshes them as a set.
        const [vitalsRes, bookRes, limitsRes, equityRes, fillsRes, reconRes, healthRes] =
          await Promise.all([
            fetch("/api/vitals"),
            fetch("/api/book"),
            fetch("/api/risk/limits"),
            fetch("/api/equity"),
            fetch("/api/fills"),
            fetch("/api/reconciliation"),
            fetch("/api/health"),
          ]);
        if (!vitalsRes.ok) throw new Error(`HTTP ${vitalsRes.status}`);
        const vitals = await vitalsRes.json();
        const book = bookRes.ok ? await bookRes.json() : null;
        const limits = limitsRes.ok ? await limitsRes.json() : [];
        const equityLog = equityRes.ok ? await equityRes.json() : [];
        const fillLog = fillsRes.ok ? await fillsRes.json() : [];
        const recon = reconRes.ok ? await reconRes.json() : null;
        const health = healthRes.ok ? await healthRes.json() : null;
        if (cancelled) return;
        setState((previous) => ({
          ...previous,
          // Oldest first, as written. Rows that fail to parse are dropped
          // rather than plotted as zero — a gap in the curve is honest, a
          // spike to the axis is not.
          equity: (equityLog as Record<string, string>[])
            .map((row) => Number(row.equity))
            .filter((value) => Number.isFinite(value)),
          // The badge said DEV unconditionally. It exists to tell the
          // operator which environment they are acting in, so asserting one
          // is the single worst thing it could do. Unknown stays as it was.
          environment: health?.environment ?? previous.environment,
          // Most recent first: a blotter is read from the top.
          trades: (fillLog as Record<string, string>[])
            .map((f) => ({
              eventTime: String(f.at),
              symbol: String(f.symbol),
              side: f.side === "SELL" ? ("SELL" as const) : ("BUY" as const),
              quantity: Number(f.quantity),
              price: Number(f.price),
              costs: Number(f.costs),
              state: "FILLED",
            }))
            .reverse(),
          // Absent means the endpoint is unreachable, which is not the same as
          // a clean reconciliation — so `checked` stays false.
          reconciliation: {
            checked: Boolean(recon?.checked),
            halted: Boolean(recon?.halted),
            haltReason: String(recon?.halt_reason ?? ""),
            cycles: Number(recon?.cycles ?? 0),
          },
          positions: (book?.positions ?? []).map((p: Record<string, number | string>) => ({
            instrumentId: String(p.instrument_id),
            symbol: String(p.symbol),
            quantity: Number(p.quantity),
            averagePrice: Number(p.average_price),
            lastPrice: Number(p.last_price),
            unrealisedPnl: Number(p.unrealised_pnl),
            weightPct: Number(p.weight_pct),
            cluster: "",
          })),
          // Read from the API, never invented. These two fields were once
          // hardcoded to `0` and `true`, so the Risk screen reported every
          // limit as passing with nothing used no matter what the book held.
          risk: (limits as Record<string, unknown>[]).map((l) => ({
            name: String(l.name),
            observed: l.observed === null ? null : Number(l.observed),
            threshold: Number(l.threshold),
            passed: l.passed === null ? null : Boolean(l.passed),
          })),
          latencyMs: performance.now() - started,
          // `Number(null)` is 0, so every one of these goes through `nullable`
          // instead. Coercing here is what made an unmeasured book look flat.
          vitals: {
            feeds: vitals.feeds,
            stalenessSeconds: nullable(vitals.staleness_seconds),
            dayPnl: nullable(vitals.day_pnl),
            dayPnlPct: nullable(vitals.day_pnl_pct),
            drawdown: nullable(vitals.drawdown),
            ladderRungs: vitals.ladder_rungs.map(Number),
            riskUtilisation: nullable(vitals.risk_utilisation),
            killEngaged: vitals.kill_engaged,
          },
        }));
      } catch {
        // A failed poll marks the feed down rather than blanking the screen.
        // Stale-but-labelled beats empty-and-ambiguous: the operator must be
        // able to tell "nothing is happening" from "I cannot see".
        if (cancelled) return;
        setState((previous) => ({
          ...previous,
          vitals: {
            ...previous.vitals,
            feeds: previous.vitals.feeds.map((f) => ({ ...f, health: "down" })),
            // Staleness keeps climbing while the API is unreachable. If it was
            // never known, it stays unknown — inventing a starting point would
            // report a precise age for a book we have never read.
            stalenessSeconds:
              previous.vitals.stalenessSeconds === null
                ? null
                : previous.vitals.stalenessSeconds + POLL_INTERVAL_MS / 1000,
          },
        }));
      }
    }

    void poll();
    const timer = window.setInterval(() => void poll(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  async function handleKill(reason: string) {
    await fetch("/api/kill", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason, operator: "console" }),
    });
  }

  return <App state={state} onKill={(reason) => void handleKill(reason)} />;
}

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <Console />
    </StrictMode>,
  );
}
