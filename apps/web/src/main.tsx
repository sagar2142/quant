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

const EMPTY: ConsoleState = {
  vitals: {
    feeds: [],
    stalenessSeconds: 0,
    dayPnl: 0,
    dayPnlPct: 0,
    drawdown: 0,
    ladderRungs: [-0.05, -0.08, -0.1],
    riskUtilisation: 0,
    killEngaged: false,
  },
  positions: [],
  trades: [],
  risk: [],
  breaks: [],
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
        const response = await fetch("/api/vitals");
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const vitals = await response.json();
        if (cancelled) return;
        setState((previous) => ({
          ...previous,
          latencyMs: performance.now() - started,
          vitals: {
            feeds: vitals.feeds,
            stalenessSeconds: vitals.staleness_seconds,
            dayPnl: Number(vitals.day_pnl),
            dayPnlPct: Number(vitals.day_pnl_pct),
            drawdown: Number(vitals.drawdown),
            ladderRungs: vitals.ladder_rungs.map(Number),
            riskUtilisation: Number(vitals.risk_utilisation),
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
            stalenessSeconds: previous.vitals.stalenessSeconds + POLL_INTERVAL_MS / 1000,
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
