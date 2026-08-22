/**
 * The vitals bar — MASTER_PLAN §12.7.
 *
 * 40px, fixed, present on every screen, never scrolls. The operator must be
 * able to read the system's true state in under three seconds, at 2am, under
 * stress — and that is only possible if the answer is always in the same place.
 *
 * Staleness above the critical threshold **also fires a Telegram alert**. The
 * UI is never the only alarm, because you will not be looking at it (§12.7).
 *
 * The kill button is deliberately oversized (40px, `--control-danger`) and
 * requires a typed confirmation. Every other control on screen is 24-32px; this
 * one is meant to be unmissable and impossible to hit by accident.
 */

import { useState } from "react";
import {
  directionGlyph,
  formatLevel,
  formatPercent,
  formatPnL,
  formatStaleness,
  signClass,
} from "../format";

export type FeedHealth = "ok" | "degraded" | "down";

/**
 * Every quantity is nullable, and the distinction is load-bearing: null means
 * "not measured" and renders as an em dash. These were previously fixed zeros,
 * so a book three days stale with a real drawdown displayed as live and flat.
 */
export interface Vitals {
  feeds: { name: string; health: FeedHealth }[];
  stalenessSeconds: number | null;
  dayPnl: number | null;
  dayPnlPct: number | null;
  /** Negative, matching the rungs: -0.0015 is a 0.15% drawdown. */
  drawdown: number | null;
  /** Ladder rungs, shallowest first, e.g. [-0.05, -0.08, -0.10]. */
  ladderRungs: number[];
  riskUtilisation: number | null;
  killEngaged: boolean;
}

/**
 * The paper loop runs once per session, so its staleness is judged in hours.
 * §12.7's 2s/10s thresholds describe a live tick feed; applied to a daily
 * batch they would paint the bar red permanently, which teaches the operator
 * to stop reading the colour.
 */
const CYCLE_WARN_SECONDS = 36 * 3600;
const CYCLE_CRITICAL_SECONDS = 96 * 3600;

function stalenessClass(seconds: number | null): string {
  // Unknown staleness is critical, not neutral: a cycle that has never run is
  // not a healthy one.
  if (seconds === null) return "text-critical";
  if (seconds > CYCLE_CRITICAL_SECONDS) return "text-critical";
  if (seconds > CYCLE_WARN_SECONDS) return "text-warn";
  return "text-secondary";
}

function healthClass(health: FeedHealth): string {
  return health === "ok"
    ? "text-profit"
    : health === "degraded"
      ? "text-warn"
      : "text-critical";
}

/**
 * Drawdown position on the pre-committed ladder (§8). Filled blocks show how
 * many rungs have engaged — the same information the risk engine acts on.
 */
function LadderMeter({ drawdown, rungs }: { drawdown: number | null; rungs: number[] }) {
  // An unmeasured drawdown engages nothing. Rungs are negative, and so is
  // `drawdown`, so this comparison reads "deeper than the rung".
  const engaged = drawdown === null ? 0 : rungs.filter((rung) => drawdown <= rung).length;
  return (
    <span className="ladder" title={`ladder: ${engaged}/${rungs.length} rungs engaged`}>
      {rungs.map((rung, index) => (
        <span
          key={rung}
          className={index < engaged ? "rung rung-on" : "rung rung-off"}
          aria-hidden="true"
        />
      ))}
    </span>
  );
}

function KillButton({
  engaged,
  onEngage,
}: {
  engaged: boolean;
  onEngage: (reason: string) => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState("");

  if (engaged) {
    // No release button, deliberately. Engaging a halt is fail-safe; releasing
    // one is the last thing standing between a known-bad book and the market,
    // and it should cost more than a click from the screen that halted it. The
    // badge names the path out so the operator is not left guessing.
    return (
      <span
        className="kill-engaged"
        title="Release is manual: POST /kill/release, or `python -m apps.cli.paper --clear-halt`"
      >
        KILLED
      </span>
    );
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className="kill"
        onClick={() => setConfirming(true)}
        aria-label="Engage kill switch"
      >
        KILL
      </button>
    );
  }

  // Typed confirmation, never a bare button (§12.8). The reason is mandatory
  // because an unattributed halt cannot be reviewed afterwards.
  return (
    <span className="kill-confirm">
      <input
        autoFocus
        value={typed}
        onChange={(event) => setTyped(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setConfirming(false);
            setTyped("");
          }
        }}
        placeholder="type KILL + reason"
        aria-label="Confirm kill switch"
      />
      <button
        type="button"
        className="kill"
        disabled={!typed.startsWith("KILL ") || typed.length < 8}
        onClick={() => {
          onEngage(typed.slice(5).trim());
          setConfirming(false);
          setTyped("");
        }}
      >
        CONFIRM
      </button>
    </span>
  );
}

export function VitalsBar({
  vitals,
  onKill,
}: {
  vitals: Vitals;
  onKill: (reason: string) => void;
}) {
  return (
    <header className="vitals" role="banner">
      <span className="feeds">
        {vitals.feeds.map((feed) => (
          <span key={feed.name} className="feed">
            <span className={`dot ${healthClass(feed.health)}`} aria-hidden="true">
              ●
            </span>
            <span className="feed-name">{feed.name}</span>
          </span>
        ))}
      </span>

      <span className={`stale ${stalenessClass(vitals.stalenessSeconds)}`}>
        stale {formatStaleness(vitals.stalenessSeconds)}
      </span>

      <span className="divider" aria-hidden="true" />

      {/* Sign and glyph both present, so direction survives a colourblind
          palette and a monochrome print (§12.2). */}
      <span className={`pnl num ${signClass(vitals.dayPnl)}`}>
        P&amp;L {formatPnL(vitals.dayPnl)} {directionGlyph(vitals.dayPnl)}
        <span className="pnl-pct">{formatPercent(vitals.dayPnlPct)}</span>
      </span>

      <span className="divider" aria-hidden="true" />

      <span className={`drawdown num ${signClass(vitals.drawdown)}`}>
        DD {formatPercent(vitals.drawdown)}
      </span>
      <LadderMeter drawdown={vitals.drawdown} rungs={vitals.ladderRungs} />

      <span className="divider" aria-hidden="true" />

      <span className="risk num">risk {formatLevel(vitals.riskUtilisation)}</span>

      <span className="spacer" />
      <KillButton engaged={vitals.killEngaged} onEngage={onKill} />
    </header>
  );
}
