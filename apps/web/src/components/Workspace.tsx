/**
 * Multi-chart workspace — MASTER_PLAN §12.6.
 *
 * One chart at a time answers "what is this name doing". A trading screen has
 * to answer "what is happening", which is a question about several names at
 * once — a position, its sector, the index it is being compared against.
 *
 * **Panels are independent, and that is deliberate.** Each carries its own
 * symbol, its own lookback and its own scale. Linking them would make one
 * comparison easy and every other one impossible; two panels showing the same
 * name over different windows is a common and useful thing to want.
 *
 * **The layout is saved per browser.** A workspace you have to rebuild every
 * morning is one you stop using, and it is exactly the kind of state that
 * belongs in `localStorage`: local to one screen, worthless to anyone else,
 * and harmless to lose.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Chart } from "./Chart";
import { Icon } from "./Icon";
import { Popout, usePopouts } from "./Popout";
import { Watchlist } from "./Watchlist";

const STORAGE_KEY = "neutron.workspace.v2";

/** Exchanges the API serves. Fetched rather than hardcoded — they do not
 *  cover the same period, and the bar says so. */
interface Venue {
  venue: string;
  sessions: number;
  first: string | null;
  last: string | null;
}

/** Lookbacks offered per pane. 0 is the whole panel history.
 *
 *  The short ranges are in sessions for daily candles and in days for
 *  intraday, which is the same number for 1D and 5D and diverges after that —
 *  the label means "how much history", and the request carries whichever the
 *  chosen interval reads. */
const RANGES: { label: string; sessions: number }[] = [
  { label: "1D", sessions: 1 },
  { label: "5D", sessions: 5 },
  { label: "1M", sessions: 21 },
  { label: "3M", sessions: 63 },
  { label: "6M", sessions: 126 },
  { label: "1Y", sessions: 252 },
  { label: "3Y", sessions: 756 },
  { label: "Max", sessions: 0 },
];

interface IntervalInfo {
  key: string;
  label: string;
  intraday: boolean;
  available: boolean;
  detail: string;
}

const LAYOUTS: { id: Layout; label: string; columns: number }[] = [
  { id: "single", label: "1", columns: 1 },
  { id: "split", label: "2", columns: 2 },
  { id: "quad", label: "4", columns: 2 },
  { id: "six", label: "6", columns: 3 },
];

type Layout = "single" | "split" | "quad" | "six";

interface Panel {
  id: string;
  symbol: string;
  sessions: number;
  logScale: boolean;
  venue: string;
  live?: boolean;
  interval?: string;
}

interface Saved {
  layout: Layout;
  panels: Panel[];
}

//: One empty pane. The panes exist so there is somewhere to put a chart; which
//: name goes in them is not a decision this file gets to make.
const EMPTY_PANELS: Panel[] = [
  { id: "p1", symbol: "", sessions: 252, logScale: false, venue: "NSE" },
];

function load(): Saved {
  // Wrapped because storage throws outright in a private window or when site
  // data is blocked, and a chart grid that fails to mount over a saved layout
  // is a worse outcome than losing the layout.
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { layout: "single", panels: EMPTY_PANELS };
    const parsed = JSON.parse(raw) as Saved;
    if (!Array.isArray(parsed.panels) || parsed.panels.length === 0) {
      return { layout: "single", panels: EMPTY_PANELS };
    }
    // A layout saved before venues existed carries no venue at all. Defaulted
    // rather than discarded: losing someone's workspace to add a field is a
    // worse trade than assuming the exchange they already had.
    return {
      layout: parsed.layout,
      panels: parsed.panels.map((panel) => ({ ...panel, venue: panel.venue ?? "NSE" })),
    };
  } catch {
    return { layout: "single", panels: EMPTY_PANELS };
  }
}

function visibleCount(layout: Layout): number {
  return layout === "single" ? 1 : layout === "split" ? 2 : layout === "quad" ? 4 : 6;
}

interface SymbolPickerProps {
  value: string;
  venue: string;
  onPick: (symbol: string) => void;
}

/** Type-ahead over the panel's own symbol list. */
function SymbolPicker({ value, venue, onPick }: SymbolPickerProps) {
  const [typing, setTyping] = useState("");
  const [matches, setMatches] = useState<{ symbol: string; name?: string }[]>([]);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!typing) {
      setMatches([]);
      return;
    }
    let cancelled = false;
    const timer = setTimeout(() => {
      fetch(`/api/symbols?q=${encodeURIComponent(typing)}&venue=${encodeURIComponent(venue)}`)
        .then((r) => (r.ok ? r.json() : []))
        .then((rows) => !cancelled && setMatches(Array.isArray(rows) ? rows.slice(0, 8) : []))
        .catch(() => !cancelled && setMatches([]));
    }, 140);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [typing, venue]);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (box.current && !box.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, []);

  const commit = (symbol: string) => {
    onPick(symbol.toUpperCase());
    setTyping("");
    setOpen(false);
  };

  return (
    <div className="symbol-picker" ref={box}>
      <input
        className="symbol-input"
        value={open ? typing : value}
        placeholder="symbol"
        onFocus={() => {
          setOpen(true);
          setTyping("");
        }}
        onChange={(event) => setTyping(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && typing) commit(matches[0]?.symbol ?? typing);
          if (event.key === "Escape") setOpen(false);
        }}
        aria-label="Symbol"
      />
      {open && matches.length > 0 && (
        <ul className="symbol-menu">
          {matches.map((m) => (
            <li key={m.symbol}>
              <button type="button" onClick={() => commit(m.symbol)}>
                <b>{m.symbol}</b>
                {m.name && <span>{m.name}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export interface WorkspaceProps {
  /** Send this symbol to the order ticket. */
  onTrade?: (symbol: string) => void;
}

export function Workspace({ onTrade }: WorkspaceProps) {
  const [state, setState] = useState<Saved>(load);
  //: Which pane, if any, has the window to itself. Not persisted: an
  //: expanded pane is a thing you are doing now, not a layout.
  const [full, setFull] = useState<string | null>(null);
  //: Panes living in their own OS window, for a second monitor.
  const popouts = usePopouts();
  const [venues, setVenues] = useState<Venue[]>([]);
  //: Which candle sizes exist, and which are usable. Asked once: intraday
  //: needs a broker, and a button that silently returns nothing is worse than
  //: one that says why it cannot.
  const [intervals, setIntervals] = useState<IntervalInfo[]>([]);

  useEffect(() => {
    if (!full) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFull(null);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [full]);

  useEffect(() => {
    fetch("/api/intervals")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setIntervals(Array.isArray(rows) ? rows : []))
      .catch(() => setIntervals([]));
  }, []);

  useEffect(() => {
    fetch("/api/venues")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setVenues(Array.isArray(rows) ? rows : []))
      .catch(() => setVenues([]));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // A workspace that cannot be saved still has to work.
    }
  }, [state]);

  const update = useCallback((id: string, patch: Partial<Panel>) => {
    setState((prev) => ({
      ...prev,
      panels: prev.panels.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    }));
  }, []);

  const addPanel = () => {
    setState((prev) => ({
      ...prev,
      panels: [
        ...prev.panels,
        {
          id: `p${Date.now()}`,
          symbol: "",
          sessions: 252,
          logScale: false,
          venue: prev.panels[prev.panels.length - 1]?.venue ?? "NSE",
        },
      ],
    }));
  };

  const removePanel = (id: string) => {
    setState((prev) => ({
      ...prev,
      // Never below one: an empty workspace offers no way back to a full one.
      panels: prev.panels.length > 1 ? prev.panels.filter((p) => p.id !== id) : prev.panels,
    }));
  };

  const visible = state.panels.slice(0, visibleCount(state.layout));
  const shown = full ? visible.filter((panel) => panel.id === full) : visible;
  const columns = full ? 1 : (LAYOUTS.find((l) => l.id === state.layout)?.columns ?? 2);
  const rows = Math.ceil(shown.length / columns);
  const paneHeight = Math.max(220, Math.floor((window.innerHeight - 170) / rows));

  return (
    <section className="charts" aria-label="Charts">
      <header className="charts-bar">
        <span className="charts-label">Layout</span>
        <div className="segmented">
          {LAYOUTS.map((l) => (
            <button
              key={l.id}
              type="button"
              className={state.layout === l.id ? "on" : ""}
              onClick={() => setState((prev) => ({ ...prev, layout: l.id }))}
              title={`${l.label} charts`}
            >
              {l.label}
            </button>
          ))}
        </div>
        <button type="button" className="ghost" onClick={addPanel}>
          <Icon name="plus" /> Add chart
        </button>
        {popouts.blocked && (
          <span className="popout-blocked" onClick={popouts.clearBlocked}>
            {popouts.blocked}
          </span>
        )}
        <span className="charts-hint">
          Scroll to zoom · Drag to pan · Double-click to reset
        </span>
      </header>

      <div className="charts-body">
        <Watchlist
          venue={shown[0]?.venue ?? "NSE"}
          active={shown[0]?.symbol}
          onPick={(symbol) =>
            //: Drives the first pane. The others keep whatever they were
            //: showing, so a watchlist click compares against the rest of the
            //: workspace rather than replacing it.
            shown[0] && update(shown[0].id, { symbol })
          }
        />
        <div className="charts-grid" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
        {shown.map((panel) => (
          <article className="chart-tile" key={panel.id}>
            <header className="panel-bar">
              {venues.length > 1 && (
                <div className="segmented small">
                  {venues.map((v) => (
                    <button
                      key={v.venue}
                      type="button"
                      className={panel.venue === v.venue ? "on" : ""}
                      onClick={() => update(panel.id, { venue: v.venue })}
                      title={`${v.sessions} sessions, ${v.first} to ${v.last}`}
                    >
                      {v.venue}
                    </button>
                  ))}
                </div>
              )}
              <SymbolPicker
                venue={panel.venue}
                value={panel.symbol}
                onPick={(s) => update(panel.id, { symbol: s })}
              />
              {intervals.length > 0 && (
                <div className="segmented small">
                  {intervals.map((entry) => (
                    <button
                      key={entry.key}
                      type="button"
                      className={(panel.interval ?? "1d") === entry.key ? "on" : ""}
                      disabled={!entry.available}
                      onClick={() => update(panel.id, { interval: entry.key })}
                      title={entry.detail}
                    >
                      {entry.label}
                    </button>
                  ))}
                </div>
              )}
              <div className="segmented small">
                {RANGES.map((r) => (
                  <button
                    key={r.label}
                    type="button"
                    className={panel.sessions === r.sessions ? "on" : ""}
                    onClick={() => update(panel.id, { sessions: r.sessions })}
                  >
                    {r.label}
                  </button>
                ))}
              </div>
              <button
                type="button"
                className={panel.live ? "ghost on" : "ghost"}
                onClick={() => update(panel.id, { live: !panel.live })}
                title="Poll a delayed vendor quote for this name"
              >
                live
              </button>
              <button
                type="button"
                className={panel.logScale ? "ghost on" : "ghost"}
                onClick={() => update(panel.id, { logScale: !panel.logScale })}
                title="Logarithmic price axis"
              >
                log
              </button>
              {onTrade && (
                <button type="button" className="ghost" onClick={() => onTrade(panel.symbol)}>
                  trade
                </button>
              )}
              <button
                type="button"
                className={popouts.windows.has(panel.id) ? "ghost on" : "ghost"}
                onClick={() => popouts.toggle(panel.id, panel.symbol || 'Chart')}
                title={
                  popouts.windows.has(panel.id)
                    ? "Bring back into this window"
                    : "Open this chart in a separate window"
                }
              >
                <Icon name="popout" />
              </button>
              <button
                type="button"
                className={full === panel.id ? "ghost on" : "ghost"}
                onClick={() => setFull((current) => (current === panel.id ? null : panel.id))}
                title={full === panel.id ? "Restore (Esc)" : "Expand to full window"}
              >
                <Icon name={full === panel.id ? "collapse" : "expand"} />
              </button>
              <button
                type="button"
                className="ghost danger"
                onClick={() => removePanel(panel.id)}
                title="Close this chart"
                disabled={state.panels.length <= 1}
              >
                <Icon name="close" />
              </button>
            </header>
            {popouts.windows.has(panel.id) ? (
              <>
                <p className="popout-note">
                  Open in a separate window
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => popouts.toggle(panel.id, panel.symbol || 'Chart')}
                  >
                    Return
                  </button>
                </p>
                <Popout
                  title={`${panel.symbol || "chart"} · ${panel.venue}`}
                  target={popouts.windows.get(panel.id) as Window}
                  onClose={() => popouts.close(panel.id)}
                >
                  <div className="popout-chart">
                    <Chart
                      symbol={panel.symbol}
                      venue={panel.venue}
                      sessions={panel.sessions}
                      interval={panel.interval ?? "1d"}
                      logScale={panel.logScale}
                      live={panel.live ?? false}
                      height={720}
                    />
                  </div>
                </Popout>
              </>
            ) : (
              <Chart
                symbol={panel.symbol}
                venue={panel.venue}
                sessions={panel.sessions}
                logScale={panel.logScale}
                live={panel.live ?? false}
                height={full === panel.id ? Math.max(320, window.innerHeight - 150) : paneHeight}
              />
            )}
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
