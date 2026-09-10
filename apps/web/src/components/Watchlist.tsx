/**
 * Watchlist — MASTER_PLAN §12.6.
 *
 * The panel a trading screen is actually read from: a short column of names,
 * the last close, the session move. Clicking one drives the chart beside it,
 * which is the whole reason it earns the space.
 *
 * **Close-to-close, not a live quote.** `/watchlist` reads the panel, so the
 * change shown is the same move the candles next to it are drawing. `/quotes`
 * exists for live delayed prices and is the right thing mid-session, but
 * putting a live price in a row whose change came from closes invites the
 * reader to subtract two numbers that do not belong together.
 */

import { useCallback, useEffect, useState } from "react";
import { Icon } from "./Icon";

const STORAGE_KEY = "neutron.watchlist.v1";

//: Nothing is watched until it is asked for. A list of names picked by
//: whoever wrote this file is not a watchlist, and a row you did not add is a
//: price you have no reason to be reading.
const NO_SYMBOLS: string[] = [];

interface Row {
  symbol: string;
  last: number;
  change_pct: number | null;
  volume: number;
  as_of: string;
}

function loadSymbols(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return NO_SYMBOLS;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return NO_SYMBOLS;
    return parsed.filter((s): s is string => typeof s === "string");
  } catch {
    return NO_SYMBOLS;
  }
}

export interface WatchlistProps {
  active?: string;
  venue: string;
  onPick: (symbol: string) => void;
}

//: Traded value in the units an Indian desk actually speaks: crore above a
//: crore, lakh below it. A raw rupee figure with nine digits is unreadable at
//: the size this list renders.
function advLabel(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "";
  if (value >= 1e7) return `${(value / 1e7).toFixed(1)} Cr`;
  return `${(value / 1e5).toFixed(1)} L`;
}

export function Watchlist({ active, venue, onPick }: WatchlistProps) {
  const [symbols, setSymbols] = useState<string[]>(loadSymbols);
  const [rows, setRows] = useState<Row[]>([]);
  const [adding, setAdding] = useState("");

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(symbols));
    } catch {
      // A watchlist that cannot be saved still has to work today.
    }
  }, [symbols]);

  useEffect(() => {
    if (symbols.length === 0) {
      setRows([]);
      return;
    }
    let cancelled = false;
    fetch(
      `/api/watchlist?symbols=${encodeURIComponent(symbols.join(","))}` +
        `&venue=${encodeURIComponent(venue)}`,
    )
      .then((r) => (r.ok ? r.json() : []))
      .then((payload) => !cancelled && setRows(Array.isArray(payload) ? payload : []))
      .catch(() => !cancelled && setRows([]));
    return () => {
      cancelled = true;
    };
  }, [symbols, venue]);

  //: Suggestions for the add box, from the same endpoint the header search
  //: uses. Typing a symbol from memory works and always did; the point is that
  //: a name half-remembered is now findable, and a typo produces no match here
  //: instead of an invisible row in the saved list.
  const [suggestions, setSuggestions] = useState<{ symbol: string; adv: number }[]>([]);

  useEffect(() => {
    const query = adding.trim();
    if (query.length < 2) {
      setSuggestions([]);
      return;
    }
    let cancelled = false;
    // Debounced for the same reason the header search is: a request per
    // keystroke puts a panel-wide ranking behind every letter.
    const timer = setTimeout(() => {
      fetch(`/api/symbols?q=${encodeURIComponent(query)}&venue=${encodeURIComponent(venue)}`)
        .then((r) => (r.ok ? r.json() : []))
        .then((rows) => !cancelled && setSuggestions(Array.isArray(rows) ? rows.slice(0, 8) : []))
        .catch(() => !cancelled && setSuggestions([]));
    }, 140);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [adding, venue]);

  const add = useCallback(() => {
    const symbol = adding.trim().toUpperCase();
    if (!symbol) return;
    setSymbols((prev) => (prev.includes(symbol) ? prev : [...prev, symbol]));
    setAdding("");
  }, [adding]);

  //: A symbol the panel does not carry never appears as a row, so it would sit
  //: in the saved list invisibly and un-removable. Shown as a miss instead.
  const missing = symbols.filter((s) => !rows.some((row) => row.symbol === s));

  return (
    <aside className="watchlist" aria-label="Watchlist">
      <header className="panel-header">Watchlist · {venue}</header>
      <ul className="watch-rows">
        {rows.map((row) => (
          <li key={row.symbol} className={row.symbol === active ? "on" : ""}>
            <button type="button" onClick={() => onPick(row.symbol)}>
              <span className="watch-symbol">{row.symbol}</span>
              <span className="watch-last">{row.last.toFixed(2)}</span>
              <span
                className={
                  row.change_pct == null ? "watch-change" : row.change_pct >= 0 ? "watch-change up" : "watch-change down"
                }
              >
                {row.change_pct == null
                  ? "—"
                  : `${row.change_pct >= 0 ? "+" : ""}${row.change_pct.toFixed(2)}%`}
              </span>
            </button>
            <button
              type="button"
              className="watch-remove"
              title={`Remove ${row.symbol}`}
              onClick={() => setSymbols((prev) => prev.filter((s) => s !== row.symbol))}
            >
              <Icon name="close" />
            </button>
          </li>
        ))}
        {missing.map((symbol) => (
          <li key={symbol} className="watch-missing">
            <span>{symbol}</span>
            <span className="muted">not found</span>
            <button
              type="button"
              className="watch-remove"
              onClick={() => setSymbols((prev) => prev.filter((s) => s !== symbol))}
            >
              <Icon name="close" />
            </button>
          </li>
        ))}
        {symbols.length === 0 && (
          <li className="watch-empty">No symbols added</li>
        )}
      </ul>
      <div className="watch-add">
        {suggestions.length > 0 && (
          <ul className="watch-suggest">
            {suggestions.map((match) => (
              <li key={match.symbol}>
                <button
                  type="button"
                  onClick={() => {
                    setSymbols((prev) =>
                      prev.includes(match.symbol) ? prev : [...prev, match.symbol],
                    );
                    setAdding("");
                    setSuggestions([]);
                  }}
                >
                  <b>{match.symbol}</b>
                  {/* Traded value, not a company name -- the endpoint carries
                      no name, and liquidity is the number that separates a
                      real listing from a thin lookalike with a similar
                      ticker. */}
                  <em>{advLabel(match.adv)}</em>
                </button>
              </li>
            ))}
          </ul>
        )}
        <input
          value={adding}
          placeholder="Add symbol"
          onChange={(event) => setAdding(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") add();
            if (event.key === "Escape") setSuggestions([]);
          }}
          spellCheck={false}
        />
        <button type="button" className="ghost" onClick={add}>
          <Icon name="plus" />
        </button>
      </div>
    </aside>
  );
}
