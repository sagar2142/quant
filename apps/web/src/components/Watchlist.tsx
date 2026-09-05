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
        <input
          value={adding}
          placeholder="Add symbol"
          onChange={(event) => setAdding(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && add()}
          spellCheck={false}
        />
        <button type="button" className="ghost" onClick={add}>
          <Icon name="plus" />
        </button>
      </div>
    </aside>
  );
}
