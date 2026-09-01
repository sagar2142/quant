/**
 * Security search — MASTER_PLAN §12.6.
 *
 * **Ranked by liquidity, not alphabetically.** The name you want is almost
 * always one you can actually trade, and an alphabetical list buries it under
 * illiquid tickers that share a prefix. `/symbols` does the ranking server-side
 * over the same panel the charts draw from.
 *
 * **The venue is part of the search, not a filter applied afterwards.** A
 * symbol can exist on one exchange and not the other, and the same ISIN trades
 * at different prices on each — so "which exchange" has to be answered before
 * "which name" means anything.
 */

import { useEffect, useRef, useState } from "react";

interface Match {
  symbol: string;
  adv: number;
  bars: number;
}

interface Venue {
  venue: string;
  sessions: number;
  first: string | null;
  last: string | null;
}

function liquidity(adv: number): string {
  if (adv >= 1e7) return `${(adv / 1e7).toFixed(1)}Cr`;
  if (adv >= 1e5) return `${(adv / 1e5).toFixed(1)}L`;
  return adv.toFixed(0);
}

export interface SymbolSearchProps {
  symbol: string;
  venue: string;
  onSymbolChange: (symbol: string) => void;
  onVenueChange: (venue: string) => void;
}

export function SymbolSearch({ symbol, venue, onSymbolChange, onVenueChange }: SymbolSearchProps) {
  const [typing, setTyping] = useState("");
  const [matches, setMatches] = useState<Match[]>([]);
  const [venues, setVenues] = useState<Venue[]>([]);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const box = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetch("/api/venues")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setVenues(Array.isArray(rows) ? rows : []))
      .catch(() => setVenues([]));
  }, []);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    // Debounced: a keystroke per request would put a panel-wide ranking behind
    // every letter typed.
    const timer = setTimeout(() => {
      fetch(`/api/symbols?q=${encodeURIComponent(typing)}&venue=${encodeURIComponent(venue)}`)
        .then((r) => (r.ok ? r.json() : []))
        .then((rows) => {
          if (cancelled) return;
          setMatches(Array.isArray(rows) ? rows.slice(0, 12) : []);
          setHighlight(0);
        })
        .catch(() => !cancelled && setMatches([]));
    }, 140);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [typing, venue, open]);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (box.current && !box.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, []);

  const commit = (picked: string) => {
    onSymbolChange(picked.toUpperCase());
    setTyping("");
    setOpen(false);
  };

  return (
    <div className="symbol-search" ref={box}>
      {venues.length > 1 && (
        <div className="segmented small">
          {venues.map((v) => (
            <button
              key={v.venue}
              type="button"
              className={venue === v.venue ? "on" : ""}
              onClick={() => onVenueChange(v.venue)}
              title={`${v.sessions.toLocaleString()} sessions, ${v.first} to ${v.last}`}
            >
              {v.venue}
            </button>
          ))}
        </div>
      )}

      <input
        className="search-input"
        value={open ? typing : symbol}
        placeholder="search securities"
        spellCheck={false}
        onFocus={() => {
          setOpen(true);
          setTyping("");
        }}
        onChange={(event) => setTyping(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setHighlight((h) => Math.min(h + 1, matches.length - 1));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setHighlight((h) => Math.max(h - 1, 0));
          } else if (event.key === "Enter") {
            const chosen = matches[highlight];
            if (chosen) commit(chosen.symbol);
            else if (typing.trim()) commit(typing.trim());
          } else if (event.key === "Escape") {
            setOpen(false);
          }
        }}
        aria-label="Search securities"
      />

      {open && (
        <ul className="search-menu">
          {matches.map((match, index) => (
            <li key={match.symbol}>
              <button
                type="button"
                className={index === highlight ? "on" : ""}
                onMouseEnter={() => setHighlight(index)}
                onClick={() => commit(match.symbol)}
              >
                <b>{match.symbol}</b>
                <span className="muted">{liquidity(match.adv)}/day</span>
                <span className="muted">{match.bars} bars</span>
              </button>
            </li>
          ))}
          {matches.length === 0 && (
            <li className="search-empty">
              {typing ? `nothing on ${venue} starting "${typing.toUpperCase()}"` : "type to search"}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}
