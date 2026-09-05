/**
 * The global command bar — MASTER_PLAN §12.9.
 *
 * One symbol, one venue, held by the shell and shared by every workspace. The
 * previous console asked for a symbol separately on each screen, so moving
 * from a chart to its analysis to a ticket meant typing the same ticker three
 * times and risking three different answers to "which name am I looking at".
 *
 * **The venue lives here for the same reason.** A name exists on one exchange
 * and not the other, and the same ISIN trades at different prices on each.
 * Choosing it per screen invites the case where the chart is NSE and the
 * ticket is BSE and nothing says so.
 */

import { useEffect, useRef, useState } from "react";
import { Logo } from "./Logo";

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

export interface CommandBarProps {
  symbol: string;
  venue: string;
  onSymbol: (symbol: string) => void;
  onVenue: (venue: string) => void;
  /** Rendered at the right — vitals, kill switch, whatever the shell owns. */
  children?: React.ReactNode;
}

export function CommandBar({ symbol, venue, onSymbol, onVenue, children }: CommandBarProps) {
  const [typing, setTyping] = useState("");
  const [matches, setMatches] = useState<Match[]>([]);
  const [venues, setVenues] = useState<Venue[]>([]);
  const [open, setOpen] = useState(false);
  const [highlight, setHighlight] = useState(0);
  const [hasOptions, setHasOptions] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  const input = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    fetch("/api/venues")
      .then((r) => (r.ok ? r.json() : []))
      .then((rows) => setVenues(Array.isArray(rows) ? rows : []))
      .catch(() => setVenues([]));
  }, []);

  //: Ctrl+K from anywhere. A terminal is used with the hands on the keyboard,
  //: and reaching for a search box is the one interaction that must never
  //: require the pointer.
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
        setTyping("");
        input.current?.focus();
        input.current?.select();
      }
    };
    document.addEventListener("keydown", shortcut);
    return () => document.removeEventListener("keydown", shortcut);
  }, []);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      fetch(`/api/symbols?q=${encodeURIComponent(typing)}&venue=${encodeURIComponent(venue)}`)
        .then((r) => (r.ok ? r.json() : []))
        .then((rows) => {
          if (cancelled) return;
          setMatches(Array.isArray(rows) ? rows.slice(0, 10) : []);
          setHighlight(0);
        })
        .catch(() => !cancelled && setMatches([]));
    }, 130);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [typing, venue, open]);

  //: Whether the current name has listed derivatives, shown as a badge. Only
  //: 216 of roughly three thousand do, and knowing before you look for a chain
  //: saves the trip.
  useEffect(() => {
    if (!symbol) {
      setHasOptions(false);
      return;
    }
    let cancelled = false;
    fetch(`/api/options/underlyings?q=${encodeURIComponent(symbol)}`)
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: string[]) => {
        if (!cancelled) setHasOptions(Array.isArray(rows) && rows.includes(symbol.toUpperCase()));
      })
      .catch(() => !cancelled && setHasOptions(false));
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (box.current && !box.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, []);

  const commit = (picked: string) => {
    onSymbol(picked.toUpperCase());
    setTyping("");
    setOpen(false);
    input.current?.blur();
  };

  return (
    <header className="commandbar">
      <div className="brand">
        <Logo size={18} />
        <span className="brand-name">NEUTRON</span>
      </div>

      <div className="segmented small venue-pick">
        {venues.map((v) => (
          <button
            key={v.venue}
            type="button"
            className={venue === v.venue ? "on" : ""}
            onClick={() => onVenue(v.venue)}
            title={`${v.sessions.toLocaleString()} sessions · ${v.first} to ${v.last}`}
          >
            {v.venue}
          </button>
        ))}
      </div>

      <div className="cmd-search" ref={box}>
        <input
          ref={input}
          className="cmd-input"
          value={open ? typing : symbol}
          placeholder="search securities   ⌃K"
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
              input.current?.blur();
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

      {symbol && (
        <span className="cmd-symbol">
          {symbol}
          {hasOptions && (
            <span className="cmd-badge" title="This name has listed options">
              F&amp;O
            </span>
          )}
        </span>
      )}

      <div className="cmd-right">{children}</div>
    </header>
  );
}
