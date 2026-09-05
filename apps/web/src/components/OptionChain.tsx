/**
 * Option chain — MASTER_PLAN §13.4.
 *
 * Calls left, puts right, strikes down the middle, the money marked. The
 * layout is conventional because a chain is read by shape: where the open
 * interest sits, where the skew steepens, which strikes have not traded.
 *
 * **Implied volatility and the Greeks are computed, not quoted.** The bhavcopy
 * carries a price and nothing else, so everything here is inverted from that
 * price under a rate the panel shows and lets you change. A blank IV is a real
 * answer — a far strike that has not traded prints a stale price no volatility
 * explains — and the row is still shown, because its open interest is often
 * the most interesting thing on the screen.
 */

import { useCallback, useEffect, useState } from "react";
import { StrategyBuilder } from "./StrategyBuilder";

interface Leg {
  contract_id: string;
  close: number;
  settlement: number;
  open_interest: number;
  oi_change: number;
  volume: number;
  trades: number;
  implied_vol: number | null;
  delta: number | null;
  gamma: number | null;
  vega: number | null;
  theta: number | null;
}

interface Row {
  strike: number;
  call: Leg | null;
  put: Leg | null;
}

interface Chain {
  underlying: string;
  session: string;
  expiry: string;
  days_to_expiry: number;
  underlying_price: number;
  lot_size: number;
  rate: number;
  rows: Row[];
  atm_strike: number | null;
}

interface Expiry {
  expiry: string;
  days: number;
  contracts: number;
  open_interest: number;
}

function oi(value: number): string {
  if (value >= 1e7) return `${(value / 1e7).toFixed(2)}Cr`;
  if (value >= 1e5) return `${(value / 1e5).toFixed(1)}L`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
  return value.toFixed(0);
}

function show(value: number | null, digits: number, suffix = ""): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${suffix}`;
}

function iv(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export interface OptionChainProps {
  underlying: string;
  /** Send a contract to the order ticket. */
  onTrade?: (terms: { expiry: string; strike: string; right: "CE" | "PE" }) => void;
}

export function OptionChain({ underlying, onTrade }: OptionChainProps) {
  const [expiries, setExpiries] = useState<Expiry[]>([]);
  const [chosen, setChosen] = useState<string>("");
  const [chain, setChain] = useState<Chain | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!underlying) return;
    let cancelled = false;
    setError("");
    setChain(null);
    fetch(`/api/options/${encodeURIComponent(underlying)}/expiries`)
      .then(async (r) => {
        if (!r.ok) {
          const body = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(body?.detail ?? `HTTP ${r.status}`);
        }
        return (await r.json()) as Expiry[];
      })
      .then((rows) => {
        if (cancelled) return;
        setExpiries(rows);
        setChosen(rows[0]?.expiry ?? "");
      })
      .catch((exc: Error) => {
        if (cancelled) return;
        setExpiries([]);
        setChosen("");
        setError(exc.message);
      });
    return () => {
      cancelled = true;
    };
  }, [underlying]);

  const load = useCallback(() => {
    if (!underlying || !chosen) return;
    let cancelled = false;
    setLoading(true);
    fetch(`/api/options/${encodeURIComponent(underlying)}/chain?expiry=${chosen}`)
      .then(async (r) => {
        if (!r.ok) {
          const body = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(body?.detail ?? `HTTP ${r.status}`);
        }
        return (await r.json()) as Chain;
      })
      .then((payload) => !cancelled && setChain(payload))
      .catch((exc: Error) => {
        if (cancelled) return;
        setChain(null);
        setError(exc.message);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [underlying, chosen]);

  useEffect(() => {
    load();
  }, [load]);

  if (error) {
    return (
      <p className="empty">
        {error.includes("no listed contracts") ? `No listed options for ${underlying}.` : error}
      </p>
    );
  }

  return (
    <div className="chain">
      <div className="chain-bar">
        <div className="segmented small">
          {expiries.slice(0, 6).map((e) => (
            <button
              key={e.expiry}
              type="button"
              className={chosen === e.expiry ? "on" : ""}
              onClick={() => setChosen(e.expiry)}
              title={`${e.contracts} contracts, ${oi(e.open_interest)} open interest`}
            >
              {e.expiry.slice(5)} · {e.days}d
            </button>
          ))}
        </div>
        {chain && (
          <span className="muted">
            spot {chain.underlying_price.toFixed(2)} · lot {chain.lot_size.toFixed(0)} · r{" "}
            {(chain.rate * 100).toFixed(2)}% · {chain.session}
          </span>
        )}
        {loading && <span className="muted">loading…</span>}
      </div>

      {chain && (
        <div className="chain-scroll">
          <table className="grid chain-table">
            <thead>
              <tr>
                <th colSpan={5} className="chain-side call">
                  Calls
                </th>
                <th className="chain-strike-head">Strike</th>
                <th colSpan={5} className="chain-side put">
                  Puts
                </th>
              </tr>
              <tr>
                <th className="num">OI</th>
                <th className="num">Vol</th>
                <th className="num">θ</th>
                <th className="num">Δ</th>
                <th className="num">IV</th>
                <th className="num">LTP</th>
                <th className="num">LTP</th>
                <th className="num">IV</th>
                <th className="num">Δ</th>
                <th className="num">θ</th>
                <th className="num">OI</th>
              </tr>
            </thead>
            <tbody>
              {chain.rows.map((row) => {
                const itmCall = row.strike < chain.underlying_price;
                const atm = chain.atm_strike === row.strike;
                return (
                  <tr key={row.strike} className={atm ? "atm" : ""}>
                    <td className={`num mono ${itmCall ? "itm" : ""}`}>
                      {row.call ? oi(row.call.open_interest) : "—"}
                    </td>
                    <td className={`num mono ${itmCall ? "itm" : ""}`}>
                      {row.call ? oi(row.call.volume) : "—"}
                    </td>
                    <td className={`num mono ${itmCall ? "itm" : ""}`}>
                      {show(row.call?.theta ?? null, 2)}
                    </td>
                    <td className={`num mono ${itmCall ? "itm" : ""}`}>
                      {show(row.call?.delta ?? null, 3)}
                    </td>
                    <td className={`num mono ${itmCall ? "itm" : ""}`}>
                      {iv(row.call?.implied_vol ?? null)}
                    </td>
                    <td className={`num mono strong ${itmCall ? "itm" : ""}`}>
                      {row.call && onTrade ? (
                        <button
                          type="button"
                          className="chain-buy"
                          title={`Trade the ${row.strike} call`}
                          onClick={() =>
                            onTrade({ expiry: chosen, strike: String(row.strike), right: "CE" })
                          }
                        >
                          {show(row.call.close, 2)}
                        </button>
                      ) : (
                        show(row.call?.close ?? null, 2)
                      )}
                    </td>

                    <td className="num mono chain-strike">{row.strike.toFixed(1)}</td>

                    <td className={`num mono strong ${!itmCall ? "itm" : ""}`}>
                      {row.put && onTrade ? (
                        <button
                          type="button"
                          className="chain-buy"
                          title={`Trade the ${row.strike} put`}
                          onClick={() =>
                            onTrade({ expiry: chosen, strike: String(row.strike), right: "PE" })
                          }
                        >
                          {show(row.put.close, 2)}
                        </button>
                      ) : (
                        show(row.put?.close ?? null, 2)
                      )}
                    </td>
                    <td className={`num mono ${!itmCall ? "itm" : ""}`}>
                      {iv(row.put?.implied_vol ?? null)}
                    </td>
                    <td className={`num mono ${!itmCall ? "itm" : ""}`}>
                      {show(row.put?.delta ?? null, 3)}
                    </td>
                    <td className={`num mono ${!itmCall ? "itm" : ""}`}>
                      {show(row.put?.theta ?? null, 2)}
                    </td>
                    <td className={`num mono ${!itmCall ? "itm" : ""}`}>
                      {row.put ? oi(row.put.open_interest) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="block-note">
            Implied volatility and Greeks derived from closing prices (Black-Scholes, European, r ={" "}
            {(chain.rate * 100).toFixed(2)}%). Blank values indicate a strike that did not trade.
          </p>

          {/* The chain above shows one contract per row; nobody trades one
              contract. The builder is here rather than on a tab of its own
              because it needs this expiry's prices and volatilities, and a
              second fetch of the same chain could disagree with what is on
              screen. */}
          <details className="builder-fold" open>
            <summary>Strategy builder</summary>
            <StrategyBuilder
              underlying={chain.underlying}
              spot={chain.underlying_price}
              lotSize={chain.lot_size}
              daysToExpiry={chain.days_to_expiry}
              rows={chain.rows}
            />
          </details>
        </div>
      )}
    </div>
  );
}
