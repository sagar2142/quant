/**
 * Order ticket — MASTER_PLAN §8, §12.7, §21.
 *
 * **Preview is the default action, and submit is not.** The interesting part of
 * placing an order is what it costs and which limit it comes near, and that is
 * arithmetic the server will do without any credentials at all. Sending is a
 * separate, deliberate second step.
 *
 * **Blocked reasons are shown, never hidden behind a disabled button.** If the
 * order cannot go, the panel says which gate is shut and what to do about it —
 * "KITE_ACCESS_TOKEN is empty" is actionable, a greyed-out button is not.
 *
 * **Nothing here simulates a fill.** There is no local order book, no pretend
 * confirmation. If the chain is incomplete the ticket reports that, because a
 * ticket that appears to work and does nothing is the worst possible object in
 * a trading system.
 */

import { useCallback, useEffect, useState } from "react";

const EQUITY_KEY = "neutron.ticket.equity";

interface Gate {
  name: string;
  ready: boolean;
  detail: string;
}

interface Status {
  can_trade: boolean;
  mode: string;
  gates: Gate[];
}

interface Check {
  name: string;
  passed: boolean;
  detail: string;
}

interface Quote {
  symbol: string;
  instrument_id: string;
  last_close: number;
  adv: number;
  as_of: string;
}

interface Preview {
  symbol: string;
  instrument_id: string;
  last_close: string;
  side: string;
  quantity: string;
  notional: string;
  estimated_costs: string;
  cash_impact: string;
  allowed: boolean;
  breaches: string[];
  checks: Check[];
}

export interface TicketProps {
  /** Pre-filled from whichever chart sent you here. */
  symbol: string;
  onSymbolChange: (symbol: string) => void;
}

export function Ticket({ symbol, onSymbolChange }: TicketProps) {
  const [status, setStatus] = useState<Status | null>(null);
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [quantity, setQuantity] = useState("100");
  const [orderType, setOrderType] = useState<"MARKET" | "LIMIT">("MARKET");
  const [limitPrice, setLimitPrice] = useState("");
  const [equity, setEquity] = useState(() => {
    try {
      return localStorage.getItem(EQUITY_KEY) ?? "1000000";
    } catch {
      return "1000000";
    }
  });
  const [quote, setQuote] = useState<Quote | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  //: Typed confirmation before anything is sent. A real-money action behind a
  //: single click is one mis-aimed pointer away from a position you did not
  //: choose.
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    fetch("/api/trade/status")
      .then((r) => (r.ok ? r.json() : null))
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  //: The last close, fetched whenever the symbol changes. A market order has
  //: no price of its own, but the risk engine still measures one against the
  //: last trade — an order it cannot price is one it cannot establish as sane,
  //: and it blocks rather than passes.
  useEffect(() => {
    if (!symbol) return;
    let cancelled = false;
    setQuote(null);
    setPreview(null);
    fetch(`/api/quote/${encodeURIComponent(symbol)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((payload) => !cancelled && setQuote(payload))
      .catch(() => !cancelled && setQuote(null));
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  useEffect(() => {
    try {
      localStorage.setItem(EQUITY_KEY, equity);
    } catch {
      // Convenience only.
    }
  }, [equity]);

  const body = useCallback(
    (reference: string) => ({
      symbol,
      side,
      quantity,
      order_type: orderType,
      limit_price: orderType === "LIMIT" ? limitPrice : null,
      reference_price: reference,
      equity,
    }),
    [symbol, side, quantity, orderType, limitPrice, equity],
  );

  const runPreview = async () => {
    setBusy(true);
    setMessage("");
    setConfirming(false);
    // The reference is the limit price when one is given, and otherwise the
    // last close. The server re-reads that close itself and measures the order
    // against it, so a screen left open overnight is caught by the price band
    // rather than trusted.
    const reference =
      orderType === "LIMIT" && limitPrice ? limitPrice : quote ? String(quote.last_close) : "";
    if (!reference) {
      setBusy(false);
      setMessage(`no price for ${symbol} — it is not in the panel`);
      return;
    }
    try {
      const response = await fetch("/api/trade/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body(reference)),
      });
      const payload = await response.json();
      if (!response.ok) {
        setPreview(null);
        setMessage(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail));
        return;
      }
      setPreview(payload as Preview);
    } catch (exc) {
      setPreview(null);
      setMessage(String(exc));
    } finally {
      setBusy(false);
    }
  };

  const send = async () => {
    if (!preview) return;
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch("/api/trade/orders", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body(preview.last_close)),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail;
        if (detail && typeof detail === "object" && Array.isArray(detail.blocked_by)) {
          setMessage(
            `${detail.error}: ` +
              detail.blocked_by.map((b: { gate: string; detail: string }) => `${b.gate} — ${b.detail}`).join(" · "),
          );
        } else {
          setMessage(typeof detail === "string" ? detail : JSON.stringify(payload));
        }
        return;
      }
      setMessage(`sent — broker order ${payload.broker_order_id}`);
      setConfirming(false);
    } catch (exc) {
      setMessage(String(exc));
    } finally {
      setBusy(false);
    }
  };

  const blocked = status?.gates.filter((g) => !g.ready) ?? [];

  return (
    <section className="ticket" aria-label="Order ticket">
      <div className="ticket-form">
        <header className="panel-header">Order</header>

        <label>
          <span>Symbol</span>
          <input
            value={symbol}
            onChange={(event) => onSymbolChange(event.target.value.toUpperCase())}
            spellCheck={false}
          />
        </label>

        <div className="side-toggle">
          <button
            type="button"
            className={side === "BUY" ? "buy on" : "buy"}
            onClick={() => setSide("BUY")}
          >
            BUY
          </button>
          <button
            type="button"
            className={side === "SELL" ? "sell on" : "sell"}
            onClick={() => setSide("SELL")}
          >
            SELL
          </button>
        </div>

        <div className="ticket-quote">
          {quote ? (
            <>
              <span className="mono big">{quote.last_close.toFixed(2)}</span>
              <span className="muted">last close · {quote.as_of}</span>
            </>
          ) : (
            <span className="muted">no price for {symbol}</span>
          )}
        </div>

        <label>
          <span>Quantity</span>
          <input value={quantity} onChange={(event) => setQuantity(event.target.value)} inputMode="numeric" />
        </label>

        <label>
          <span>Type</span>
          <select value={orderType} onChange={(event) => setOrderType(event.target.value as "MARKET" | "LIMIT")}>
            <option value="MARKET">Market</option>
            <option value="LIMIT">Limit</option>
          </select>
        </label>

        {orderType === "LIMIT" && (
          <label>
            <span>Limit price</span>
            <input value={limitPrice} onChange={(event) => setLimitPrice(event.target.value)} inputMode="decimal" />
          </label>
        )}

        <label>
          <span>Capital</span>
          <input value={equity} onChange={(event) => setEquity(event.target.value)} inputMode="numeric" />
          <small>
            Every percentage limit is a fraction of this. There is no book to read it from — paper
            trading was removed on purpose.
          </small>
        </label>

        {quote && Number(quantity) > 0 && (
          <p className="ticket-value">
            ≈ {(Number(quantity) * quote.last_close).toLocaleString("en-IN", {
              maximumFractionDigits: 0,
            })}{" "}
            <span className="muted">
              ({((Number(quantity) * quote.last_close * 100) / Number(equity || 1)).toFixed(1)}% of
              capital)
            </span>
          </p>
        )}

        <button type="button" className="primary" onClick={runPreview} disabled={busy}>
          {busy ? "…" : "Preview"}
        </button>
      </div>

      <div className="ticket-result">
        <header className="panel-header">
          Risk check
          {status && (
            <span className={status.can_trade ? "badge live" : "badge blocked"}>{status.mode}</span>
          )}
        </header>

        {preview ? (
          <>
            <dl className="ticket-figures">
              <div>
                <dt>Instrument</dt>
                <dd className="mono">{preview.instrument_id}</dd>
              </div>
              <div>
                <dt>Last close</dt>
                <dd className="mono">{preview.last_close}</dd>
              </div>
              <div>
                <dt>Notional</dt>
                <dd className="mono">{preview.notional}</dd>
              </div>
              <div>
                <dt>Est. costs</dt>
                <dd className="mono">{preview.estimated_costs}</dd>
              </div>
              <div>
                <dt>Cash impact</dt>
                <dd className="mono">{preview.cash_impact}</dd>
              </div>
            </dl>

            <ul className="check-list">
              {preview.checks.map((check) => (
                <li key={check.name} className={check.passed ? "pass" : "fail"}>
                  <span className="check-mark">{check.passed ? "✓" : "✗"}</span>
                  <span className="check-name">{check.name}</span>
                  <span className="check-detail">{check.detail}</span>
                </li>
              ))}
            </ul>

            {preview.allowed ? (
              confirming ? (
                <div className="confirm">
                  <p>
                    Send <b>{preview.side}</b> {preview.quantity} {preview.symbol} to the exchange?
                  </p>
                  <button type="button" className="danger" onClick={send} disabled={busy}>
                    Confirm
                  </button>
                  <button type="button" className="ghost" onClick={() => setConfirming(false)}>
                    Cancel
                  </button>
                </div>
              ) : (
                <button type="button" className="primary" onClick={() => setConfirming(true)}>
                  Send order
                </button>
              )
            ) : (
              <p className="rejected">Risk engine rejected this order. It cannot be sent.</p>
            )}
          </>
        ) : (
          <p className="empty">Enter an order and press Preview.</p>
        )}

        {message && <p className="ticket-message">{message}</p>}

        {blocked.length > 0 && (
          <div className="gates">
            <h4>Not armed for live trading</h4>
            <ul>
              {blocked.map((gate) => (
                <li key={gate.name}>
                  <b>{gate.name}</b> — {gate.detail}
                </li>
              ))}
            </ul>
            <p className="gates-note">
              Preview works regardless. Sending requires all four.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
