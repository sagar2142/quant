/**
 * Order book — working orders, positions and fills, read from the venue.
 *
 * Named for what it shows rather than for the screen it sits on: the shell
 * already has a `Blotter`, which lists trades this system recorded. This lists
 * what the *exchange* is holding, and the two are only equal when nothing has
 * gone wrong.
 *
 * **An empty table and "I could not ask" are different answers.** The API
 * refuses these reads when the broker is not armed rather than returning `[]`,
 * and this panel shows that refusal as a refusal. A screen that renders "no
 * open orders" because it never managed to call the exchange is telling you
 * the one thing you must not believe.
 *
 * **The venue is the authority.** None of this is a local record of what was
 * sent: between submission and now sit rejections, partial fills and
 * cancellations nobody here initiated, and only the broker knows about them.
 */

import { useCallback, useEffect, useState } from "react";

interface WorkingOrder {
  broker_order_id: string;
  instrument_id: string;
  side: string;
  quantity: string;
  filled_quantity: string;
  pending_quantity: string;
  price: string | null;
  status: string;
  placed_at: string;
}

interface HeldPosition {
  instrument_id: string;
  quantity: string;
  average_price: string;
}

interface ExecutedFill {
  broker_fill_id: string;
  broker_order_id: string;
  instrument_id: string;
  side: string;
  quantity: string;
  price: string;
}

interface Blocked {
  gate: string;
  detail: string;
}

interface Loaded<T> {
  rows: T[];
  blocked: Blocked[];
  error: string;
}

const EMPTY = { rows: [], blocked: [], error: "" };

async function read<T>(path: string): Promise<Loaded<T>> {
  try {
    const response = await fetch(path);
    const payload = await response.json();
    if (response.ok) return { rows: payload as T[], blocked: [], error: "" };

    const detail = payload.detail;
    if (detail && typeof detail === "object" && Array.isArray(detail.blocked_by)) {
      return { rows: [], blocked: detail.blocked_by as Blocked[], error: "" };
    }
    return { rows: [], blocked: [], error: typeof detail === "string" ? detail : `HTTP ${response.status}` };
  } catch (exc) {
    return { rows: [], blocked: [], error: String(exc) };
  }
}

type Tab = "orders" | "positions" | "fills";

export function OrderBook() {
  const [tab, setTab] = useState<Tab>("orders");
  const [orders, setOrders] = useState<Loaded<WorkingOrder>>(EMPTY);
  const [positions, setPositions] = useState<Loaded<HeldPosition>>(EMPTY);
  const [fills, setFills] = useState<Loaded<ExecutedFill>>(EMPTY);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");

  const refresh = useCallback(async () => {
    setBusy(true);
    const [o, p, f] = await Promise.all([
      read<WorkingOrder>("/api/trade/orders"),
      read<HeldPosition>("/api/trade/positions"),
      read<ExecutedFill>("/api/trade/fills"),
    ]);
    setOrders(o);
    setPositions(p);
    setFills(f);
    setBusy(false);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const cancel = async (id: string) => {
    setNote("");
    const response = await fetch(`/api/trade/orders/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    const payload = await response.json().catch(() => null);
    setNote(
      response.ok
        ? `cancelled ${id}`
        : `could not cancel ${id}: ${JSON.stringify(payload?.detail ?? response.status)}`,
    );
    void refresh();
  };

  const current: Loaded<unknown> = tab === "orders" ? orders : tab === "positions" ? positions : fills;

  return (
    <section className="blotter" aria-label="Order book">
      <header className="charts-bar">
        <div className="segmented">
          {(["orders", "positions", "fills"] as Tab[]).map((id) => (
            <button key={id} type="button" className={tab === id ? "on" : ""} onClick={() => setTab(id)}>
              {id}
            </button>
          ))}
        </div>
        <button type="button" className="ghost" onClick={() => void refresh()} disabled={busy}>
          {busy ? "…" : "Refresh"}
        </button>
        <span className="charts-hint">Read from the broker</span>
      </header>

      <div className="blotter-body">
        {current.blocked.length > 0 && (
          <div className="gates">
            <h4>Broker unavailable</h4>
            <ul>
              {current.blocked.map((b) => (
                <li key={b.gate}>
                  <b>{b.gate}</b> — {b.detail}
                </li>
              ))}
            </ul>
            <p className="gates-note">
              No request was made to the exchange. This is not an empty book.
            </p>
          </div>
        )}

        {current.error && <p className="ticket-message">{current.error}</p>}

        {current.blocked.length === 0 && !current.error && (
          <>
            {tab === "orders" && (
              <table className="grid">
                <thead>
                  <tr>
                    <th>Order</th>
                    <th>Instrument</th>
                    <th>Side</th>
                    <th className="num">Qty</th>
                    <th className="num">Filled</th>
                    <th className="num">Price</th>
                    <th>Status</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {orders.rows.map((row) => (
                    <tr key={row.broker_order_id}>
                      <td className="mono">{row.broker_order_id}</td>
                      <td className="mono">{row.instrument_id}</td>
                      <td className={row.side === "BUY" ? "up" : "down"}>{row.side}</td>
                      <td className="num mono">{row.quantity}</td>
                      <td className="num mono">{row.filled_quantity}</td>
                      <td className="num mono">{row.price ?? "MKT"}</td>
                      <td>{row.status}</td>
                      <td>
                        <button type="button" className="ghost danger" onClick={() => void cancel(row.broker_order_id)}>
                          Cancel
                        </button>
                      </td>
                    </tr>
                  ))}
                  {orders.rows.length === 0 && (
                    <tr>
                      <td colSpan={8} className="empty">
                        No orders today
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}

            {tab === "positions" && (
              <table className="grid">
                <thead>
                  <tr>
                    <th>Instrument</th>
                    <th className="num">Quantity</th>
                    <th className="num">Average price</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.rows.map((row) => (
                    <tr key={row.instrument_id}>
                      <td className="mono">{row.instrument_id}</td>
                      <td className="num mono">{row.quantity}</td>
                      <td className="num mono">{row.average_price}</td>
                    </tr>
                  ))}
                  {positions.rows.length === 0 && (
                    <tr>
                      <td colSpan={3} className="empty">
                        No open positions
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}

            {tab === "fills" && (
              <table className="grid">
                <thead>
                  <tr>
                    <th>Fill</th>
                    <th>Order</th>
                    <th>Instrument</th>
                    <th>Side</th>
                    <th className="num">Qty</th>
                    <th className="num">Price</th>
                  </tr>
                </thead>
                <tbody>
                  {fills.rows.map((row) => (
                    <tr key={row.broker_fill_id}>
                      <td className="mono">{row.broker_fill_id}</td>
                      <td className="mono">{row.broker_order_id}</td>
                      <td className="mono">{row.instrument_id}</td>
                      <td className={row.side === "BUY" ? "up" : "down"}>{row.side}</td>
                      <td className="num mono">{row.quantity}</td>
                      <td className="num mono">{row.price}</td>
                    </tr>
                  ))}
                  {fills.rows.length === 0 && (
                    <tr>
                      <td colSpan={6} className="empty">
                        No fills today
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            )}
          </>
        )}

        {note && <p className="ticket-message">{note}</p>}
      </div>
    </section>
  );
}
