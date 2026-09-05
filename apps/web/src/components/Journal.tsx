/**
 * Trade journal — MASTER_PLAN §9, §M7.
 *
 * **The cost model had never been checked against anything.** A backtest
 * charges an estimate at every fill, and the only test that estimate ever
 * faced was whether the equity curve looked believable — which it always will,
 * because the model produced it. This is the other number.
 *
 * **The decomposition is the point, not the total.** Delay cost is a process
 * problem, execution cost is a broker or order-type problem, and fees are the
 * schedule. One "slippage" figure tells you something is wrong and nothing
 * about which of the three, so they are shown apart and only then summed.
 *
 * **An em dash is not a zero.** An order with no reference price has an
 * unknown cost, and rendering that as ₹0.00 would average into the totals and
 * quietly flatter them. Coverage says how much of the book could be measured
 * at all.
 */

import { useCallback, useEffect, useState } from "react";
import { formatCount, formatMoney } from "../format";

interface Row {
  order_id: string;
  symbol: string;
  side: string;
  strategy_id: string;
  mode: string;
  decision_time: string;
  quantity_ordered: string;
  quantity_filled: string;
  fill_rate: number;
  average_fill: string | null;
  intended_price: string | null;
  fees: string;
  delay_cost: string | null;
  execution_cost: string | null;
  slippage: string | null;
  slippage_bps: number | null;
  implementation_shortfall: string | null;
}

interface Journal {
  orders: number;
  filled: number;
  measured: number;
  coverage: number;
  notional: string;
  fees: string;
  delay_cost: string | null;
  execution_cost: string | null;
  slippage: string | null;
  implementation_shortfall: string | null;
  shortfall_bps: number | null;
  rows: Row[];
  worst: string[];
  note: string;
}

/** Money, or an em dash for a cost that was never measurable. */
function cost(value: string | null): string {
  return value === null ? "—" : formatMoney(Number(value));
}

function costClass(value: string | null): string {
  if (value === null) return "text-secondary";
  return Number(value) > 0 ? "text-loss" : "text-profit";
}

export function Journal({ apiBase = "/api" }: { apiBase?: string }) {
  const [journal, setJournal] = useState<Journal | null>(null);
  const [mode, setMode] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    const query = mode ? `?mode=${mode}` : "";
    const response = await fetch(`${apiBase}/journal${query}`);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
      setError(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      return;
    }
    setJournal((await response.json()) as Journal);
  }, [apiBase, mode]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="analytics">
      <div className="analytics-bar">
        <select
          className="analytics-select"
          value={mode}
          onChange={(event) => setMode(event.target.value)}
          aria-label="mode"
        >
          <option value="">all orders</option>
          <option value="PAPER">paper</option>
          <option value="LIVE">live</option>
        </select>
        <span className="analytics-spacer" />
        <button type="button" className="analytics-run" onClick={() => void load()}>
          Refresh
        </button>
      </div>

      <div className="analytics-body">
        {error ? <div className="analytics-note text-critical">{error}</div> : null}
        {journal?.note ? <div className="analytics-note">{journal.note}</div> : null}

        {journal && journal.orders > 0 ? (
          <>
            <div className="stat-row">
              <div className="stat">
                <div className="stat-label">delay cost</div>
                <div className={`stat-value ${costClass(journal.delay_cost)}`}>
                  {cost(journal.delay_cost)}
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">execution cost</div>
                <div className={`stat-value ${costClass(journal.execution_cost)}`}>
                  {cost(journal.execution_cost)}
                </div>
              </div>
              <div className="stat">
                <div className="stat-label">fees</div>
                <div className="stat-value text-loss">{formatMoney(Number(journal.fees))}</div>
              </div>
              <div className="stat">
                <div className="stat-label">shortfall</div>
                <div className={`stat-value ${costClass(journal.implementation_shortfall)}`}>
                  {cost(journal.implementation_shortfall)}
                  {journal.shortfall_bps === null
                    ? ""
                    : ` · ${journal.shortfall_bps.toFixed(1)}bps`}
                </div>
              </div>
            </div>

            {/* A total computed from a third of the book must not be mistaken
                for the whole book's. */}
            {journal.measured < journal.filled ? (
              <div className="analytics-note">
                {journal.filled - journal.measured} of {journal.filled} filled orders had no
                reference price. Their cost is unmeasured, not zero — coverage is{" "}
                {(journal.coverage * 100).toFixed(0)}%.
              </div>
            ) : null}

            <table className="grid">
              <thead>
                <tr>
                  <th>order</th>
                  <th>side</th>
                  <th className="num">filled</th>
                  <th className="num">avg fill</th>
                  <th className="num">delay</th>
                  <th className="num">execution</th>
                  <th className="num">fees</th>
                  <th className="num">shortfall</th>
                  <th className="num">bps</th>
                </tr>
              </thead>
              <tbody>
                {journal.rows.map((row) => (
                  <tr
                    key={row.order_id}
                    className={journal.worst.includes(row.order_id) ? "journal-worst" : ""}
                  >
                    <td>{row.symbol}</td>
                    <td className={row.side === "BUY" ? "text-profit" : "text-loss"}>
                      {row.side}
                    </td>
                    <td className="num mono">
                      {formatCount(Number(row.quantity_filled))} /{" "}
                      {formatCount(Number(row.quantity_ordered))}
                    </td>
                    <td className="num mono">
                      {row.average_fill === null ? "—" : Number(row.average_fill).toFixed(2)}
                    </td>
                    <td className={`num mono ${costClass(row.delay_cost)}`}>
                      {cost(row.delay_cost)}
                    </td>
                    <td className={`num mono ${costClass(row.execution_cost)}`}>
                      {cost(row.execution_cost)}
                    </td>
                    <td className="num mono">{formatMoney(Number(row.fees))}</td>
                    <td className={`num mono ${costClass(row.implementation_shortfall)}`}>
                      {cost(row.implementation_shortfall)}
                    </td>
                    <td className="num mono">
                      {row.slippage_bps === null ? "—" : row.slippage_bps.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <p className="block-note">
              Shortfall is measured against the decision price where one was recorded, and against
              the order's own intended price otherwise. Positive is a cost on both sides: paying
              above the reference on a buy and receiving below it on a sell are the same loss.
            </p>
          </>
        ) : null}
      </div>
    </div>
  );
}
