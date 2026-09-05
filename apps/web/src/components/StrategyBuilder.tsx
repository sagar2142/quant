/**
 * Multi-leg option structures — MASTER_PLAN §M6.
 *
 * **The chain shows one contract per row and nobody trades one contract.** A
 * spread's risk is not two rows read separately; the offset between the legs
 * is the entire reason the structure exists, and a strike-by-strike table is
 * exactly the view that cannot show it.
 *
 * **The unbounded case is drawn as unbounded.** A short call's loss has no
 * floor. Printing the worst value of whatever price range happened to be
 * plotted would give a finite, confident, wrong number on the one position
 * where the number matters most — so it reads "unbounded", in red, and the
 * chart says so too.
 *
 * **Two curves, not one.** The hockey-stick is what the position pays at
 * expiry; the smooth line is what it is worth now. A position can be well
 * under water on the first and comfortably ahead on the second, and a builder
 * that showed only the first would misprice every day before the last one.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { formatMoney } from "../format";

interface StructureRow {
  name: string;
  label: string;
  strikes: number;
  legs: { kind: string; quantity: number; strike_index: number }[];
}

interface ChainLeg {
  close: number;
  implied_vol: number | null;
}

export interface ChainStrike {
  strike: number;
  call: ChainLeg | null;
  put: ChainLeg | null;
}

interface Point {
  price: number;
  profit: number;
}

interface Analysis {
  underlying: string;
  spot: number;
  days_to_expiry: number;
  net_premium: number;
  payoff: Point[];
  value: Point[];
  breakevens: number[];
  max_profit: number | null;
  max_loss: number | null;
  delta: number | null;
  gamma: number | null;
  vega: number | null;
  theta: number | null;
  rho: number | null;
  note: string;
}

export interface StrategyBuilderProps {
  underlying: string;
  spot: number;
  lotSize: number;
  daysToExpiry: number;
  rows: ChainStrike[];
}

/** The strike nearest a target, from those actually listed. */
function nearest(rows: ChainStrike[], target: number): number {
  return rows.reduce(
    (best, row) => (Math.abs(row.strike - target) < Math.abs(best - target) ? row.strike : best),
    rows[0]?.strike ?? target,
  );
}

function money(value: number | null): string {
  return value === null ? "unbounded" : formatMoney(value);
}

export function StrategyBuilder({
  underlying,
  spot,
  lotSize,
  daysToExpiry,
  rows,
}: StrategyBuilderProps) {
  const [structures, setStructures] = useState<StructureRow[]>([]);
  const [chosen, setChosen] = useState("bull_call_spread");
  const [width, setWidth] = useState(1);
  const [lots, setLots] = useState(1);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    void (async () => {
      const response = await fetch("/api/options/structures");
      if (response.ok) setStructures((await response.json()) as StructureRow[]);
    })();
  }, []);

  const structure = structures.find((s) => s.name === chosen);

  //: Strikes are picked outward from the money in steps of `width` listed
  //: strikes, so the same structure works on a name with ₹5 spacing and one
  //: with ₹100 — spacing is a property of the contract, not of the trade.
  const strikes = useMemo(() => {
    if (!structure || rows.length === 0) return [];
    const ladder = [...rows].sort((a, b) => a.strike - b.strike);
    const atmIndex = ladder.findIndex((r) => r.strike === nearest(ladder, spot));
    const count = structure.strikes;
    // Centred on the money: a four-strike condor straddles it, a one-strike
    // structure sits on it.
    let start = atmIndex - Math.floor(((count - 1) * width) / 2);
    // Slide the whole window inside the ladder rather than clamping each leg
    // independently. Clamping produced *duplicate* strikes near the edges — a
    // butterfly whose three legs landed on one strike, which prices and plots
    // perfectly well while being a position nobody asked for and nobody could
    // trade.
    const span = (count - 1) * width;
    start = Math.max(0, Math.min(start, ladder.length - 1 - span));

    const picked: number[] = [];
    for (let i = 0; i < count; i += 1) {
      const entry = ladder[start + i * width];
      if (entry) picked.push(entry.strike);
    }
    // If the ladder cannot supply distinct strikes at this width, say so
    // rather than building a degenerate structure.
    return new Set(picked).size === count ? picked : [];
  }, [structure, rows, spot, width]);

  const legs = useMemo(() => {
    if (!structure) return [];
    const ordered = [...strikes].sort((a, b) => a - b);
    return structure.legs.map((leg) => {
      const strike = ordered[leg.strike_index] ?? spot;
      const row = rows.find((r) => r.strike === strike);
      const side = leg.kind === "CE" ? row?.call : leg.kind === "PE" ? row?.put : null;
      return {
        kind: leg.kind,
        quantity: leg.quantity * lots,
        strike: leg.kind === "EQ" ? 0 : strike,
        price: leg.kind === "EQ" ? spot : (side?.close ?? 0),
        implied_vol: leg.kind === "EQ" ? null : (side?.implied_vol ?? null),
      };
    });
  }, [structure, strikes, rows, spot, lots]);

  const run = useCallback(async () => {
    if (legs.length === 0) return;
    setError("");
    const response = await fetch("/api/options/strategy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        underlying,
        spot,
        lot_size: lotSize,
        days_to_expiry: daysToExpiry,
        legs,
      }),
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
      setError(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      setAnalysis(null);
      return;
    }
    setAnalysis((await response.json()) as Analysis);
  }, [legs, underlying, spot, lotSize, daysToExpiry]);

  useEffect(() => {
    void run();
  }, [run]);

  const chart = useMemo(() => {
    if (!analysis || analysis.payoff.length < 2) return null;
    const width_ = 600;
    const height = 160;
    const series = [...analysis.payoff, ...analysis.value];
    const xs = analysis.payoff.map((p) => p.price);
    const ys = series.map((p) => p.profit);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);
    const y0 = Math.min(...ys);
    const y1 = Math.max(...ys);
    const xSpan = x1 - x0 || 1;
    const ySpan = y1 - y0 || 1;
    const project = (p: Point, index: number) =>
      `${index === 0 ? "M" : "L"}${(((p.price - x0) / xSpan) * width_).toFixed(1)} ` +
      `${(height - ((p.profit - y0) / ySpan) * height).toFixed(1)}`;
    return {
      width: width_,
      height,
      payoff: analysis.payoff.map(project).join(" "),
      value: analysis.value.map(project).join(" "),
      // Where zero profit sits, so the eye can separate gain from loss
      // without reading the axis.
      zero: height - ((0 - y0) / ySpan) * height,
      spotX: ((spot - x0) / xSpan) * width_,
    };
  }, [analysis, spot]);

  if (rows.length === 0) {
    return <div className="text-secondary">No chain loaded — pick an expiry first.</div>;
  }

  return (
    <div className="builder">
      <div className="builder-bar">
        <select
          className="analytics-select"
          value={chosen}
          onChange={(event) => setChosen(event.target.value)}
          aria-label="structure"
        >
          {structures.map((s) => (
            <option key={s.name} value={s.name}>
              {s.label}
            </option>
          ))}
        </select>

        <select
          className="analytics-select"
          value={width}
          onChange={(event) => setWidth(Number(event.target.value))}
          aria-label="strike spacing"
          title="Distance between legs, in listed strikes"
        >
          {[1, 2, 3, 4, 6].map((n) => (
            <option key={n} value={n}>
              {n} strike{n > 1 ? "s" : ""} apart
            </option>
          ))}
        </select>

        <select
          className="analytics-select"
          value={lots}
          onChange={(event) => setLots(Number(event.target.value))}
          aria-label="lots"
        >
          {[1, 2, 5, 10].map((n) => (
            <option key={n} value={n}>
              {n} lot{n > 1 ? "s" : ""}
            </option>
          ))}
        </select>

        <span className="analytics-spacer" />
        <span className="text-secondary">
          {daysToExpiry}d · lot {lotSize} · spot {spot.toFixed(2)}
        </span>
      </div>

      {error ? <div className="analytics-note text-critical">{error}</div> : null}

      {structure && strikes.length === 0 ? (
        <div className="analytics-note">
          This expiry does not list {structure.strikes} distinct strikes {width} apart. Narrow the
          spacing, or pick a structure with fewer legs.
        </div>
      ) : null}

      <table className="grid builder-legs">
        <thead>
          <tr>
            <th>leg</th>
            <th className="num">strike</th>
            <th className="num">lots</th>
            <th className="num">price</th>
            <th className="num">IV</th>
          </tr>
        </thead>
        <tbody>
          {legs.map((leg, index) => (
            <tr key={`${leg.kind}-${leg.strike}-${index}`}>
              <td className={leg.quantity > 0 ? "text-profit" : "text-loss"}>
                {leg.quantity > 0 ? "long" : "short"} {leg.kind === "EQ" ? "underlying" : leg.kind}
              </td>
              <td className="num">{leg.kind === "EQ" ? "—" : leg.strike.toFixed(2)}</td>
              <td className="num">{Math.abs(leg.quantity)}</td>
              <td className="num">{leg.price.toFixed(2)}</td>
              <td className="num">
                {leg.implied_vol === null ? "—" : `${(leg.implied_vol * 100).toFixed(1)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {analysis ? (
        <>
          <div className="stat-row">
            <div className="stat">
              <div className="stat-label">{analysis.net_premium >= 0 ? "debit" : "credit"}</div>
              <div className="stat-value">{formatMoney(Math.abs(analysis.net_premium))}</div>
            </div>
            <div className="stat">
              <div className="stat-label">max profit</div>
              <div
                className={`stat-value ${analysis.max_profit === null ? "text-profit" : ""}`}
              >
                {money(analysis.max_profit)}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">max loss</div>
              {/* Unbounded is the one that has to be unmissable. */}
              <div
                className={`stat-value ${
                  analysis.max_loss === null ? "text-critical" : "text-loss"
                }`}
              >
                {money(analysis.max_loss)}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">break-even</div>
              <div className="stat-value">
                {analysis.breakevens.length === 0
                  ? "none"
                  : analysis.breakevens.map((b) => b.toFixed(0)).join(" / ")}
              </div>
            </div>
          </div>

          {chart ? (
            <svg
              className="builder-chart"
              viewBox={`0 0 ${chart.width} ${chart.height}`}
              preserveAspectRatio="none"
            >
              <line
                x1={0}
                y1={chart.zero}
                x2={chart.width}
                y2={chart.zero}
                className="builder-zero"
              />
              <line
                x1={chart.spotX}
                y1={0}
                x2={chart.spotX}
                y2={chart.height}
                className="builder-spot"
              />
              {chart.value ? <path d={chart.value} className="builder-value" /> : null}
              <path d={chart.payoff} className="builder-payoff" />
            </svg>
          ) : null}

          <div className="builder-legend">
            <span className="builder-key payoff" /> at expiry
            <span className="builder-key value" /> marked now
            {analysis.value.length === 0 ? (
              <span className="text-secondary">
                — present value withheld: a leg has no implied volatility
              </span>
            ) : null}
          </div>

          <dl className="builder-greeks">
            <dt>delta</dt>
            <dd>{analysis.delta === null ? "—" : analysis.delta.toFixed(1)}</dd>
            <dt>gamma</dt>
            <dd>{analysis.gamma === null ? "—" : analysis.gamma.toFixed(3)}</dd>
            <dt>vega</dt>
            <dd>{analysis.vega === null ? "—" : analysis.vega.toFixed(1)}</dd>
            <dt>theta/day</dt>
            <dd className={analysis.theta !== null && analysis.theta < 0 ? "text-loss" : ""}>
              {analysis.theta === null ? "—" : analysis.theta.toFixed(1)}
            </dd>
          </dl>

          {analysis.note ? <div className="analytics-note">{analysis.note}</div> : null}
        </>
      ) : null}
    </div>
  );
}
