/**
 * Price and drawdown charts — MASTER_PLAN §12.6.
 *
 * Inline SVG, no charting library. A line and a filled area are about forty
 * lines of path arithmetic; the alternative is a dependency measured in
 * hundreds of kilobytes to draw two shapes.
 *
 * **The drawdown panel is not decoration.** An equity line alone invites
 * "that looks good"; the depth and duration underneath it is what a position
 * would actually have felt like, and showing them together is the honest
 * pairing.
 */

const WIDTH = 720;
const HEIGHT = 132;
const PAD = 4;

function path(values: number[], height: number): string {
  if (values.length < 2) return "";
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const step = (WIDTH - PAD * 2) / (values.length - 1);
  return values
    .map((value, index) => {
      const x = PAD + index * step;
      const y = PAD + (height - PAD * 2) * (1 - (value - low) / span);
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function PriceChart({ closes, label }: { closes: number[]; label: string }) {
  const first = closes[0];
  const last = closes[closes.length - 1];
  if (closes.length < 2 || first === undefined || last === undefined) {
    return <div className="empty">Not enough history to plot.</div>;
  }

  const line = path(closes, HEIGHT);
  const rising = last >= first;
  const stroke = rising ? "var(--profit)" : "var(--loss)";

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`${label} price`}
    >
      <path d={`${line} L${WIDTH - PAD},${HEIGHT} L${PAD},${HEIGHT} Z`} fill={stroke} fillOpacity="0.08" />
      <path d={line} fill="none" stroke={stroke} strokeWidth="1.5" />
    </svg>
  );
}

export function DrawdownChart({ closes }: { closes: number[] }) {
  if (closes.length < 2) return null;

  // Running peak, then distance below it. Computed here rather than fetched:
  // it is a transform of data already on screen, and a round trip to compute
  // a cumulative maximum would be absurd.
  let peak = closes[0] ?? 0;
  const drawdown = closes.map((close) => {
    peak = Math.max(peak, close);
    return close / peak - 1;
  });

  const line = path(drawdown, HEIGHT);
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="drawdown"
    >
      <path d={`${line} L${WIDTH - PAD},${PAD} L${PAD},${PAD} Z`} fill="var(--loss)" fillOpacity="0.14" />
      <path d={line} fill="none" stroke="var(--loss)" strokeWidth="1.5" />
    </svg>
  );
}
