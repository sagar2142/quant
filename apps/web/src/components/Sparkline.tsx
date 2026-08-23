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

/**
 * Return distribution, with the VaR and CVaR thresholds drawn on it.
 *
 * **A kurtosis of 7.2 means nothing as a number.** The same figure printed in a
 * table and drawn as a histogram with the 5% tail shaded are different pieces
 * of information: the second shows how far the left tail actually reaches, and
 * that is the part that ends a strategy. VaR is where the shading starts; CVaR
 * is the average of everything beyond it, which is why they are drawn as two
 * marks rather than one.
 */
export function ReturnHistogram({
  closes,
  var5,
  cvar5,
  bins = 41,
}: {
  closes: number[];
  var5: number | null;
  cvar5: number | null;
  bins?: number;
}) {
  if (closes.length < 3) return <div className="empty">Not enough history.</div>;

  const returns: number[] = [];
  for (let i = 1; i < closes.length; i += 1) {
    const prev = closes[i - 1];
    const cur = closes[i];
    if (prev && cur && prev > 0) returns.push(cur / prev - 1);
  }
  if (returns.length < 2) return <div className="empty">Not enough history.</div>;

  // Symmetric bounds so the asymmetry of the distribution is visible against a
  // centred axis rather than hidden by a shifted one.
  const bound = Math.max(...returns.map(Math.abs)) || 1;
  const counts = new Array<number>(bins).fill(0);
  for (const r of returns) {
    const idx = Math.min(bins - 1, Math.floor(((r + bound) / (2 * bound)) * bins));
    counts[idx] = (counts[idx] ?? 0) + 1;
  }
  const peak = Math.max(...counts) || 1;
  const barWidth = (WIDTH - PAD * 2) / bins;
  const xOf = (value: number) => PAD + ((value + bound) / (2 * bound)) * (WIDTH - PAD * 2);

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="daily return distribution"
    >
      {counts.map((count, i) => {
        const height = (count / peak) * (HEIGHT - PAD * 2);
        const left = PAD + i * barWidth;
        // Bars entirely beyond VaR are the tail the ratio hides.
        const centre = -bound + ((i + 0.5) / bins) * 2 * bound;
        const inTail = var5 !== null && centre <= var5;
        return (
          <rect
            key={i}
            x={left}
            y={HEIGHT - PAD - height}
            width={Math.max(1, barWidth - 0.5)}
            height={height}
            fill={inTail ? "var(--loss)" : "var(--text-secondary)"}
            fillOpacity={inTail ? 0.75 : 0.45}
          />
        );
      })}
      {var5 !== null ? (
        <line
          x1={xOf(var5)}
          x2={xOf(var5)}
          y1={PAD}
          y2={HEIGHT - PAD}
          stroke="var(--loss)"
          strokeWidth="1"
          strokeDasharray="3 2"
        />
      ) : null}
      {cvar5 !== null ? (
        <line
          x1={xOf(cvar5)}
          x2={xOf(cvar5)}
          y1={PAD}
          y2={HEIGHT - PAD}
          stroke="var(--loss)"
          strokeWidth="1.5"
        />
      ) : null}
    </svg>
  );
}

/**
 * Autocorrelation by lag. Negative at lag 1 is the mean-reversion signature.
 *
 * Drawn rather than tabulated because the *shape* across lags is the signal: a
 * single negative lag-1 bar is short-horizon reversal, while a decaying
 * positive series is trend. Four numbers in a row do not show that; four bars
 * either side of zero do.
 */
export function AutocorrelationBars({ values }: { values: Record<string, number> }) {
  const lags = Object.keys(values)
    .map(Number)
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);
  if (lags.length === 0) return <div className="empty">No autocorrelation computed.</div>;

  const scale = Math.max(0.1, ...lags.map((l) => Math.abs(values[String(l)] ?? 0)));
  const slot = (WIDTH - PAD * 2) / lags.length;
  const zero = HEIGHT / 2;

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="autocorrelation by lag"
    >
      <line x1={PAD} x2={WIDTH - PAD} y1={zero} y2={zero} stroke="var(--border)" strokeWidth="1" />
      {lags.map((lag, i) => {
        const value = values[String(lag)] ?? 0;
        const height = (Math.abs(value) / scale) * (HEIGHT / 2 - PAD);
        return (
          <rect
            key={lag}
            x={PAD + i * slot + slot * 0.28}
            y={value >= 0 ? zero - height : zero}
            width={slot * 0.44}
            height={Math.max(1, height)}
            fill={value >= 0 ? "var(--profit)" : "var(--loss)"}
            fillOpacity="0.8"
          />
        );
      })}
    </svg>
  );
}

/**
 * Trailing volatility, annualised, with the long-run level drawn across it.
 *
 * The profile reports a regime as a word — "elevated", "normal", "compressed" —
 * derived from where EWMA volatility sits against realised. The word is the
 * conclusion; this is the evidence, and it shows whether the name has been
 * there for a week or a year.
 */
export function VolatilityChart({ closes, window = 21 }: { closes: number[]; window?: number }) {
  if (closes.length < window + 2) return <div className="empty">Not enough history.</div>;

  const returns: number[] = [];
  for (let i = 1; i < closes.length; i += 1) {
    const prev = closes[i - 1];
    const cur = closes[i];
    if (prev && cur && prev > 0) returns.push(cur / prev - 1);
  }

  const rolling: number[] = [];
  for (let i = window; i <= returns.length; i += 1) {
    const slice = returns.slice(i - window, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    rolling.push(Math.sqrt(variance * 252));
  }
  if (rolling.length < 2) return <div className="empty">Not enough history.</div>;

  const average = rolling.reduce((a, b) => a + b, 0) / rolling.length;
  const low = Math.min(...rolling);
  const high = Math.max(...rolling);
  const span = high - low || 1;
  const yOf = (v: number) => PAD + (HEIGHT - PAD * 2) * (1 - (v - low) / span);

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label="rolling annualised volatility"
    >
      <line
        x1={PAD}
        x2={WIDTH - PAD}
        y1={yOf(average)}
        y2={yOf(average)}
        stroke="var(--text-secondary)"
        strokeWidth="1"
        strokeDasharray="4 3"
      />
      <path d={path(rolling, HEIGHT)} fill="none" stroke="var(--warn)" strokeWidth="1.5" />
    </svg>
  );
}
