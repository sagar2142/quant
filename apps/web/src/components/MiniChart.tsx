/**
 * Small canvas primitives for the analysis blocks — MASTER_PLAN §12.6.
 *
 * A line, an area and a histogram, drawn the same way `Chart` draws candles:
 * on canvas, from plain arrays, with no per-point DOM. These render beside a
 * 1,891-candle chart in the same window, and eight SVG panes of a thousand
 * points each would make the whole screen stutter on every hover.
 *
 * **No axis labels beyond the extremes.** These are shape-readers — is the
 * drawdown deepening, is the distribution left-tailed, is volatility in a new
 * regime. The exact numbers live in the stat blocks next to them, and
 * repeating them here would spend pixels to say something twice.
 */

import { useCallback, useEffect, useRef } from "react";

export type MiniKind = "line" | "area" | "histogram";

export interface MiniChartProps {
  values: number[];
  kind?: MiniKind;
  height?: number;
  colour?: string;
  /** Draw a reference line at this value, e.g. zero. */
  baseline?: number;
  /** Labels for the first and last point, drawn at the ends. */
  first?: string;
  last?: string;
}

const AXIS = "#5e6772";
const GRID = "#1e2329";

export function MiniChart({
  values,
  kind = "line",
  height = 96,
  colour = "#4a9eff",
  baseline,
  first,
  last,
}: MiniChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const width = wrap.clientWidth;
    if (width <= 0) return;

    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "9px ui-monospace, monospace";

    const usable = values.filter((v) => Number.isFinite(v));
    if (usable.length === 0) {
      ctx.fillStyle = AXIS;
      ctx.fillText("no data", 4, height / 2);
      return;
    }

    const pad = { top: 6, right: 4, bottom: first || last ? 12 : 4, left: 4 };
    const plotH = height - pad.top - pad.bottom;
    const plotW = width - pad.left - pad.right;

    let low = Math.min(...usable);
    let high = Math.max(...usable);
    if (baseline != null) {
      low = Math.min(low, baseline);
      high = Math.max(high, baseline);
    }
    // A flat series has no range to scale against and would divide by zero,
    // so it is given one and drawn down the middle — which is what "it did not
    // move" should look like.
    if (high === low) {
      high += 1;
      low -= 1;
    }
    const toY = (v: number): number => pad.top + plotH - ((v - low) / (high - low)) * plotH;

    if (baseline != null) {
      const y = Math.round(toY(baseline)) + 0.5;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
    }

    if (kind === "histogram") {
      const slot = plotW / values.length;
      ctx.fillStyle = colour;
      values.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const y = toY(v);
        const zero = toY(Math.max(low, 0));
        ctx.fillRect(pad.left + i * slot, Math.min(y, zero), Math.max(slot - 1, 1), Math.abs(zero - y) || 1);
      });
    } else {
      const step = plotW / Math.max(values.length - 1, 1);
      ctx.beginPath();
      let started = false;
      values.forEach((v, i) => {
        if (!Number.isFinite(v)) {
          started = false;
          return;
        }
        const x = pad.left + i * step;
        const y = toY(v);
        if (started) ctx.lineTo(x, y);
        else {
          ctx.moveTo(x, y);
          started = true;
        }
      });

      if (kind === "area") {
        const zero = toY(baseline ?? Math.max(low, 0));
        ctx.lineTo(pad.left + (values.length - 1) * step, zero);
        ctx.lineTo(pad.left, zero);
        ctx.closePath();
        ctx.fillStyle = `${colour}33`;
        ctx.fill();
        ctx.beginPath();
        started = false;
        values.forEach((v, i) => {
          if (!Number.isFinite(v)) {
            started = false;
            return;
          }
          const x = pad.left + i * step;
          const y = toY(v);
          if (started) ctx.lineTo(x, y);
          else {
            ctx.moveTo(x, y);
            started = true;
          }
        });
      }
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (first || last) {
      ctx.fillStyle = AXIS;
      if (first) ctx.fillText(first, pad.left, height - 2);
      if (last) {
        const w = ctx.measureText(last).width;
        ctx.fillText(last, width - pad.right - w, height - 2);
      }
    }
  }, [values, kind, height, colour, baseline, first, last]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const observer = new ResizeObserver(() => draw());
    const wrap = wrapRef.current;
    if (wrap) observer.observe(wrap);
    return () => observer.disconnect();
  }, [draw]);

  return (
    <div ref={wrapRef} className="mini-chart" style={{ height }}>
      <canvas ref={canvasRef} />
    </div>
  );
}

/** Peak-to-trough decline at each point, as a negative fraction. */
export function drawdownSeries(closes: number[]): number[] {
  let peak = -Infinity;
  return closes.map((close) => {
    peak = Math.max(peak, close);
    return peak > 0 ? close / peak - 1 : 0;
  });
}

/** Simple returns from a close series. */
export function returnSeries(closes: number[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < closes.length; i += 1) {
    const previous = closes[i - 1];
    const current = closes[i];
    if (previous == null || current == null || previous === 0) continue;
    out.push(current / previous - 1);
  }
  return out;
}

/** Annualised standard deviation over a trailing window. */
export function rollingVol(returns: number[], window: number): number[] {
  const out: number[] = new Array(returns.length).fill(Number.NaN);
  for (let i = window - 1; i < returns.length; i += 1) {
    const slice = returns.slice(i - window + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / (slice.length - 1);
    out[i] = Math.sqrt(variance) * Math.sqrt(252);
  }
  return out;
}

/** Bucket counts for a histogram, and the edges they sit between. */
export function histogram(values: number[], buckets = 41): { counts: number[]; low: number; high: number } {
  const usable = values.filter((v) => Number.isFinite(v));
  if (usable.length === 0) return { counts: [], low: 0, high: 0 };
  const low = Math.min(...usable);
  const high = Math.max(...usable);
  const counts = new Array(buckets).fill(0);
  const span = high - low || 1;
  for (const v of usable) {
    const index = Math.min(buckets - 1, Math.floor(((v - low) / span) * buckets));
    counts[index] += 1;
  }
  return { counts, low, high };
}
