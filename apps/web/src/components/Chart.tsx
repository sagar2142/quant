/**
 * Candlestick chart — MASTER_PLAN §12.6.
 *
 * The console could describe a security in detail and draw only a sparkline of
 * its closes, which is the one view that cannot answer "what happened that
 * day". A candle carries the open, the high, the low and the close; the
 * sparkline threw four of those five away because `/series` returned only the
 * close. `/security/{symbol}/ohlc` returns all of them.
 *
 * **Canvas, not SVG.** The NSE panel is 1,891 sessions per name, and 1,891 DOM
 * nodes per pane re-laid-out on every crosshair move is not a chart, it is a
 * slideshow. Canvas draws the same frame in one pass.
 *
 * **The payload is transposed into candles on arrival.** The API sends column
 * arrays, which are compact on the wire and wrong to draw from: under
 * `noUncheckedIndexedAccess` every `open[i]` is possibly undefined, and five
 * parallel arrays give five chances to read the wrong row. One array of
 * candles is indexed once, and every loop below iterates values.
 *
 * **Prices are back-adjusted server-side** (§9). A 1:1 bonus read from raw
 * closes is a -50% candle that never traded, which is exactly the kind of
 * artefact a chart makes look like a real event.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

interface OhlcPayload {
  symbol: string;
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface Candle {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ChartProps {
  symbol: string;
  /** Which exchange. A dual-listed name has a different price on each. */
  venue?: string;
  /** Sessions to load. 0 is the whole panel. */
  sessions?: number;
  /**
   * Poll the vendor for a delayed last-traded price and show it beside the
   * close. Off by default: it is a network call per pane, and a six-pane
   * workspace should not make six of them unasked.
   */
  live?: boolean;
  /** Moving averages to overlay, in sessions. */
  overlays?: number[];
  /**
   * Log price axis. Right for multi-year spans, where a linear axis makes an
   * old 10% move look smaller than a recent one.
   */
  logScale?: boolean;
  height?: number;
}

/** Fraction of the pane given to volume. */
const VOLUME_SHARE = 0.22;
const PADDING = { top: 8, right: 62, bottom: 20, left: 8 };
/** Fewest candles a zoom may show, so the view cannot collapse to nothing. */
const MIN_VISIBLE = 12;

const UP = "#26a69a";
const DOWN = "#ef5350";
const GRID = "#1e2329";
const AXIS = "#5e6772";
const CROSSHAIR = "#98a1ac";
const OVERLAY_COLOURS = ["#4a9eff", "#f0a93b", "#b48ead"];

function toCandles(payload: OhlcPayload): Candle[] {
  const out: Candle[] = [];
  for (let i = 0; i < payload.close.length; i += 1) {
    const open = payload.open[i];
    const high = payload.high[i];
    const low = payload.low[i];
    const close = payload.close[i];
    const volume = payload.volume[i];
    const date = payload.dates[i];
    // A row missing any leg is dropped rather than defaulted. A candle drawn
    // with a zero low is a chart of something that did not happen.
    if (
      open == null ||
      high == null ||
      low == null ||
      close == null ||
      volume == null ||
      date == null
    ) {
      continue;
    }
    out.push({ date, open, high, low, close, volume });
  }
  return out;
}

function niceTicks(low: number, high: number, count: number): number[] {
  if (!Number.isFinite(low) || !Number.isFinite(high) || high <= low) return [];
  const raw = (high - low) / count;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(low / step) * step; v <= high; v += step) ticks.push(v);
  return ticks;
}

/** Simple moving average over closes, null until the window is full. */
function sma(candles: Candle[], window: number): (number | null)[] {
  const out: (number | null)[] = new Array(candles.length).fill(null);
  let sum = 0;
  candles.forEach((candle, i) => {
    sum += candle.close;
    const leaving = candles[i - window];
    if (leaving) sum -= leaving.close;
    if (i >= window - 1) out[i] = sum / window;
  });
  return out;
}

function formatVolume(v: number): string {
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)}Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)}L`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(0);
}

export function Chart({
  symbol,
  venue = "NSE",
  sessions = 0,
  live = false,
  overlays = [20, 50, 200],
  logScale = false,
  height = 380,
}: ChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [hover, setHover] = useState<{ index: number; y: number } | null>(null);
  //: Visible slice, as indices into the full series. Zoom narrows it, pan
  //: slides it. Held here rather than derived so the view survives a redraw.
  const [view, setView] = useState<{ from: number; to: number }>({ from: 0, to: 0 });
  const [quote, setQuote] = useState<{ price: number; age: number } | null>(null);
  const dragRef = useRef<{ x: number; from: number; to: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(
      `/api/security/${encodeURIComponent(symbol)}/ohlc` +
        `?sessions=${sessions}&venue=${encodeURIComponent(venue)}`,
    )
      .then(async (response) => {
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(body?.detail ?? `HTTP ${response.status}`);
        }
        return (await response.json()) as OhlcPayload;
      })
      .then((payload) => {
        if (cancelled) return;
        const rows = toCandles(payload);
        setCandles(rows);
        setView({ from: 0, to: rows.length });
      })
      .catch((exc: Error) => {
        if (cancelled) return;
        setCandles([]);
        setError(exc.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, sessions, venue]);

  //: The last candle is the last *close*, which mid-session is hours old. The
  //: vendor price is delayed too, and its own age is shown rather than implied:
  //: a delayed quote presented as live is worse than no quote. It is never
  //: mixed into the candles — everything drawn above carries a receive_time and
  //: this does not, so joining them would break the point-in-time discipline
  //: the research rests on (§3).
  useEffect(() => {
    if (!live || !symbol) return;
    let cancelled = false;
    const poll = () => {
      fetch(`/api/quotes?symbols=${encodeURIComponent(symbol)}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((payload) => {
          if (cancelled || !payload) return;
          const row = payload.quotes?.[symbol];
          setQuote(row ? { price: Number(row.price), age: Number(row.age_seconds) } : null);
        })
        .catch(() => !cancelled && setQuote(null));
    };
    poll();
    const timer = setInterval(poll, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [live, symbol]);

  const averages = useMemo(
    () => overlays.map((window) => sma(candles, window)),
    [candles, overlays],
  );

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap || candles.length === 0) return;

    const ratio = window.devicePixelRatio || 1;
    const width = wrap.clientWidth;
    if (width <= 0) return;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "10px ui-monospace, monospace";

    const from = Math.max(0, view.from);
    const to = Math.min(candles.length, view.to);
    const visible = candles.slice(from, to);
    if (visible.length === 0) return;

    const plotW = width - PADDING.left - PADDING.right;
    const volumeH = (height - PADDING.top - PADDING.bottom) * VOLUME_SHARE;
    const priceH = height - PADDING.top - PADDING.bottom - volumeH - 6;

    let low = Infinity;
    let high = -Infinity;
    let maxVolume = 0;
    visible.forEach((candle) => {
      if (candle.low < low) low = candle.low;
      if (candle.high > high) high = candle.high;
      if (candle.volume > maxVolume) maxVolume = candle.volume;
    });
    averages.forEach((series) => {
      for (let i = from; i < to; i += 1) {
        const value = series[i];
        if (value == null) continue;
        if (value < low) low = value;
        if (value > high) high = value;
      }
    });
    if (!Number.isFinite(low) || !Number.isFinite(high)) return;

    const pad = (high - low) * 0.06 || high * 0.02 || 1;
    low -= pad;
    high += pad;
    //: A log axis cannot show a non-positive price. Prices are positive, but a
    //: padded floor can cross zero on a penny stock, so it is clamped rather
    //: than left to produce NaN across the whole pane.
    if (logScale) low = Math.max(low, 1e-6);

    const logLow = Math.log(Math.max(low, 1e-6));
    const logHigh = Math.log(Math.max(high, 1e-6));
    const toY = (price: number): number => {
      if (logScale) {
        const t = (Math.log(Math.max(price, 1e-6)) - logLow) / (logHigh - logLow);
        return PADDING.top + priceH - t * priceH;
      }
      return PADDING.top + priceH - ((price - low) / (high - low)) * priceH;
    };
    const slot = plotW / visible.length;
    const toX = (index: number): number => PADDING.left + (index - from) * slot + slot / 2;

    // ── grid and price axis ────────────────────────────────────────────────
    ctx.lineWidth = 1;
    niceTicks(low, high, 5).forEach((tick) => {
      const y = Math.round(toY(tick)) + 0.5;
      ctx.strokeStyle = GRID;
      ctx.beginPath();
      ctx.moveTo(PADDING.left, y);
      ctx.lineTo(width - PADDING.right, y);
      ctx.stroke();
      ctx.fillStyle = AXIS;
      ctx.fillText(tick.toFixed(tick >= 1000 ? 0 : 2), width - PADDING.right + 5, y + 3);
    });

    // ── volume ─────────────────────────────────────────────────────────────
    const volumeTop = PADDING.top + priceH + 6;
    if (maxVolume > 0) {
      visible.forEach((candle, k) => {
        const h = (candle.volume / maxVolume) * volumeH;
        ctx.fillStyle = candle.close >= candle.open ? "#26a69a44" : "#ef535044";
        ctx.fillRect(
          toX(from + k) - Math.max(slot * 0.35, 0.5),
          volumeTop + volumeH - h,
          Math.max(slot * 0.7, 1),
          h,
        );
      });
    }

    // ── candles ────────────────────────────────────────────────────────────
    const bodyW = Math.max(slot * 0.66, 1);
    visible.forEach((candle, k) => {
      const colour = candle.close >= candle.open ? UP : DOWN;
      const x = toX(from + k);
      ctx.strokeStyle = colour;
      ctx.fillStyle = colour;

      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, toY(candle.high));
      ctx.lineTo(Math.round(x) + 0.5, toY(candle.low));
      ctx.stroke();

      const yOpen = toY(candle.open);
      const yClose = toY(candle.close);
      // A doji has zero body height and would draw nothing at all, so it is
      // floored to one pixel: a session that opened and closed at the same
      // price is information, not the absence of it.
      const bodyH = Math.max(Math.abs(yClose - yOpen), 1);
      const top = Math.min(yOpen, yClose);
      if (slot > 2.5) ctx.fillRect(x - bodyW / 2, top, bodyW, bodyH);
      else ctx.fillRect(x - 0.5, top, 1, bodyH);
    });

    // ── moving averages ────────────────────────────────────────────────────
    averages.forEach((series, index) => {
      ctx.strokeStyle = OVERLAY_COLOURS[index % OVERLAY_COLOURS.length] ?? AXIS;
      ctx.lineWidth = 1;
      ctx.beginPath();
      let started = false;
      for (let i = from; i < to; i += 1) {
        const value = series[i];
        if (value == null) {
          started = false;
          continue;
        }
        const x = toX(i);
        const y = toY(value);
        if (started) ctx.lineTo(x, y);
        else {
          ctx.moveTo(x, y);
          started = true;
        }
      }
      ctx.stroke();
    });

    // ── time axis ──────────────────────────────────────────────────────────
    ctx.fillStyle = AXIS;
    const labels = Math.max(2, Math.min(8, Math.floor(plotW / 90)));
    for (let k = 0; k < labels; k += 1) {
      const offset = Math.floor((k * (visible.length - 1)) / Math.max(labels - 1, 1));
      const candle = visible[offset];
      if (!candle) continue;
      const text = candle.date.slice(2);
      const w = ctx.measureText(text).width;
      const x = toX(from + offset) - w / 2;
      ctx.fillText(text, Math.min(Math.max(x, 2), width - PADDING.right - w), height - 6);
    }

    // ── crosshair ──────────────────────────────────────────────────────────
    if (hover && hover.index >= from && hover.index < to) {
      const x = toX(hover.index);
      ctx.strokeStyle = CROSSHAIR;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, PADDING.top);
      ctx.lineTo(Math.round(x) + 0.5, height - PADDING.bottom);
      ctx.moveTo(PADDING.left, Math.round(hover.y) + 0.5);
      ctx.lineTo(width - PADDING.right, Math.round(hover.y) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [candles, view, hover, averages, height, logScale]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const observer = new ResizeObserver(() => draw());
    const wrap = wrapRef.current;
    if (wrap) observer.observe(wrap);
    return () => observer.disconnect();
  }, [draw]);

  const indexAt = useCallback(
    (clientX: number): number | null => {
      const wrap = wrapRef.current;
      if (!wrap || candles.length === 0) return null;
      const rect = wrap.getBoundingClientRect();
      const plotW = rect.width - PADDING.left - PADDING.right;
      const span = view.to - view.from;
      if (span <= 0 || plotW <= 0) return null;
      const i = view.from + Math.floor(((clientX - rect.left - PADDING.left) / plotW) * span);
      return Math.min(view.to - 1, Math.max(view.from, i));
    },
    [candles.length, view],
  );

  const onMove = (event: React.MouseEvent<HTMLDivElement>) => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const drag = dragRef.current;

    if (drag) {
      const span = drag.to - drag.from;
      const plotW = rect.width - PADDING.left - PADDING.right;
      const shift = Math.round((-(event.clientX - drag.x) / plotW) * span);
      let from = drag.from + shift;
      let to = drag.to + shift;
      if (from < 0) {
        to -= from;
        from = 0;
      }
      if (to > candles.length) {
        from -= to - candles.length;
        to = candles.length;
      }
      setView({ from: Math.max(0, from), to });
      return;
    }

    const index = indexAt(event.clientX);
    if (index != null) setHover({ index, y: event.clientY - rect.top });
  };

  const onWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (candles.length === 0) return;
    const anchor = indexAt(event.clientX);
    if (anchor == null) return;
    const span = view.to - view.from;
    const next = Math.round(span * (event.deltaY > 0 ? 1.15 : 0.87));
    const width = Math.min(candles.length, Math.max(MIN_VISIBLE, next));
    // Zoom about the cursor, so the candle under the pointer stays put.
    // Zooming about the centre makes a chart feel like it is fighting you.
    const share = (anchor - view.from) / Math.max(span, 1);
    let from = Math.round(anchor - share * width);
    let to = from + width;
    if (from < 0) {
      to -= from;
      from = 0;
    }
    if (to > candles.length) {
      from = Math.max(0, candles.length - width);
      to = candles.length;
    }
    setView({ from, to });
  };

  const active = hover ? candles[hover.index] : undefined;
  const previous = hover ? candles[hover.index - 1] : undefined;
  const change = active && previous ? ((active.close - previous.close) / previous.close) * 100 : null;

  return (
    <div className="chart">
      <div className="chart-readout">
        <span className="chart-symbol">{symbol}</span>
        <span className="chart-venue">{venue}</span>
        {quote && (
          <span className="chart-live" title="Delayed vendor quote, never mixed into the candles">
            LTP <b>{quote.price.toFixed(2)}</b>
            <em>{quote.age < 90 ? `${Math.round(quote.age)}s` : `${Math.round(quote.age / 60)}m`} old</em>
          </span>
        )}
        {active ? (
          <>
            <span className="chart-date">{active.date}</span>
            <span>
              O <b>{active.open.toFixed(2)}</b>
            </span>
            <span>
              H <b>{active.high.toFixed(2)}</b>
            </span>
            <span>
              L <b>{active.low.toFixed(2)}</b>
            </span>
            <span>
              C <b>{active.close.toFixed(2)}</b>
            </span>
            <span>V {formatVolume(active.volume)}</span>
            {change != null && (
              <span className={change >= 0 ? "up" : "down"}>
                {change >= 0 ? "+" : ""}
                {change.toFixed(2)}%
              </span>
            )}
          </>
        ) : (
          <span className="chart-hint">
            {loading ? "loading…" : error ? error : "scroll to zoom · drag to pan"}
          </span>
        )}
        <span className="chart-legend">
          {overlays.map((w, i) => (
            <span key={w} style={{ color: OVERLAY_COLOURS[i % OVERLAY_COLOURS.length] }}>
              MA{w}
            </span>
          ))}
        </span>
      </div>
      <div
        ref={wrapRef}
        className="chart-canvas"
        style={{ height }}
        onMouseMove={onMove}
        onMouseLeave={() => {
          setHover(null);
          dragRef.current = null;
        }}
        onMouseDown={(event) => {
          dragRef.current = { x: event.clientX, from: view.from, to: view.to };
        }}
        onMouseUp={() => {
          dragRef.current = null;
        }}
        onWheel={onWheel}
        onDoubleClick={() => setView({ from: 0, to: candles.length })}
      >
        <canvas ref={canvasRef} />
        {error && <div className="chart-error">{error}</div>}
      </div>
    </div>
  );
}
