/**
 * Number formatting — MASTER_PLAN §12.3.
 *
 * **Every number in the console renders through this module.** Not one
 * component formats its own, because that is exactly how `-0.5%` and
 * `(0.50)%` end up on the same screen, and how a P&L loses its sign.
 *
 * The rules, in one place:
 *
 *   P&L            always signed        +₹12,340.50 / −₹8,220.00
 *   percent change always signed, 2dp   +1.24% / −0.87%
 *   ratios         2dp, unsigned        1.42
 *   null / NA      em dash              —   (never blank, never 0)
 *   zero           explicit             0.00
 *
 * A minus sign, never parentheses: the accounting convention is harder to
 * scan quickly, and speed of reading is the whole point of a console.
 */

/** Indian grouping: ₹12,34,567 rather than ₹1,234,567. */
export type Grouping = "indian" | "western";

const EM_DASH = "—";

/** True for values that are genuinely absent, as opposed to zero. */
function isAbsent(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || !Number.isFinite(value);
}

/**
 * Indian digit grouping: last three digits, then pairs.
 * 1234567 → "12,34,567"
 */
function groupIndian(whole: string): string {
  if (whole.length <= 3) return whole;
  const last3 = whole.slice(-3);
  const rest = whole.slice(0, -3);
  return `${rest.replace(/\B(?=(\d{2})+(?!\d))/g, ",")},${last3}`;
}

function groupWestern(whole: string): string {
  return whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function withGrouping(value: number, decimals: number, grouping: Grouping): string {
  const parts = Math.abs(value).toFixed(decimals).split(".");
  // `noUncheckedIndexedAccess` is on: indexing yields `string | undefined`, and
  // the fallback is real rather than decorative — toFixed(0) has no fraction.
  const whole = parts[0] ?? "0";
  const fraction = parts[1];
  const grouped = grouping === "indian" ? groupIndian(whole) : groupWestern(whole);
  return fraction ? `${grouped}.${fraction}` : grouped;
}

/** Currency amount, unsigned. Use `formatPnL` for anything that can be negative. */
export function formatMoney(
  value: number | null | undefined,
  currency = "₹",
  grouping: Grouping = "indian",
  decimals = 2,
): string {
  if (isAbsent(value)) return EM_DASH;
  const sign = value < 0 ? "−" : "";
  return `${sign}${currency}${withGrouping(value, decimals, grouping)}`;
}

/**
 * P&L. **Always signed**, so the direction survives a colourblind palette and
 * a monochrome print (§12.2).
 */
export function formatPnL(
  value: number | null | undefined,
  currency = "₹",
  grouping: Grouping = "indian",
): string {
  if (isAbsent(value)) return EM_DASH;
  // U+2212 minus, not a hyphen: it aligns with digits in tabular figures.
  const sign = value < 0 ? "−" : "+";
  return `${sign}${currency}${withGrouping(value, 2, grouping)}`;
}

/** Percentage change. Always signed, 2dp. Input is a fraction: 0.0124 → +1.24%. */
export function formatPercent(
  value: number | null | undefined,
  decimals = 2,
): string {
  if (isAbsent(value)) return EM_DASH;
  const sign = value < 0 ? "−" : "+";
  return `${sign}${Math.abs(value * 100).toFixed(decimals)}%`;
}

/** Unsigned percentage, for levels rather than changes (exposure, utilisation). */
export function formatLevel(
  value: number | null | undefined,
  decimals = 1,
): string {
  if (isAbsent(value)) return EM_DASH;
  return `${(value * 100).toFixed(decimals)}%`;
}

/** A ratio such as Sharpe. Unsigned formatting; negatives keep their minus. */
export function formatRatio(
  value: number | null | undefined,
  decimals = 2,
): string {
  if (isAbsent(value)) return EM_DASH;
  return value < 0
    ? `−${Math.abs(value).toFixed(decimals)}`
    : value.toFixed(decimals);
}

/** Price, at the instrument's tick precision. */
export function formatPrice(
  value: number | null | undefined,
  decimals = 2,
  grouping: Grouping = "indian",
): string {
  if (isAbsent(value)) return EM_DASH;
  return withGrouping(value, decimals, grouping);
}

/** Integer count. */
export function formatCount(value: number | null | undefined): string {
  if (isAbsent(value)) return EM_DASH;
  return Math.round(value).toString();
}

/**
 * Direction glyph. Paired with every P&L so that colour is never the sole
 * encoding of direction (§12.2).
 */
export function directionGlyph(value: number | null | undefined): string {
  if (isAbsent(value) || value === 0) return "";
  return value > 0 ? "▲" : "▼";
}

/** Semantic class for a signed value. Returns the flat token at exactly zero. */
export function signClass(value: number | null | undefined): string {
  if (isAbsent(value) || value === 0) return "text-flat";
  return value > 0 ? "text-profit" : "text-loss";
}

/** Feed staleness, in the shortest readable form. */
export function formatStaleness(seconds: number | null | undefined): string {
  if (isAbsent(seconds)) return EM_DASH;
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}
