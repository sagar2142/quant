/**
 * Icons — MASTER_PLAN §12.2.
 *
 * Real SVG, not glyphs. The console was drawing its interface out of Unicode
 * characters — ✓, ✗, ⤢, ▥ — which is fine until it is not: those render in
 * whatever font the platform substitutes, so a check mark is a thin hairline
 * on one machine and a coloured emoji on another, and ▥ has no glyph at all in
 * several common monospace faces. An icon that renders as a tofu box is worse
 * than no icon.
 *
 * **Sized in `em` and coloured by `currentColor`**, so an icon inherits the
 * type scale and the state colour of whatever it sits in. A pass mark is green
 * because its row is green, not because the icon hardcodes it.
 *
 * Stroke-based on a 24-grid, `stroke-width: 2`, round caps — one visual family
 * rather than a collection of found symbols.
 */

export type IconName =
  | "check"
  | "cross"
  | "expand"
  | "collapse"
  | "close"
  | "market"
  | "research"
  | "trade"
  | "system"
  | "guide"
  | "search"
  | "plus"
  | "refresh"
  | "popout";

const PATHS: Record<IconName, React.ReactNode> = {
  check: <polyline points="20 6 9 17 4 12" />,
  cross: (
    <>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </>
  ),
  expand: (
    <>
      <polyline points="15 3 21 3 21 9" />
      <polyline points="9 21 3 21 3 15" />
      <line x1="21" y1="3" x2="14" y2="10" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </>
  ),
  collapse: (
    <>
      <polyline points="4 14 10 14 10 20" />
      <polyline points="20 10 14 10 14 4" />
      <line x1="14" y1="10" x2="21" y2="3" />
      <line x1="3" y1="21" x2="10" y2="14" />
    </>
  ),
  close: (
    <>
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </>
  ),
  //: Candles, for the charts workspace.
  market: (
    <>
      <line x1="7" y1="3" x2="7" y2="21" />
      <rect x="4" y="7" width="6" height="9" rx="1" />
      <line x1="17" y1="3" x2="17" y2="21" />
      <rect x="14" y="10" width="6" height="7" rx="1" />
    </>
  ),
  //: A lens over a series, for evaluating one name.
  research: (
    <>
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
      <polyline points="8 12 10.5 9.5 13 12.5 15 10" />
    </>
  ),
  //: Two directions, for order entry.
  trade: (
    <>
      <polyline points="7 3 7 21" />
      <polyline points="4 6 7 3 10 6" />
      <polyline points="17 21 17 3" />
      <polyline points="14 18 17 21 20 18" />
    </>
  ),
  //: A warning triangle, for limits and gates.
  system: (
    <>
      <path d="M12 3 L22 20 L2 20 Z" />
      <line x1="12" y1="10" x2="12" y2="14" />
      <line x1="12" y1="17" x2="12" y2="17" />
    </>
  ),
  guide: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M9.5 9.5a2.6 2.6 0 1 1 3.4 2.5c-.6.2-.9.8-.9 1.4v.6" />
      <line x1="12" y1="17.5" x2="12" y2="17.5" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </>
  ),
  plus: (
    <>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </>
  ),
  //: A pane leaving its frame, for a panel that opens in its own OS window.
  popout: (
    <>
      <path d="M13 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-8" />
      <polyline points="15 3 21 3 21 9" />
      <line x1="10" y1="14" x2="21" y2="3" />
    </>
  ),
  refresh: (
    <>
      <polyline points="21 4 21 10 15 10" />
      <path d="M20 14a8 8 0 1 1-2-7l3 3" />
    </>
  ),
};

export interface IconProps {
  name: IconName;
  /** Multiple of the current font size. */
  size?: number;
  title?: string;
}

export function Icon({ name, size = 1, title }: IconProps) {
  return (
    <svg
      className="icon"
      viewBox="0 0 24 24"
      width={`${size}em`}
      height={`${size}em`}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      // Decorative unless it is the only label, in which case the title is the
      // accessible name and the role changes with it.
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      focusable="false"
    >
      {title && <title>{title}</title>}
      {PATHS[name]}
    </svg>
  );
}
