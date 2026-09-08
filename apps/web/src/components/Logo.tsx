/**
 * The Neutron mark — MASTER_PLAN §12.2.
 *
 * A ring cut through its middle: a particle, opened along zero.
 *
 * **The cut is the whole idea.** A neutron carries no charge, and the two arcs
 * are equal, opposite and separated by exactly the line they are measured
 * against. A market-neutral book is the same statement — the finding this
 * system's research keeps returning to is that a signal which looked like alpha
 * was 72.7% beta, which is what is left once the opposite side cancels. The
 * figure is unchanged by a 180° rotation about its centre, so the symmetry is
 * structural rather than decorative.
 *
 * **The gap is a measurement, not a space.** Everything this system reports is
 * signed — P&L, factor contribution, slippage, drawdown — and the console's own
 * rule is that zero is a measurement rather than an absence. The mark draws
 * zero as the one thing that separates the two halves.
 *
 * **Two arcs, because a favicon is sixteen pixels.** Earlier attempts failed
 * there and the failure is always the same: four bars rasterised to a smudge
 * when the strokes and their gaps both fell under two pixels, and offsetting
 * the two halves horizontally read as a printing error rather than as
 * displacement. A closed curve survives downsampling in a way an arrangement of
 * separate strokes does not — the eye completes a circle from very little.
 *
 * Round terminals on both arcs, so the cut reads as a deliberate slot with
 * finished ends rather than a shape that was masked off.
 *
 * `currentColor` throughout, so the mark takes the colour of whatever it sits
 * in rather than carrying a brand colour that fights every context.
 */

export interface LogoProps {
  /** Pixel size. The mark is square. */
  size?: number;
  /** Draw the wordmark beside it. */
  wordmark?: boolean;
  title?: string;
}

//: Radius 9 in a 32-unit box, stroked at 5.4, leaves the drawn mark spanning
//: 4.3 to 27.7 — a little under three units of margin on every side, which is
//: what keeps it from crowding text set beside it.
//:
//: The arcs meet y = 16 ∓ 1.7, so each terminal sits at x = 16 ± √(9² − 1.7²)
//: = 16 ± 8.84. A wider gap breaks the circle before the eye can close it; a
//: narrower one closes up entirely at sixteen pixels and the cut disappears.
const ARC_LEFT_X = 7.16;
const ARC_RIGHT_X = 24.84;
const UPPER_Y = 14.3;
const LOWER_Y = 17.7;

//: Both arcs are the major one — their endpoints sit on the *same* side of
//: centre as the arc itself — so the large-arc flag is set on each. The sweep
//: differs because they curve away from one another.
const UPPER_ARC = `M ${ARC_LEFT_X} ${UPPER_Y} A 9 9 0 1 1 ${ARC_RIGHT_X} ${UPPER_Y}`;
const LOWER_ARC = `M ${ARC_LEFT_X} ${LOWER_Y} A 9 9 0 1 0 ${ARC_RIGHT_X} ${LOWER_Y}`;

export function Logo({ size = 20, wordmark = false, title }: LogoProps) {
  const mark = (
    <svg
      viewBox="0 0 32 32"
      width={size}
      height={size}
      fill="none"
      className="logo-mark"
      aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined}
      focusable="false"
    >
      {title && <title>{title}</title>}
      <path d={UPPER_ARC} stroke="currentColor" strokeWidth="5.4" strokeLinecap="round" />
      <path d={LOWER_ARC} stroke="currentColor" strokeWidth="5.4" strokeLinecap="round" />
    </svg>
  );

  if (!wordmark) return mark;

  return (
    <span className="logo">
      {mark}
      <span className="logo-word">NEUTRON</span>
    </span>
  );
}
