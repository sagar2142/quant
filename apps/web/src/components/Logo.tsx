/**
 * The Neutron mark — MASTER_PLAN §12.2.
 *
 * Two bars around a zero axis, one up and one down, placed so the figure is
 * unchanged by a 180° rotation about its centre. That symmetry *is* the name.
 * A neutron carries no charge; a market-neutral book carries no market — and
 * the central finding of this system's research is that a signal which looked
 * like alpha was 72.7% beta, which is the same statement about what is left
 * once the opposite side cancels.
 *
 * **The axis is drawn, and it is the point.** A bar chart without a zero line
 * is a picture of magnitudes; with one it is a picture of sign. Everything
 * this system measures — P&L, factor contribution, slippage, drawdown — is
 * signed, and the console's own rule is that zero is a measurement rather than
 * an absence.
 *
 * **Drawn to survive 16 pixels**, and cut down until it did. A favicon is the
 * smallest thing an interface renders, and the first version of this mark used
 * four bars — which rasterised to a smudge at tab size, where the strokes and
 * their gaps both fell below two pixels. Three strokes is the whole design now,
 * and the offset silhouette is what carries it once the axis has thinned away.
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

//: The bars, as (x, height). Positive is above the axis, negative below.
//:
//: **Two, not four.** The first version had four, alternating, summing to
//: zero — a truer picture of a book, and unreadable at sixteen pixels: four
//: strokes and their gaps landed at about one and a half pixels each and the
//: mark rendered as a smudge. Two bars can be half as wide again, and the
//: rotational symmetry survives the loss where the detail did not.
const BARS: readonly (readonly [number, number])[] = [
  [11, 10],
  [21, -10],
];

const AXIS_Y = 16;

//: The axis runs five units past the outermost bar on each side — enough to
//: read as a reference line rather than a crossbar, short enough not to leave
//: a tail hanging off the mark.
const AXIS_X0 = 6;
const AXIS_X1 = 26;

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

      {/* Zero. Thinner and dimmer than the bars: it is the reference they are
          measured against, not one of them. */}
      <line
        x1={AXIS_X0}
        y1={AXIS_Y}
        x2={AXIS_X1}
        y2={AXIS_Y}
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.45"
      />

      {BARS.map(([x, height]) => (
        <line
          key={x}
          x1={x}
          y1={AXIS_Y}
          x2={x}
          y2={AXIS_Y - height}
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
        />
      ))}
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
