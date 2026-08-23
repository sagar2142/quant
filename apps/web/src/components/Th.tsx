/**
 * A table heading that explains itself — MASTER_PLAN §12.3.
 *
 * **Every column on this console is a term of art.** `maxDD`, `hurst`,
 * `process`, `IC` and `observed` are meaningful to someone who already knows
 * them and opaque to everyone else, and a trading console that requires prior
 * knowledge to read is a console that gets misread under pressure.
 *
 * The definition lives in `glossary.ts`, keyed on the visible text, so the same
 * column cannot mean two things on two screens. A heading with no entry renders
 * exactly as a plain `<th>` — adding a column never breaks the table, it just
 * does not gain a tooltip until someone writes one.
 */

import { describe } from "../glossary";

export function Th({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  const definition = describe(children);
  if (!definition) {
    return <th className={className}>{children}</th>;
  }
  return (
    <th className={className}>
      {/* `title` rather than a custom popover: it survives keyboard focus,
          screen readers and a browser with JavaScript disabled, and this is a
          definition rather than an interaction. */}
      <abbr className="th-defined" title={definition}>
        {children}
      </abbr>
    </th>
  );
}
