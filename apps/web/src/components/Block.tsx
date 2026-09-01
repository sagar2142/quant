/**
 * A panel that can take over the window — MASTER_PLAN §12.6.
 *
 * A dense grid answers "how does this name look overall". Reading a chain
 * strike by strike, or finding the session a drawdown started, needs the same
 * block with the whole screen. Both are the same content; only the space
 * differs, so this is a size toggle rather than a second component.
 *
 * **Expanding is not a route.** The block stays mounted and keeps its state —
 * a chart holds its zoom, a chain its expiry — because losing your place is
 * exactly what makes a "detail view" annoying enough that people stop using
 * it. Escape restores, and so does the same control.
 */

import { useEffect, type ReactNode } from "react";

export interface BlockProps {
  title: string;
  /** Stable id, used by the parent to track which block is expanded. */
  id: string;
  expanded: boolean;
  onToggle: (id: string) => void;
  /** Span the full grid width when not expanded. */
  wide?: boolean;
  /** Shown in the header, right of the title. */
  meta?: ReactNode;
  children: ReactNode;
}

export function Block({ title, id, expanded, onToggle, wide, meta, children }: BlockProps) {
  useEffect(() => {
    if (!expanded) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onToggle(id);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [expanded, id, onToggle]);

  const classes = ["block", wide ? "block-wide" : "", expanded ? "block-expanded" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <article className={classes}>
      <header className="panel-header block-head">
        <span>{title}</span>
        {meta}
        <button
          type="button"
          className="block-expand"
          onClick={() => onToggle(id)}
          title={expanded ? "Restore (Esc)" : "Expand to full window"}
          aria-label={expanded ? `Restore ${title}` : `Expand ${title}`}
          aria-pressed={expanded}
        >
          {expanded ? "⤡" : "⤢"}
        </button>
      </header>
      {children}
    </article>
  );
}
