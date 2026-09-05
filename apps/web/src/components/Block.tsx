/**
 * A panel that can take over the window, or leave it — MASTER_PLAN §12.6.
 *
 * Two different sizes of "show me more", because they answer different
 * questions:
 *
 *   expand    give this block the whole console
 *   pop out   give this block its own OS window, for the second monitor
 *
 * **In both cases the block stays mounted.** Expanding is a size toggle and
 * popping out is a portal, so a chart keeps its zoom and a chain keeps its
 * expiry either way, and a popped-out panel still follows the symbol chosen in
 * the main window — it is the same component, not a copy of it.
 */

import { useEffect, type ReactNode } from "react";
import { Icon } from "./Icon";
import { Popout } from "./Popout";

export interface BlockProps {
  title: string;
  /** Stable id, used by the parent to track which block is expanded. */
  id: string;
  expanded: boolean;
  onToggle: (id: string) => void;
  /** The window this block is living in, if it has been popped out. */
  popWindow?: Window | null;
  onPopout?: (id: string, title: string) => void;
  /** Span the full grid width when not expanded. */
  wide?: boolean;
  /** Shown in the header, right of the title. */
  meta?: ReactNode;
  children: ReactNode;
}

export function Block({
  title,
  id,
  expanded,
  onToggle,
  popWindow = null,
  onPopout,
  wide,
  meta,
  children,
}: BlockProps) {
  useEffect(() => {
    if (!expanded) return;
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onToggle(id);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [expanded, id, onToggle]);

  const popped = popWindow != null;

  const controls = (
    <>
      {meta}
      {onPopout && (
        <button
          type="button"
          className={popped ? "block-expand on" : "block-expand"}
          onClick={() => onPopout(id, title)}
          title={popped ? "Bring back into this window" : "Open in a separate window"}
          aria-label={popped ? `Return ${title}` : `Open ${title} in a separate window`}
          aria-pressed={popped}
        >
          <Icon name="popout" />
        </button>
      )}
      {!popped && (
        <button
          type="button"
          className="block-expand"
          onClick={() => onToggle(id)}
          title={expanded ? "Restore (Esc)" : "Expand to full window"}
          aria-label={expanded ? `Restore ${title}` : `Expand ${title}`}
          aria-pressed={expanded}
        >
          <Icon name={expanded ? "collapse" : "expand"} />
        </button>
      )}
    </>
  );

  const body = (
    <article className={popped ? "block block-popped" : classesFor(wide, expanded)}>
      <header className="panel-header block-head">
        <span>{title}</span>
        {controls}
      </header>
      {children}
    </article>
  );

  if (popped && onPopout && popWindow) {
    return (
      <>
        {/* The block keeps its slot in the grid so the layout does not reflow
            around a hole, and the placeholder says where the panel went — a
            block that simply vanished would read as a bug. */}
        <article className={classesFor(wide, false)}>
          <header className="panel-header block-head">
            <span>{title}</span>
            <button
              type="button"
              className="block-expand on"
              onClick={() => onPopout(id, title)}
              title="Bring back into this window"
            >
              <Icon name="popout" />
            </button>
          </header>
          <p className="popout-note">
            Open in a separate window
            <button type="button" className="ghost" onClick={() => onPopout(id, title)}>
              Return
            </button>
          </p>
        </article>

        <Popout title={title} target={popWindow} onClose={() => onPopout(id, title)}>
          {body}
        </Popout>
      </>
    );
  }

  return body;
}

function classesFor(wide: boolean | undefined, expanded: boolean): string {
  return ["block", wide ? "block-wide" : "", expanded ? "block-expanded" : ""]
    .filter(Boolean)
    .join(" ");
}
