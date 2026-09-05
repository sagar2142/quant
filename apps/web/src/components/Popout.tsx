/**
 * Panels in their own OS window — MASTER_PLAN §12.9.
 *
 * A trading desk is several monitors. Expanding a block to fill the browser
 * window answers "show me this bigger"; it does not answer "put the chain on
 * the second screen while I watch the chart on the first", which is the layout
 * the work actually wants.
 *
 * **The window is opened in the click handler, not in an effect.** This is the
 * whole reason the first version did nothing: browsers permit `window.open`
 * only while a user gesture is being handled, and a React effect runs after
 * render, outside that window. The popup was blocked silently — no error, no
 * dialog, just a button that appeared to do nothing. `usePopouts.toggle` now
 * opens it synchronously and hands the live `Window` to the portal.
 *
 * **The panel is moved, not re-rendered.** React portals keep the component
 * mounted in the original tree, so it holds its state, its subscriptions and
 * its fetches. A chart keeps its zoom and a chain keeps its expiry when it pops
 * out, and the parent keeps driving it: change the symbol in the main window
 * and the popped-out panel follows, because it is the same component.
 *
 * **Stylesheets are copied and kept in sync.** A new window starts with an
 * empty document, so nothing the console has styled applies. In a production
 * build that is a handful of `<link>` tags; under Vite it is injected `<style>`
 * elements that change on every hot reload, which is why this observes the head
 * rather than copying once and hoping.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/** Copy every stylesheet from the host document into a popped-out one. */
function adoptStyles(target: Document): () => void {
  const copy = () => {
    // Rebuilt rather than diffed. There are a few dozen nodes at most, and a
    // stale rule surviving a hot reload is a panel that looks wrong for
    // reasons nobody can find.
    target.querySelectorAll("[data-neutron-style]").forEach((node) => node.remove());
    document.querySelectorAll('style, link[rel="stylesheet"]').forEach((node) => {
      const clone = node.cloneNode(true) as HTMLElement;
      clone.setAttribute("data-neutron-style", "");
      target.head.appendChild(clone);
    });
  };

  copy();
  const observer = new MutationObserver(copy);
  observer.observe(document.head, { childList: true, subtree: true });
  return () => observer.disconnect();
}

/** Prepare a freshly opened window to host a panel. */
function dress(child: Window, title: string): HTMLElement {
  child.document.title = `${title} · Neutron`;
  // The theme lives on the root element and the background on the body; a
  // fresh document has neither, so the panel would open on white.
  child.document.documentElement.setAttribute("data-theme", "dark");
  child.document.body.style.margin = "0";
  child.document.body.style.background = "#0b0d10";

  const host = child.document.createElement("div");
  host.className = "popout-root";
  child.document.body.appendChild(host);
  return host;
}

export interface PopoutProps {
  title: string;
  /** The window opened by the click handler. */
  target: Window;
  onClose: () => void;
  children: React.ReactNode;
}

export function Popout({ title, target, onClose, children }: PopoutProps) {
  const [mount, setMount] = useState<HTMLElement | null>(null);
  //: `onClose` comes from a parent that re-renders; holding it in a ref keeps
  //: the effect from tearing the window down and rebuilding it on every
  //: render, which would flash the panel and lose its scroll position.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (target.closed) {
      closeRef.current();
      return;
    }

    const host = dress(target, title);
    const stopWatching = adoptStyles(target.document);

    const handleUnload = () => closeRef.current();
    target.addEventListener("beforeunload", handleUnload);

    // Closing the main window must not leave orphans behind.
    const closeChild = () => target.close();
    window.addEventListener("beforeunload", closeChild);

    setMount(host);
    target.focus();

    return () => {
      // **The window is not closed here.** React StrictMode mounts, unmounts
      // and remounts every effect in development, so closing on cleanup shut
      // the window microseconds after opening it — the panel flashed and
      // vanished, which looked like a broken button rather than a lifecycle
      // problem. The window belongs to `usePopouts`, which opened it; this
      // effect only borrows it to portal into.
      stopWatching();
      target.removeEventListener("beforeunload", handleUnload);
      window.removeEventListener("beforeunload", closeChild);
      host.remove();
      setMount(null);
    };
  }, [target, title]);

  return mount ? createPortal(children, mount) : null;
}

export interface PopoutState {
  /** Live windows, keyed by panel id. */
  windows: Map<string, Window>;
  toggle: (id: string, title?: string) => void;
  close: (id: string) => void;
  /** Set when the browser refused to open a window, so the UI can say so. */
  blocked: string;
  clearBlocked: () => void;
}

/**
 * Tracks which panels are in their own window.
 *
 * `toggle` opens the window itself, synchronously, because it is called
 * straight from a click handler and that is the only context in which a
 * browser will allow it.
 */
export function usePopouts(): PopoutState {
  const [windows, setWindows] = useState<Map<string, Window>>(new Map());
  const [blocked, setBlocked] = useState("");

  //: The hook opened these, so the hook closes them — on unmount, and only on
  //: a real one. Held in a ref so the cleanup sees the current set without
  //: re-running every time a window is added, which would close the window it
  //: had just opened.
  const live = useRef(windows);
  live.current = windows;
  useEffect(
    () => () => {
      live.current.forEach((child) => child.close());
    },
    [],
  );

  const close = useCallback((id: string) => {
    setWindows((current) => {
      if (!current.has(id)) return current;
      const next = new Map(current);
      next.delete(id);
      return next;
    });
  }, []);

  const toggle = useCallback(
    (id: string, title = "Panel") => {
      setBlocked("");
      const existing = windows.get(id);
      if (existing) {
        existing.close();
        close(id);
        return;
      }

      const child = window.open(
        "",
        `neutron-${id}`,
        "width=1100,height=800,menubar=no,toolbar=no,location=no,status=no",
      );
      if (!child) {
        // Say so. A control that silently does nothing is indistinguishable
        // from a broken one, and this is the failure people actually hit.
        setBlocked(
          `${title} could not open: the browser blocked the pop-up. Allow pop-ups for this site and try again.`,
        );
        return;
      }
      setWindows((current) => new Map(current).set(id, child));
    },
    [windows, close],
  );

  const clearBlocked = useCallback(() => setBlocked(""), []);

  return { windows, toggle, close, blocked, clearBlocked };
}
