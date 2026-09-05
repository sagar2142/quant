/**
 * The desk: whole features in their own windows — MASTER_PLAN §12.9.
 *
 * A quant desk is several monitors showing different things at once — this
 * name's chain here, that name's drawdown there, the order book on a third
 * screen. That is not one console made bigger; it is many consoles, each
 * pointed somewhere else, open together.
 *
 * Three things make that work rather than merely possible:
 *
 * **Instance ids, not feature ids.** Windows are keyed per opening, so the
 * same feature can be detached as many times as you like. Keying on the
 * feature is exactly what limits you to one Research window.
 *
 * **Each window owns its symbol by default.** Three Security windows all
 * following the main console would show one name three times, which is a wider
 * view of one thing rather than a view of three.
 *
 * **Link groups, for when you do want them to move together.** A window in
 * group A follows every other window in group A: change the symbol in the
 * chart and the chain beside it follows. Ungrouped windows stay independent.
 * This is the pattern every real terminal settles on, because both behaviours
 * are wanted and neither is right all the time.
 *
 * **The desk is remembered, and reopened on request rather than automatically.**
 * A browser only permits `window.open` while it is handling a user gesture, so
 * restoring six windows on page load is not something the page may do by
 * itself — it can only offer, and the operator clicks once.
 */

import { useCallback, useEffect, useState } from "react";
import { CommandBar } from "./CommandBar";
import { Icon } from "./Icon";
import { Popout } from "./Popout";
import { ResearchDesk, type ResearchTab } from "./ResearchDesk";
import { SystemPanel } from "./SystemPanel";
import type { ContractTerms } from "./Ticket";
import { TradeDesk } from "./TradeDesk";
import { Workspace as ChartWorkspace } from "./Workspace";

export type DetachedKind = "market" | "research" | "trade" | "system";

export const DETACHED_LABELS: Record<DetachedKind, string> = {
  market: "Charts",
  research: "Security",
  trade: "Trade",
  system: "System",
};

/** Colour-coded link groups, plus "independent". */
export const LINK_GROUPS = ["", "A", "B", "C"] as const;
export type LinkGroup = (typeof LINK_GROUPS)[number];

const DESK_KEY = "neutron.desk.v1";

/** What is remembered about a window — never the `Window` itself. */
interface SavedWindow {
  kind: DetachedKind;
  symbol: string;
  venue: string;
  link: LinkGroup;
}

export interface DetachedInstance extends SavedWindow {
  /** Unique per opening, so the same feature can be detached many times. */
  id: string;
  target: Window;
}

function loadDesk(): SavedWindow[] {
  try {
    const raw = localStorage.getItem(DESK_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? (parsed as SavedWindow[]) : [];
  } catch {
    return [];
  }
}

interface BodyProps {
  instance: DetachedInstance;
  /** The symbol its link group is on, when it belongs to one. */
  linkedSymbol: string | null;
  onLinkedSymbol: (link: LinkGroup, symbol: string) => void;
  onLink: (id: string, link: LinkGroup) => void;
  onSymbol: (id: string, symbol: string) => void;
}

/** One detached feature, with state of its own. */
function DetachedBody({ instance, linkedSymbol, onLinkedSymbol, onLink, onSymbol }: BodyProps) {
  const [ownSymbol, setOwnSymbol] = useState(instance.symbol);
  const [venue, setVenue] = useState(instance.venue);
  const [tab, setTab] = useState<ResearchTab>("security");
  const [contract, setContract] = useState<ContractTerms | null>(null);

  //: A linked window shows its group's symbol; an independent one shows its
  //: own. Reading through rather than copying, so a group change lands here
  //: without a round trip.
  const symbol = instance.link ? (linkedSymbol ?? ownSymbol) : ownSymbol;

  const pick = useCallback(
    (next: string) => {
      setOwnSymbol(next);
      setContract(null);
      onSymbol(instance.id, next);
      if (instance.link) onLinkedSymbol(instance.link, next);
    },
    [instance.id, instance.link, onLinkedSymbol, onSymbol],
  );

  return (
    <div className="detached">
      <CommandBar symbol={symbol} venue={venue} onSymbol={pick} onVenue={setVenue}>
        <div className="link-picker" title="Link group: windows in the same group follow each other">
          {LINK_GROUPS.map((group) => (
            <button
              key={group || "none"}
              type="button"
              className={
                instance.link === group ? `link-chip link-${group || "none"} on` : `link-chip link-${group || "none"}`
              }
              onClick={() => onLink(instance.id, group)}
              title={group ? `Follow group ${group}` : "Independent"}
            >
              {group || "—"}
            </button>
          ))}
        </div>
        <span className="detached-kind">{DETACHED_LABELS[instance.kind]}</span>
      </CommandBar>

      <main className="detached-stage">
        {instance.kind === "market" && <ChartWorkspace onTrade={pick} />}

        {instance.kind === "research" && (
          <ResearchDesk
            tab={tab}
            onTab={setTab}
            symbol={symbol}
            venue={venue}
            onSymbol={pick}
            onVenue={setVenue}
            onTrade={pick}
            onTradeContract={(picked, terms) => {
              pick(picked);
              setContract(terms);
            }}
          />
        )}

        {instance.kind === "trade" && (
          <TradeDesk
            symbol={symbol}
            onSymbolChange={pick}
            contract={contract}
            onContractChange={setContract}
          />
        )}

        {instance.kind === "system" && <SystemPanel />}
      </main>

      <footer className="statusbar">
        <span className="mono">{venue}</span>
        <span className="muted">{symbol || "No security"}</span>
        {instance.link && <span className={`link-chip link-${instance.link} on`}>{instance.link}</span>}
        <span className="status-right muted">Detached window</span>
      </footer>
    </div>
  );
}

export interface DetachedProps {
  state: DetachedState;
}

/** Renders every detached window. */
export function Detached({ state }: DetachedProps) {
  return (
    <>
      {state.instances.map((instance) => (
        <Popout
          key={instance.id}
          title={`${DETACHED_LABELS[instance.kind]}${instance.symbol ? ` · ${instance.symbol}` : ""}`}
          target={instance.target}
          onClose={() => state.close(instance.id)}
        >
          <DetachedBody
            instance={instance}
            linkedSymbol={instance.link ? (state.groups[instance.link] ?? null) : null}
            onLinkedSymbol={state.setGroupSymbol}
            onLink={state.setLink}
            onSymbol={state.setSymbol}
          />
        </Popout>
      ))}
    </>
  );
}

export interface DetachedState {
  instances: DetachedInstance[];
  /** Symbol each link group is currently on. */
  groups: Partial<Record<LinkGroup, string>>;
  /** How many windows the last saved desk had, for the restore prompt. */
  savedCount: number;
  open: (kind: DetachedKind, symbol: string, venue: string, link?: LinkGroup) => void;
  close: (id: string) => void;
  closeAll: () => void;
  restore: () => void;
  setLink: (id: string, link: LinkGroup) => void;
  setSymbol: (id: string, symbol: string) => void;
  setGroupSymbol: (link: LinkGroup, symbol: string) => void;
  blocked: string;
  clearBlocked: () => void;
}

export function useDetached(): DetachedState {
  const [instances, setInstances] = useState<DetachedInstance[]>([]);
  const [groups, setGroups] = useState<Partial<Record<LinkGroup, string>>>({});
  const [saved, setSaved] = useState<SavedWindow[]>(loadDesk);
  const [blocked, setBlocked] = useState("");

  //: The desk is saved whenever it changes, so closing the console and coming
  //: back offers the same layout. Only the description is stored — a `Window`
  //: cannot be serialised, and would be meaningless in a later session anyway.
  useEffect(() => {
    if (instances.length === 0) return;
    try {
      localStorage.setItem(
        DESK_KEY,
        JSON.stringify(
          instances.map(({ kind, symbol, venue, link }) => ({ kind, symbol, venue, link })),
        ),
      );
      setSaved(instances.map(({ kind, symbol, venue, link }) => ({ kind, symbol, venue, link })));
    } catch {
      // A desk that cannot be remembered still has to work today.
    }
  }, [instances]);

  const openWindow = useCallback(
    (kind: DetachedKind, symbol: string, venue: string, link: LinkGroup): boolean => {
      const id = `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const target = window.open(
        "",
        id,
        "width=1280,height=860,menubar=no,toolbar=no,location=no,status=no",
      );
      if (!target) return false;
      setInstances((current) => [...current, { id, kind, target, symbol, venue, link }]);
      return true;
    },
    [],
  );

  const open = useCallback(
    (kind: DetachedKind, symbol: string, venue: string, link: LinkGroup = "") => {
      setBlocked("");
      if (!openWindow(kind, symbol, venue, link)) {
        setBlocked(
          `${DETACHED_LABELS[kind]} could not open: the browser blocked the pop-up. ` +
            "Allow pop-ups for this site and try again.",
        );
      }
    },
    [openWindow],
  );

  const restore = useCallback(() => {
    setBlocked("");
    let refused = 0;
    for (const entry of saved) {
      if (!openWindow(entry.kind, entry.symbol, entry.venue, entry.link)) refused += 1;
    }
    if (refused) {
      setBlocked(
        `${refused} window${refused === 1 ? "" : "s"} were blocked. Allow pop-ups for this site, ` +
          "then restore again.",
      );
    }
  }, [saved, openWindow]);

  const close = useCallback((id: string) => {
    setInstances((current) => {
      current.find((instance) => instance.id === id)?.target.close();
      return current.filter((instance) => instance.id !== id);
    });
  }, []);

  const closeAll = useCallback(() => {
    setInstances((current) => {
      current.forEach((instance) => instance.target.close());
      return [];
    });
  }, []);

  const setLink = useCallback((id: string, link: LinkGroup) => {
    setInstances((current) =>
      current.map((instance) => (instance.id === id ? { ...instance, link } : instance)),
    );
  }, []);

  const setSymbol = useCallback((id: string, symbol: string) => {
    setInstances((current) =>
      current.map((instance) => (instance.id === id ? { ...instance, symbol } : instance)),
    );
  }, []);

  const setGroupSymbol = useCallback((link: LinkGroup, symbol: string) => {
    if (!link) return;
    setGroups((current) => ({ ...current, [link]: symbol }));
  }, []);

  const clearBlocked = useCallback(() => setBlocked(""), []);

  return {
    instances,
    groups,
    savedCount: saved.length,
    open,
    close,
    closeAll,
    restore,
    setLink,
    setSymbol,
    setGroupSymbol,
    blocked,
    clearBlocked,
  };
}

/** The desk roster: what is open, and how to open more. */
export function DeskBar({
  state,
  symbol,
  venue,
}: {
  state: DetachedState;
  symbol: string;
  venue: string;
}) {
  const kinds: DetachedKind[] = ["market", "research", "trade", "system"];
  return (
    <div className="deskbar">
      <span className="charts-label">Desk</span>
      {kinds.map((kind) => (
        <button
          key={kind}
          type="button"
          className="ghost"
          onClick={() => state.open(kind, symbol, venue)}
          title={`Open another ${DETACHED_LABELS[kind]} window`}
        >
          <Icon name="popout" /> {DETACHED_LABELS[kind]}
        </button>
      ))}

      {state.instances.length > 0 && (
        <>
          <span className="muted">
            {state.instances.length} open
          </span>
          <button type="button" className="ghost danger" onClick={state.closeAll}>
            close all
          </button>
        </>
      )}

      {state.instances.length === 0 && state.savedCount > 0 && (
        <button type="button" className="ghost on" onClick={state.restore}>
          restore {state.savedCount} window{state.savedCount === 1 ? "" : "s"}
        </button>
      )}

      {state.blocked && (
        <span className="popout-blocked" onClick={state.clearBlocked}>
          {state.blocked}
        </span>
      )}
    </div>
  );
}
