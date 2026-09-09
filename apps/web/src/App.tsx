/**
 * Ops console shell — MASTER_PLAN §12.6, §12.9.
 *
 * **Four workspaces, not fourteen screens.** The previous shell listed every
 * module as its own nav entry, which put four permanently empty screens in
 * front of the operator: Overview, Positions, Blotter and Reconcile all read
 * the paper-trading state that was deliberately removed, so `/book` returns an
 * empty book, `/equity` an empty curve and `/reconciliation` `checked: false`.
 * A screen that is always blank is worse than one that is absent — it teaches
 * you to stop looking at screens.
 *
 * What remains is organised by what you are doing rather than by which module
 * implements it:
 *
 *     MARKET     watch      charts and watchlist
 *     RESEARCH   evaluate   security, screener, cross-section, factors, risk
 *     TRADE      act        the ticket and the venue's own book, together
 *     SYSTEM     verify     data coverage, gates, limits
 *
 * **The symbol and venue belong to the shell.** Every screen used to ask for
 * them separately, so moving between a chart, its analysis and a ticket meant
 * typing the same ticker three times and risking three different answers to
 * "which security am I looking at".
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { AccountPanel, useAccount } from "./components/Account";
import type { AccountAction } from "./components/AccountMenu";
import { AccountSettings, type SettingsSection } from "./components/AccountSettings";
import { AuthPage } from "./components/AuthPage";
import { Logo } from "./components/Logo";
import { CommandBar } from "./components/CommandBar";
import {
  Detached,
  DeskBar,
  useDetached,
  type DetachedKind,
} from "./components/Detached";
import { Icon, type IconName } from "./components/Icon";
import { ResearchDesk, type ResearchTab } from "./components/ResearchDesk";
import { SystemPanel } from "./components/SystemPanel";
import type { ContractTerms } from "./components/Ticket";
import { TradeDesk } from "./components/TradeDesk";
import { Tutorial } from "./components/Tutorial";
import { VitalsBar, type Vitals } from "./components/VitalsBar";
import { Workspace as ChartWorkspace } from "./components/Workspace";
import { DEFAULT_LOCATION, readLocation, writeLocation, type Location } from "./location";
import "./tokens.css";
import "./shell.css";
import "./layout.css";
import "./terminal.css";

//: Marks that the guide has been shown, so it opens once and is a reference
//: thereafter.
const SEEN_TUTORIAL = "neutron.tutorial.seen";

//: Marks that the operator chose to work without an account.
const SKIPPED_AUTH = "neutron.auth.skipped";

type Workspace = "market" | "research" | "trade" | "system" | "tutorial";

const WORKSPACES: {
  id: Workspace;
  label: string;
  icon: IconName;
  key: string;
  hint: string;
}[] = [
  { id: "market", label: "Market", icon: "market", key: "1", hint: "Charts and watchlist" },
  {
    id: "research",
    label: "Research",
    icon: "research",
    key: "2",
    hint: "Evaluate a security or the universe",
  },
  { id: "trade", label: "Trade", icon: "trade", key: "3", hint: "Order entry and the venue book" },
  {
    id: "system",
    label: "System",
    icon: "system",
    key: "4",
    hint: "Data coverage, gates and limits",
  },
  { id: "tutorial", label: "Guide", icon: "guide", key: "5", hint: "How the research loop works" },
];

export interface Position {
  instrumentId: string;
  symbol: string;
  quantity: number;
  averagePrice: number;
  lastPrice: number;
  unrealisedPnl: number;
  weightPct: number;
  cluster: string;
}

export interface Trade {
  eventTime: string;
  symbol: string;
  side: "BUY" | "SELL";
  quantity: number;
  price: number;
  costs: number;
  state: string;
}

export interface RiskRow {
  name: string;
  /**
   * Null for limits checked per order, which a book at rest has no value for.
   * Rendered as an em dash. This was previously hardcoded to 0, which reads as
   * "measured, nothing used" — the opposite of what it meant.
   */
  observed: number | null;
  threshold: number;
  /** Null travels with a null observation: unmeasured is not passing. */
  passed: boolean | null;
}

/** What the last reconciliation actually established, if it ran at all. */
export interface ReconciliationStatus {
  checked: boolean;
  halted: boolean;
  haltReason: string;
  cycles: number;
}

export interface Break {
  instrumentId: string;
  kind: string;
  internal: number;
  broker: number;
}

export interface ConsoleState {
  vitals: Vitals;
  /**
   * Equity at the close of each completed cycle, oldest first. Empty until a
   * broker is connected — the paper source that used to fill it was removed,
   * and inventing a curve to make the screen look alive is exactly the failure
   * the rest of this system exists to avoid.
   */
  equity: number[];
  positions: Position[];
  trades: Trade[];
  risk: RiskRow[];
  breaks: Break[];
  reconciliation: ReconciliationStatus;
  environment: "dev" | "paper" | "live";
  gitSha: string;
  latencyMs: number;
}

export function App({
  state,
  onKill,
}: {
  state: ConsoleState;
  onKill: (reason: string) => void;
}) {
  // First visit opens the guide. What a new operator lacks is not button
  // locations but the order of operations, and someone who starts at the
  // backtester reads a rising curve as a discovery rather than as the first of
  // twelve questions.
  //: Read once from the URL. A refresh, a bookmark and a back button all
  //: arrive the same way, so the console reconstructs itself from the address
  //: rather than resetting to a default nobody asked for.
  const initial = useMemo<Location>(() => readLocation(), []);

  const [workspace, setWorkspace] = useState<Workspace>(() => {
    if (window.location.hash) return initial.workspace;
    try {
      return window.localStorage.getItem(SEEN_TUTORIAL) ? "market" : "tutorial";
    } catch {
      // Private browsing and some hardened configurations throw on access.
      // Losing the preference is harmless; failing to render is not.
      return "market";
    }
  });
  const [tab, setTab] = useState<ResearchTab>(initial.tab as ResearchTab);

  useEffect(() => {
    if (workspace !== "tutorial") return;
    try {
      window.localStorage.setItem(SEEN_TUTORIAL, "1");
    } catch {
      /* see above */
    }
  }, [workspace]);

  //: Symbol, venue and contract live here so every workspace is looking at the
  //: same security. Held in the shell rather than per screen, because the
  //: alternative is a chart on one name and a ticket on another with nothing
  //: on screen saying so.
  const [symbol, setSymbol] = useState(initial.symbol);
  const [venue, setVenue] = useState(initial.venue);
  const [contract, setContract] = useState<ContractTerms | null>(null);

  //: Whole features living in their own OS windows. Several of the same
  //: feature are allowed on purpose — three Security windows on three symbols
  //: is the layout a multi-monitor desk actually wants.
  const detached = useDetached();

  //: Who is signed in, and which account surface is showing. `null` means the
  //: console itself; the auth and settings pages take over the whole window
  //: because neither belongs in a dropdown.
  const {
    status: accountStatus,
    resolved: accountResolved,
    refresh: refreshAccount,
    signOut,
  } = useAccount();
  const [accountView, setAccountView] = useState<"none" | "auth" | "settings">(initial.account);
  const [settingsSection, setSettingsSection] = useState<SettingsSection>(
    initial.section as SettingsSection,
  );
  //: Set once the operator has chosen to work without an account, so the auth
  //: page does not reappear on every status refresh.
  const [skippedAuth, setSkippedAuth] = useState(() => {
    // Remembered, so "continue without an account" is answered once rather
    // than on every reload.
    try {
      return window.localStorage.getItem(SKIPPED_AUTH) === "1";
    } catch {
      return false;
    }
  });

  const onAccountAction = useCallback(
    (action: AccountAction) => {
      if (action === "signout") {
        void signOut();
        return;
      }
      if (action === "signin") {
        setAccountView("auth");
        return;
      }
      const section: SettingsSection =
        action === "broker" ? "broker" : action === "preferences" ? "preferences" : "profile";
      setSettingsSection(section);
      setAccountView("settings");
    },
    [signOut],
  );

  //: Preferences arrive from the signed-in account. Applied rather than
  //: merged into component defaults, so signing in on a second machine gives
  //: the same console rather than a different one that happens to look alike.
  const applySettings = useCallback(
    (settings: Record<string, unknown>) => {
      // The URL wins. A link that names a venue is an instruction; a saved
      // preference is a default, and a default must not overrule the thing
      // the operator just opened.
      if (initial.venue !== DEFAULT_LOCATION.venue) return;
      const preferred = settings.default_venue;
      if (preferred === "NSE" || preferred === "BSE") setVenue(preferred);
    },
    [initial.venue],
  );

  useEffect(() => {
    if (accountStatus?.account) applySettings(accountStatus.account.settings);
  }, [accountStatus, applySettings]);

  const pickSymbol = useCallback((next: string) => {
    setSymbol(next);
    // A new name is not the old contract. Leaving it set would aim the ticket
    // at an option on a security you are no longer looking at.
    setContract(null);
  }, []);

  // Keyboard-first: a digit per workspace. The mouse is optional (§12.8).
  const handleKey = useCallback((event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const match = WORKSPACES.find((w) => w.key === event.key);
    if (match) setWorkspace(match.id);
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [handleKey]);

  //: Keep the address in step with the console. `replaceState` rather than a
  //: hash assignment, so moving between workspaces does not fill the back
  //: button with a dozen entries nobody wants to walk through.
  useEffect(() => {
    const next = writeLocation({
      workspace,
      tab,
      symbol,
      venue,
      account: accountView,
      section: settingsSection,
    });
    if (next !== window.location.hash) {
      window.history.replaceState(null, "", next);
    }
  }, [workspace, tab, symbol, venue, accountView, settingsSection]);

  //: And the other direction, for the back button and a pasted link.
  useEffect(() => {
    const onHashChange = () => {
      const there = readLocation();
      setWorkspace(there.workspace);
      setTab(there.tab as ResearchTab);
      setSymbol(there.symbol);
      setVenue(there.venue);
      setAccountView(there.account);
      setSettingsSection(there.section as SettingsSection);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  //: Until the status comes back there is no honest screen to show. Rendering
  //: the console and replacing it a moment later is the worst of the options:
  //: it shows a workspace the operator may not be entitled to, and the swap
  //: reads as the app having crashed and recovered. A cold Atlas connection
  //: takes seconds, so this is not a hypothetical frame.
  if (!accountResolved && accountView !== "auth") {
    return (
      <div className="authpage">
        <div className="auth-card auth-loading">
          <Logo size={38} />
          <p className="muted">Checking your session…</p>
        </div>
      </div>
    );
  }

  //: Shown before the console on a fresh install, and whenever the operator
  //: asks for it. Not a lock — the page says so itself — but the right first
  //: screen when no account exists yet.
  const needsAuth =
    accountView === "auth" ||
    (!skippedAuth &&
      accountStatus !== null &&
      accountStatus.accounts_available &&
      !accountStatus.signed_in);

  if (needsAuth) {
    return (
      <AuthPage
        userCount={accountStatus?.user_count ?? 0}
        apiTokenConfigured={accountStatus?.api_token_configured ?? false}
        accountsAvailable={accountStatus?.accounts_available ?? false}
        backend={accountStatus?.backend ?? "postgres"}
        onAuthenticated={() => {
          setAccountView("none");
          void refreshAccount();
        }}
        onSkip={() => {
          setSkippedAuth(true);
          try {
            window.localStorage.setItem(SKIPPED_AUTH, "1");
          } catch {
            /* the choice is a convenience; failing to render is not */
          }
          setAccountView("none");
        }}
      />
    );
  }

  if (accountView === "settings") {
    return (
      <AccountSettings
        account={accountStatus?.account ?? null}
        section={settingsSection}
        onSection={setSettingsSection}
        onClose={() => setAccountView("none")}
        onChanged={() => void refreshAccount()}
      />
    );
  }

  return (
    <div className="shell">
      <CommandBar symbol={symbol} venue={venue} onSymbol={pickSymbol} onVenue={setVenue}>
        <VitalsBar vitals={state.vitals} onKill={onKill} />
      </CommandBar>

      <div className="shell-body">
        <nav className="rail" aria-label="Workspaces">
          {WORKSPACES.map((item) => (
            <div key={item.id} className="rail-slot">
              <button
                type="button"
                className={workspace === item.id ? "rail-item on" : "rail-item"}
                onClick={() => setWorkspace(item.id)}
                title={`${item.hint}  (${item.key})`}
                aria-current={workspace === item.id ? "page" : undefined}
              >
                <span className="rail-icon">
                  <Icon name={item.icon} size={1.15} />
                </span>
                <span className="rail-label">{item.label}</span>
              </button>
              {item.id !== "tutorial" && (
                <button
                  type="button"
                  className="rail-detach"
                  onClick={() => detached.open(item.id as DetachedKind, symbol, venue)}
                  title={`Open ${item.label} in a new window — as many as you like`}
                  aria-label={`Open ${item.label} in a new window`}
                >
                  <Icon name="popout" />
                </button>
              )}
            </div>
          ))}
          <AccountPanel status={accountStatus} onAction={onAccountAction} />
        </nav>

        <main className="stage">
          <DeskBar state={detached} symbol={symbol} venue={venue} />

          {workspace === "market" && (
            <ChartWorkspace
              onTrade={(picked) => {
                pickSymbol(picked);
                setWorkspace("trade");
              }}
            />
          )}

          {workspace === "research" && (
            <ResearchDesk
              tab={tab}
              onTab={setTab}
              symbol={symbol}
              venue={venue}
              onSymbol={pickSymbol}
              onVenue={setVenue}
              onTrade={(picked) => {
                pickSymbol(picked);
                setWorkspace("trade");
              }}
              onTradeContract={(picked, terms) => {
                setSymbol(picked);
                setContract(terms);
                setWorkspace("trade");
              }}
            />
          )}

          {workspace === "trade" && (
            <TradeDesk
              symbol={symbol}
              onSymbolChange={pickSymbol}
              contract={contract}
              onContractChange={setContract}
            />
          )}

          {workspace === "system" && <SystemPanel />}

          {workspace === "tutorial" && (
            <div className="guide">
              <Tutorial onDismiss={() => setWorkspace("market")} />
            </div>
          )}
        </main>
      </div>

      <footer className="statusbar">
        <span className="mono">{venue}</span>
        <span className="muted">{symbol || "No security"}</span>
        <span className="muted">{WORKSPACES.find((w) => w.id === workspace)?.hint ?? ""}</span>
        <span className="status-right muted">⌃K Search · 1–5 Workspaces · ⧉ Detach</span>
      </footer>

      <Detached state={detached} />
    </div>
  );
}
