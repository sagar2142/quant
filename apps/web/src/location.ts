/**
 * Where you are, in the URL — MASTER_PLAN §12.9.
 *
 * The console kept its location in component state, so a refresh returned you
 * to the default workspace with the security you were looking at forgotten.
 * On a screen people leave open all day and reload whenever something looks
 * stale, that is not a small annoyance: it loses the thing you had set up.
 *
 * **The hash, not the path.** A hash route needs no server rewrite rule, which
 * matters because this is served by Vite in development and by a static bundle
 * in production, and neither should have to learn the console's routes.
 *
 * **The symbol and venue travel in the URL too.** That makes a view shareable
 * and, more usefully here, makes a detached window's address describe exactly
 * what it shows: `#/research/security?symbol=TCS&venue=NSE` is a complete
 * instruction for reconstructing it.
 */

export type Workspace = "market" | "research" | "trade" | "system" | "tutorial";
export type AccountView = "none" | "auth" | "settings";

export interface Location {
  workspace: Workspace;
  /** Which tab inside Research. */
  tab: string;
  symbol: string;
  venue: string;
  /** The account surface, when one is covering the console. */
  account: AccountView;
  /** Which settings section, when `account` is "settings". */
  section: string;
}

const WORKSPACES: Workspace[] = ["market", "research", "trade", "system", "tutorial"];

export const DEFAULT_LOCATION: Location = {
  workspace: "market",
  tab: "security",
  symbol: "",
  venue: "NSE",
  account: "none",
  section: "profile",
};

/** Parse `#/research/security?symbol=TCS&venue=NSE` into a location. */
export function readLocation(hash: string = window.location.hash): Location {
  const raw = hash.replace(/^#\/?/, "");
  if (!raw) return { ...DEFAULT_LOCATION };

  const [path, query = ""] = raw.split("?");
  const parts = (path ?? "").split("/").filter(Boolean);
  const params = new URLSearchParams(query);

  const head = parts[0] ?? "";
  const venue = (params.get("venue") ?? "NSE").toUpperCase();
  const symbol = (params.get("symbol") ?? "").toUpperCase();

  // The account surfaces are routes of their own, so signing in or opening
  // settings survives a refresh rather than dropping you back on the console.
  if (head === "settings") {
    return {
      ...DEFAULT_LOCATION,
      account: "settings",
      section: parts[1] ?? "profile",
      symbol,
      venue,
    };
  }
  if (head === "signin") {
    return { ...DEFAULT_LOCATION, account: "auth", symbol, venue };
  }

  const workspace = WORKSPACES.includes(head as Workspace) ? (head as Workspace) : "market";
  return {
    workspace,
    // Only Research has tabs, so a second path segment anywhere else is
    // meaningless and is dropped. Without this an unrecognised address like
    // `#/garbage/nonsense` normalised to `market` while quietly keeping
    // `nonsense` as the tab — the same URL then read back differently than it
    // was written, which is the sort of thing that makes a back button behave
    // unpredictably.
    tab: workspace === "research" ? (parts[1] ?? "security") : "security",
    symbol,
    venue,
    account: "none",
    section: "profile",
  };
}

/** Render a location back into a hash. */
export function writeLocation(location: Location): string {
  const params = new URLSearchParams();
  if (location.symbol) params.set("symbol", location.symbol);
  // Only when it is not the default, so an ordinary URL stays short enough to
  // read at a glance.
  if (location.venue && location.venue !== "NSE") params.set("venue", location.venue);
  const query = params.toString();

  let path: string;
  if (location.account === "settings") path = `settings/${location.section}`;
  else if (location.account === "auth") path = "signin";
  else if (location.workspace === "research") path = `research/${location.tab}`;
  else path = location.workspace;

  return `#/${path}${query ? `?${query}` : ""}`;
}
