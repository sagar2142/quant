/**
 * The account control in the rail — MASTER_PLAN §13.7.
 *
 * Just the button and its menu. The work moved out: `AuthPage` is where you
 * sign in, `AccountSettings` is where preferences and broker keys live, and
 * `AccountMenu` decides which of those you are going to. The previous version
 * was one popover trying to be all three, which put a password field and a
 * broker key field in a 280px box hanging off the bottom of a sidebar.
 */

import { useCallback, useEffect, useState } from "react";
import { AccountMenu, type AccountAction } from "./AccountMenu";
import { Icon } from "./Icon";

export interface Account {
  user_id: string;
  email: string;
  display_name: string;
  settings: Record<string, unknown>;
  created_at: string;
  last_login_at: string | null;
}

export interface AccountStatus {
  signed_in: boolean;
  account: Account | null;
  accounts_available: boolean;
  api_token_configured: boolean;
  user_count: number;
  /** Which database holds accounts: "mongodb" or "postgres". */
  backend: string;
}

export interface AccountPanelProps {
  status: AccountStatus | null;
  onAction: (action: AccountAction) => void;
}

export function AccountPanel({ status, onAction }: AccountPanelProps) {
  const [open, setOpen] = useState(false);

  if (!status) {
    return (
      <div className="account">
        <span className="account-offline">offline</span>
      </div>
    );
  }

  const account = status.account;
  const initial = (account?.display_name || account?.email || "?").slice(0, 1).toUpperCase();

  return (
    <div className="account">
      <button
        type="button"
        className="account-button"
        onClick={() => setOpen((current) => !current)}
        title={account ? `Signed in as ${account.email}` : "Sign in or create an account"}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="account-avatar">{account ? initial : <Icon name="guide" />}</span>
        <span className="account-label">
          {account ? account.display_name || "Account" : "Sign in"}
        </span>
      </button>

      {open && (
        <AccountMenu
          account={account}
          onDismiss={() => setOpen(false)}
          onAction={(action) => {
            setOpen(false);
            onAction(action);
          }}
        />
      )}
    </div>
  );
}

/** Reads `/account/status`, and re-reads it on demand. */
export function useAccount() {
  const [status, setStatus] = useState<AccountStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/account/status");
      setStatus(response.ok ? ((await response.json()) as AccountStatus) : null);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signOut = useCallback(async () => {
    await fetch("/api/account/logout", { method: "POST" });
    await refresh();
  }, [refresh]);

  return { status, refresh, signOut };
}
