/**
 * The account menu — MASTER_PLAN §12.6, §13.7.
 *
 * A list of destinations, in the shape every desktop tool has settled on: who
 * you are at the top, then the things you might want to change, then the way
 * out. The previous version was a popover that tried to be the settings screen
 * as well, which meant a password field and a broker key field in a 280px box
 * anchored to the bottom of a sidebar.
 *
 * Menu and settings are now separate: this decides where to go, and
 * `AccountSettings` is where the work happens.
 */

import { useEffect, useRef } from "react";
import { Icon, type IconName } from "./Icon";

export interface AccountSummary {
  email: string;
  display_name: string;
}

export type AccountAction = "settings" | "broker" | "preferences" | "signin" | "signout";

interface Entry {
  id: AccountAction;
  label: string;
  icon?: IconName;
  /** Separated from what sits above it. */
  divider?: boolean;
  /** Shown only when signed in, or only when signed out. */
  when: "in" | "out" | "always";
  danger?: boolean;
}

const ENTRIES: Entry[] = [
  { id: "settings", label: "Account settings…", when: "in" },
  { id: "broker", label: "Manage Broker API keys…", when: "in" },
  { id: "preferences", label: "Manage console preferences…", when: "in" },
  { id: "signin", label: "Sign in to Neutron…", when: "out", divider: true },
  { id: "signout", label: "Sign out", when: "in", divider: true, danger: true },
];

export interface AccountMenuProps {
  account: AccountSummary | null;
  onAction: (action: AccountAction) => void;
  onDismiss: () => void;
}

export function AccountMenu({ account, onAction, onDismiss }: AccountMenuProps) {
  const box = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const away = (event: MouseEvent) => {
      if (box.current && !box.current.contains(event.target as Node)) onDismiss();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDismiss();
    };
    document.addEventListener("mousedown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("mousedown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [onDismiss]);

  const visible = ENTRIES.filter(
    (entry) =>
      entry.when === "always" || (entry.when === "in") === (account !== null),
  );

  return (
    <div className="account-menu" ref={box} role="menu">
      {account && (
        <div className="account-menu-head">
          <span className="account-avatar">
            {(account.display_name || account.email).slice(0, 1).toUpperCase()}
          </span>
          <div className="account-menu-id">
            <div className="account-name">{account.display_name || account.email}</div>
            <div className="muted">{account.email}</div>
          </div>
        </div>
      )}

      <ul>
        {visible.map((entry) => (
          <li key={entry.id} className={entry.divider ? "divided" : undefined}>
            <button
              type="button"
              role="menuitem"
              className={entry.danger ? "danger" : undefined}
              onClick={() => onAction(entry.id)}
            >
              {entry.icon && <Icon name={entry.icon} />}
              {entry.label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
