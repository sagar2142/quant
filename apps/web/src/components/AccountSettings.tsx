/**
 * Account settings — MASTER_PLAN §13.7, §21.
 *
 * A full page with sections, because what lives here does not fit a dropdown:
 * an identity, console preferences, broker credentials that can move money,
 * and a plain statement of what this system's access model actually is.
 *
 * **The security section tells the truth rather than reassuring.** It reports
 * whether the API requires a token, whether credentials can be encrypted at
 * all, and what each answer means. A settings screen that only ever shows
 * green ticks teaches an operator to stop reading it.
 */

import { useCallback, useEffect, useState } from "react";
import { Icon } from "./Icon";
import { Logo } from "./Logo";

interface Account {
  user_id: string;
  email: string;
  display_name: string;
  settings: Record<string, unknown>;
  created_at: string;
  last_login_at: string | null;
}

interface BrokerStatus {
  broker: string;
  configured: boolean;
  api_key_hint: string;
  access_token_hint: string;
  updated_at: string | null;
}

interface VaultStatus {
  encryption_available: boolean;
  brokers: BrokerStatus[];
  environment_overrides: boolean;
}

export type SettingsSection = "profile" | "preferences" | "broker";

const SECTIONS: { id: SettingsSection; label: string }[] = [
  { id: "profile", label: "Profile" },
  { id: "preferences", label: "Preferences" },
  { id: "broker", label: "Broker API" },
];

export interface AccountSettingsProps {
  account: Account | null;
  section: SettingsSection;
  onSection: (section: SettingsSection) => void;
  onClose: () => void;
  onChanged: () => void;
}

export function AccountSettings({
  account,
  section,
  onSection,
  onClose,
  onChanged,
}: AccountSettingsProps) {
  const [vault, setVault] = useState<VaultStatus | null>(null);
  const broker = "groww";
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const loadVault = useCallback(async () => {
    const response = await fetch("/api/account/broker");
    if (response.ok) setVault((await response.json()) as VaultStatus);
  }, []);

  useEffect(() => {
    void loadVault();
  }, [loadVault]);

  const saveSetting = async (patch: Record<string, unknown>) => {
    const response = await fetch("/api/account/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    setMessage(response.ok ? "Saved" : "Could not save");
    if (response.ok) onChanged();
  };

  const storeKeys = async () => {
    setBusy(true);
    setMessage("");
    try {
      const response = await fetch("/api/account/broker", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          broker,
          api_key: apiKey,
          api_secret: apiSecret,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        // A validation failure arrives as a list of field errors. Showing the
        // status code instead — which is what this did — tells an operator a
        // number and leaves them to guess which field the server disliked.
        const detail = payload.detail;
        setMessage(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail
                  .map((entry: { loc?: unknown[]; msg?: string }) => {
                    const field = Array.isArray(entry.loc) ? entry.loc.slice(1).join(".") : "";
                    return field ? `${field}: ${entry.msg ?? "invalid"}` : (entry.msg ?? "invalid");
                  })
                  .join("; ")
              : `Request failed (${response.status})`,
        );
        return;
      }
      setVault(payload as VaultStatus);
      // Cleared the moment they are stored. The server will not hand them
      // back, so a populated form is the one place they remain visible.
      setApiKey("");
      setApiSecret("");
      setMessage(`${broker} credentials stored`);
    } finally {
      setBusy(false);
    }
  };

  const forget = async (which: string) => {
    const response = await fetch(`/api/account/broker/${which}`, { method: "DELETE" });
    if (response.ok) setVault((await response.json()) as VaultStatus);
  };

  const settings = account?.settings ?? {};

  return (
    <div className="settings-page">
      <header className="settings-head">
        <Logo size={22} />
        <h1>Settings</h1>
        <button type="button" className="ghost" onClick={onClose}>
          <Icon name="close" /> Close
        </button>
      </header>

      <div className="settings-body">
        <nav className="settings-nav" aria-label="Settings sections">
          {SECTIONS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              className={section === entry.id ? "on" : ""}
              onClick={() => onSection(entry.id)}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        <div className="settings-content">
          {!account && (
            <p className="auth-note">Sign in to manage profile, preferences and API keys.</p>
          )}

          {section === "profile" && account && (
            <section>
              <h2>Profile</h2>
              <dl className="stats">
                <div className="stat">
                  <dt>Name</dt>
                  <dd>{account.display_name || "—"}</dd>
                </div>
                <div className="stat">
                  <dt>Email</dt>
                  <dd className="mono">{account.email}</dd>
                </div>
                <div className="stat">
                  <dt>Created</dt>
                  <dd className="mono">{account.created_at.slice(0, 10)}</dd>
                </div>
                <div className="stat">
                  <dt>Last sign-in</dt>
                  <dd className="mono">{account.last_login_at?.slice(0, 10) ?? "—"}</dd>
                </div>
              </dl>
            </section>
          )}

          {section === "preferences" && account && (
            <>
              <section>
                <h2>Market</h2>

                <label>
                  <span>Default venue</span>
                  <select
                    value={String(settings.default_venue ?? "NSE")}
                    onChange={(event) => void saveSetting({ default_venue: event.target.value })}
                  >
                    <option value="NSE">NSE</option>
                    <option value="BSE">BSE</option>
                  </select>
                  <small>Which exchange every screen opens on.</small>
                </label>

                <label>
                  <span>Default capital</span>
                  <input
                    defaultValue={String(settings.default_capital ?? "")}
                    placeholder="Account capital"
                    inputMode="numeric"
                    onBlur={(event) =>
                      event.target.value &&
                      void saveSetting({ default_capital: event.target.value })
                    }
                  />
                  <small>Basis for position and exposure limits in the order ticket.</small>
                </label>

                <label>
                  <span>Default order type</span>
                  <select
                    value={String(settings.default_order_type ?? "MARKET")}
                    onChange={(event) =>
                      void saveSetting({ default_order_type: event.target.value })
                    }
                  >
                    <option value="MARKET">Market</option>
                    <option value="LIMIT">Limit</option>
                  </select>
                </label>
              </section>

              <section>
                <h2>Charts</h2>

                <label>
                  <span>Default interval</span>
                  <select
                    value={String(settings.default_interval ?? "1d")}
                    onChange={(event) => void saveSetting({ default_interval: event.target.value })}
                  >
                    <option value="1m">1 minute</option>
                    <option value="5m">5 minutes</option>
                    <option value="10m">10 minutes</option>
                    <option value="1h">1 hour</option>
                    <option value="4h">4 hours</option>
                    <option value="1d">1 day</option>
                    <option value="1w">1 week</option>
                  </select>
                  <small>
                    Intraday intervals come from the broker and need a connection; daily and
                    weekly come from the stored panel.
                  </small>
                </label>

                <label>
                  <span>Default range</span>
                  <select
                    value={String(settings.default_sessions ?? 252)}
                    onChange={(event) =>
                      void saveSetting({ default_sessions: Number(event.target.value) })
                    }
                  >
                    <option value="1">1 day</option>
                    <option value="5">5 days</option>
                    <option value="21">1 month</option>
                    <option value="63">3 months</option>
                    <option value="126">6 months</option>
                    <option value="252">1 year</option>
                    <option value="756">3 years</option>
                    <option value="0">Maximum</option>
                  </select>
                </label>

                <label>
                  <span>Moving averages</span>
                  <input
                    defaultValue={
                      Array.isArray(settings.chart_overlays)
                        ? (settings.chart_overlays as number[]).join(", ")
                        : "20, 50, 200"
                    }
                    placeholder="20, 50, 200"
                    onBlur={(event) =>
                      void saveSetting({
                        chart_overlays: event.target.value
                          .split(",")
                          .map((part) => Number(part.trim()))
                          .filter((n) => Number.isFinite(n) && n > 0),
                      })
                    }
                  />
                  <small>Periods drawn over price. Leave empty for none.</small>
                </label>

                <label className="settings-check">
                  <input
                    type="checkbox"
                    checked={Boolean(settings.default_log_scale)}
                    onChange={(event) =>
                      void saveSetting({ default_log_scale: event.target.checked })
                    }
                  />
                  <span>Logarithmic price axis</span>
                  <small>
                    Preferable over multi-year spans, where a linear axis makes an older move look
                    smaller than a recent one of the same size.
                  </small>
                </label>
              </section>

              <section>
                <h2>Watchlist</h2>
                <label>
                  <span>Symbols</span>
                  <input
                    defaultValue={
                      Array.isArray(settings.watchlist)
                        ? (settings.watchlist as string[]).join(", ")
                        : ""
                    }
                    placeholder="RELIANCE, TCS, INFY"
                    onBlur={(event) =>
                      void saveSetting({
                        watchlist: event.target.value
                          .split(",")
                          .map((part) => part.trim().toUpperCase())
                          .filter(Boolean),
                      })
                    }
                  />
                  <small>Comma separated. Loads into the Market workspace.</small>
                </label>
              </section>

              <section>
                <h2>Console</h2>
                <label>
                  <span>Refresh interval</span>
                  <select
                    value={String(settings.refresh_seconds ?? 1)}
                    onChange={(event) =>
                      void saveSetting({ refresh_seconds: Number(event.target.value) })
                    }
                  >
                    <option value="1">1 second</option>
                    <option value="5">5 seconds</option>
                    <option value="15">15 seconds</option>
                    <option value="60">1 minute</option>
                    <option value="0">Paused</option>
                  </select>
                  <small>How often vitals and health are polled.</small>
                </label>
              </section>
            </>
          )}

          {section === "broker" && (
            <section>
              <h2>Broker API</h2>

              {vault && !vault.encryption_available && (
                <p className="account-warning">
                  <code>NEUTRON_SECRET_KEY</code> is not set. Credentials cannot be stored without
                  it — generate one with <code>python -m apps.cli.vault --generate</code>.
                </p>
              )}

              {vault?.environment_overrides && (
                <p className="account-warning">
                  Environment credentials are set and take priority over stored keys.
                </p>
              )}

              {vault?.brokers
                .filter((entry) => entry.configured)
                .map((entry) => (
                  <div className="broker-row" key={entry.broker}>
                    <span className="mono">{entry.broker}</span>
                    <span className="muted">
                      key {entry.api_key_hint}
                      {entry.updated_at && ` · ${entry.updated_at.slice(0, 10)}`}
                    </span>
                    <button
                      type="button"
                      className="ghost danger"
                      onClick={() => void forget(entry.broker)}
                    >
                      Remove
                    </button>
                  </div>
                ))}

              <label>
                <span>API key</span>
                <input
                  value={apiKey}
                  type="password"
                  autoComplete="off"
                  placeholder="From Groww → Trading APIs"
                  onChange={(event) => setApiKey(event.target.value)}
                />
              </label>

              <label>
                <span>Secret key</span>
                <input
                  value={apiSecret}
                  type="password"
                  autoComplete="off"
                  placeholder="Shown once when the key is created"
                  onChange={(event) => setApiSecret(event.target.value)}
                />
                <small>
                  The access token is generated from these and refreshed automatically before it
                  expires at 06:00 IST. Both are stored encrypted and never displayed again.
                </small>
              </label>

              <button
                type="button"
                className="primary"
                disabled={busy || !account || !apiKey || !apiSecret}
                onClick={() => void storeKeys()}
              >
                {busy ? "…" : "Store encrypted"}
              </button>
            </section>
          )}

          {message && <p className="ticket-message">{message}</p>}
        </div>
      </div>
    </div>
  );
}
