/**
 * Sign in and register — MASTER_PLAN §13.7.
 *
 * A page rather than a popover, because this is where an operator decides who
 * they are and a cramped dropdown is the wrong place to type a password.
 *
 * **It says what signing in does, and does not do.** On this system an account
 * identifies the operator and carries their settings and broker keys. It does
 * not gate the API: that binds to loopback and answers anything on this
 * machine unless `NEUTRON_API_TOKEN` is set. So the page offers a way past
 * itself, labelled honestly, rather than pretending to be a lock. A login
 * screen that looks like protection and is not would be worse than none on a
 * system that can place real orders.
 */

import { useEffect, useState } from "react";
import { Logo } from "./Logo";

export type AuthMode = "login" | "register";

export interface AuthPageProps {
  /** How many accounts exist, so a fresh install opens on register. */
  userCount: number;
  apiTokenConfigured: boolean;
  accountsAvailable: boolean;
  /** Which database holds accounts: "mongodb" or "postgres". */
  backend?: string;
  onAuthenticated: () => void;
  /** Continue without an account. */
  onSkip: () => void;
}

export function AuthPage({
  userCount,
  apiTokenConfigured,
  accountsAvailable,
  backend = "postgres",
  onAuthenticated,
  onSkip,
}: AuthPageProps) {
  const [mode, setMode] = useState<AuthMode>(userCount === 0 ? "register" : "login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setMode(userCount === 0 ? "register" : "login");
  }, [userCount]);

  const submit = async () => {
    setBusy(true);
    setMessage("");
    try {
      const body =
        mode === "register" ? { email, password, display_name: displayName } : { email, password };
      const response = await fetch(`/api/account/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = payload.detail;
        setMessage(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? // A pydantic validation error — the password rule surfaces
                // here. Show the message, not the schema path.
                detail.map((d: { msg?: string }) => d.msg ?? "invalid").join("; ")
              : `HTTP ${response.status}`,
        );
        return;
      }
      setPassword("");
      onAuthenticated();
    } catch (exc) {
      setMessage(String(exc));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="authpage">
      <div className="auth-card">
        <header className="auth-head">
          <Logo size={38} />
          <div>
            <h1>NEUTRON</h1>
            <p className="muted">Quantitative research and execution</p>
          </div>
        </header>

        {!accountsAvailable ? (
          <>
            {/* Name the database that is actually configured. Telling a
                Mongo install to start Postgres sends the operator to debug a
                service the system is not using. */}
            <p className="auth-note">
              {backend === "mongodb" ? (
                <>
                  MongoDB is unreachable. Check <code>MONGODB_URI</code> and that this machine
                  is allowed by the cluster's network access list, or continue without an
                  account.
                </>
              ) : (
                <>
                  Database unavailable. Start it with <code>docker compose up -d postgres</code>,
                  or continue without an account.
                </>
              )}
            </p>
            <button type="button" className="primary" onClick={onSkip}>
              Continue without an account
            </button>
          </>
        ) : (
          <>
            <div className="segmented auth-tabs">
              <button
                type="button"
                className={mode === "login" ? "on" : ""}
                onClick={() => setMode("login")}
              >
                Sign in
              </button>
              <button
                type="button"
                className={mode === "register" ? "on" : ""}
                onClick={() => setMode("register")}
              >
                Create account
              </button>
            </div>

            {userCount === 0 && (
              <p className="auth-note">No accounts exist. Create the first.</p>
            )}

            <form
              className="auth-form"
              onSubmit={(event) => {
                event.preventDefault();
                void submit();
              }}
            >
              {mode === "register" && (
                <label>
                  <span>Name</span>
                  <input
                    value={displayName}
                    placeholder="Full name"
                    autoComplete="name"
                    onChange={(event) => setDisplayName(event.target.value)}
                  />
                </label>
              )}

              <label>
                <span>Email</span>
                <input
                  value={email}
                  type="email"
                  autoComplete="username"
                  autoFocus
                  onChange={(event) => setEmail(event.target.value)}
                />
              </label>

              <label>
                <span>Password</span>
                <input
                  value={password}
                  type="password"
                  autoComplete={mode === "register" ? "new-password" : "current-password"}
                  onChange={(event) => setPassword(event.target.value)}
                />
                {mode === "register" && (
                  <small>Minimum 12 characters.</small>
                )}
              </label>

              <button type="submit" className="primary" disabled={busy || !email || !password}>
                {busy ? "…" : mode === "register" ? "Create account" : "Sign in"}
              </button>
            </form>

            {message && <p className="ticket-message">{message}</p>}

            <button type="button" className="auth-skip" onClick={onSkip}>
              Continue without an account
            </button>
          </>
        )}

        {/* The honest footnote. An account is identification here, not a lock,
            and the page says which rather than letting the reader assume. */}
        <p className="auth-fineprint">
          {apiTokenConfigured
            ? "API access requires a token."
            : "Accounts hold preferences and broker credentials. API access is controlled separately by NEUTRON_API_TOKEN."}
        </p>
      </div>
    </div>
  );
}
