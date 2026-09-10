/**
 * Which names share a stationary spread — MASTER_PLAN §6.
 *
 * **The one strategy family the console could not reach.** Engle-Granger, the
 * hedge ratio and the spread have been built and tested since the mathematics
 * went in, and nothing outside a test had ever called them: pairs trading
 * existed as a library and not as a screen.
 *
 * **Every unordered pair is tested, and the count is on the screen.** Which
 * pair cointegrates is exactly what is not known in advance, so a search is the
 * honest way to look — and searching is also the textbook way to manufacture a
 * false positive. Fifteen tests and one significant result is not one
 * significant result, and the reader cannot discount for that without being
 * told the denominator.
 *
 * **Tradable is the column to read, not cointegrated.** A spread whose
 * half-life exceeds a quarter of the sample cannot be verified inside it, and
 * one that takes a year to close is a directional position wearing a
 * pairs-trade label — its costs will exceed its reversion (§7.1).
 */

import { useCallback, useEffect, useState } from "react";

interface PairRow {
  a: string;
  b: string;
  cointegrated: boolean;
  tradable: boolean;
  hedge_ratio: number;
  intercept: number;
  adf_pvalue: number;
  half_life_bars: number | null;
  correlation: number;
  observations: number;
  spread_verdict: string;
  spread_z: number | null;
}

interface PairsResponse {
  sessions: number;
  tested: number;
  rows: PairRow[];
  note: string;
}

//: Beyond this many standard deviations the spread is far enough from its own
//: mean to be an entry rather than a position to hold. Not a limit — the risk
//: engine owns those — but the number at which the screen stops being neutral.
const ENTRY_Z = 2.0;

const EXAMPLES = "HDFCBANK,ICICIBANK,AXISBANK,KOTAKBANK,SBIN,INDUSINDBK";

export function Pairs({ apiBase = "/api" }: { apiBase?: string }) {
  const [symbols, setSymbols] = useState(EXAMPLES);
  const [pending, setPending] = useState(EXAMPLES);
  const [sessions, setSessions] = useState(500);
  const [data, setData] = useState<PairsResponse | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(
        `${apiBase}/pairs?symbols=${encodeURIComponent(symbols)}&sessions=${sessions}`,
      );
      const body = await response.json();
      if (!response.ok) {
        setError(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
        setData(null);
        return;
      }
      setData(body as PairsResponse);
    } catch (exc) {
      setError(String(exc));
    } finally {
      setBusy(false);
    }
  }, [apiBase, symbols, sessions]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="pairs">
      <form
        className="analytics-bar"
        onSubmit={(event) => {
          event.preventDefault();
          setSymbols(pending);
        }}
      >
        <input
          className="analytics-input"
          value={pending}
          onChange={(event) => setPending(event.target.value)}
          placeholder="Comma-separated symbols"
          aria-label="symbols"
        />
        <select
          className="analytics-select"
          value={sessions}
          onChange={(event) => setSessions(Number(event.target.value))}
          aria-label="sessions"
        >
          {[250, 500, 750, 1000].map((n) => (
            <option key={n} value={n}>
              {n} sessions
            </option>
          ))}
        </select>
        <button type="submit" className="primary" disabled={busy}>
          {busy ? "…" : "Test"}
        </button>
      </form>

      {error && <div className="analytics-note text-critical">{error}</div>}
      {data && <div className="analytics-note">{data.note}</div>}

      {data && data.rows.length > 0 && (
        <div className="scroll-x">
          <table className="grid">
            <thead>
              <tr>
                <th>pair</th>
                <th className="num">hedge</th>
                <th className="num">ADF p</th>
                <th>verdict</th>
                <th className="num">half-life</th>
                <th className="num">corr</th>
                <th className="num">spread z</th>
                <th>&nbsp;</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={`${row.a}/${row.b}`} className={row.tradable ? "" : "text-secondary"}>
                  <td className="mono">
                    {row.a} / {row.b}
                  </td>
                  <td className="num mono">{row.hedge_ratio.toFixed(3)}</td>
                  <td className="num mono">{row.adf_pvalue.toFixed(4)}</td>
                  {/* The field to read when a low p-value sits beside a false
                      verdict: it needs ADF and KPSS to agree, so a spread can
                      reject the unit root and still be INCONCLUSIVE. */}
                  <td className="mono">{row.spread_verdict}</td>
                  <td className="num mono">
                    {row.half_life_bars === null ? "—" : row.half_life_bars.toFixed(0)}
                  </td>
                  <td className="num mono">{row.correlation.toFixed(2)}</td>
                  {/* Null, not zero. A spread with no dispersion has no
                      z-score, and zero would read as sitting on its mean. */}
                  <td
                    className={
                      row.spread_z !== null && Math.abs(row.spread_z) >= ENTRY_Z
                        ? "num mono up"
                        : "num mono"
                    }
                  >
                    {row.spread_z === null ? "—" : row.spread_z.toFixed(2)}
                  </td>
                  <td>
                    {row.tradable && <span className="badge live">TRADABLE</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="block-note">
        <strong>Tradable</strong> means cointegrated <em>and</em> reverting fast enough to
        pay for the round trip — a half-life beyond a quarter of the sample cannot be
        verified inside it. A spread more than {ENTRY_Z} deviations from its own mean is
        the entry; a cointegrated pair sitting at zero has nothing to do today.
      </p>
    </div>
  );
}
