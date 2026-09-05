/**
 * What the universe is made of — MASTER_PLAN §1.1, §8.
 *
 * **Nothing on any screen knew what a company does.** Every view was built
 * from price, so a universe eleven-thirtieths in one industry looked, from all
 * of them, like thirty independent positions. On the real top-30 that is
 * exactly the case: 36.7% Financial Services.
 *
 * Sits beside the correlation matrix rather than replacing it. The two answer
 * different questions and disagree usefully — two banks that do not move
 * together are two bets whatever the label says, and two names in different
 * industries that move as one are a single bet the label would miss. Seeing
 * both is the point.
 *
 * **Coverage is on the screen, not implied.** A company delisted before the
 * classification was observed appears in no current constituent list, so the
 * names missing from this breakdown are disproportionately the ones that
 * failed — the direction that flatters. A reader who cannot see the coverage
 * cannot discount for it.
 */

import { useCallback, useEffect, useState } from "react";
import { formatPercent } from "../format";

interface SectorRow {
  industry: string;
  names: number;
  share: number;
}

interface SectorBreakdown {
  observed_at: string | null;
  coverage: number;
  universe: number;
  rows: SectorRow[];
  note: string;
}

//: Above this share of the universe in one industry, the concentration is the
//: headline rather than a detail. Not a limit — the risk engine owns those —
//: but the number at which "diversified" stops being the right word.
const CONCENTRATED = 0.25;

export function Sectors({ apiBase = "/api" }: { apiBase?: string }) {
  const [top, setTop] = useState(100);
  const [data, setData] = useState<SectorBreakdown | null>(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setError("");
    const response = await fetch(`${apiBase}/sectors?top=${top}`);
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as { detail?: unknown };
      setError(typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`);
      return;
    }
    setData((await response.json()) as SectorBreakdown);
  }, [apiBase, top]);

  useEffect(() => {
    void load();
  }, [load]);

  const largest = data?.rows[0];

  return (
    <div className="sectors">
      <div className="analytics-bar">
        <select
          className="analytics-select"
          value={top}
          onChange={(event) => setTop(Number(event.target.value))}
          aria-label="universe size"
        >
          {[30, 50, 100, 200, 300].map((n) => (
            <option key={n} value={n}>
              top {n}
            </option>
          ))}
        </select>
        <span className="analytics-spacer" />
        {data?.observed_at ? (
          <span className="text-secondary">classified {data.observed_at}</span>
        ) : null}
      </div>

      {error ? <div className="analytics-note text-critical">{error}</div> : null}
      {data?.note ? <div className="analytics-note">{data.note}</div> : null}

      {data && data.rows.length > 0 ? (
        <>
          {/* The concentration first. A ranked list read top-down buries the
              one number that changes how the rest should be read. */}
          {largest && largest.share >= CONCENTRATED ? (
            <div className="verdict-banner verdict-warn">
              <span className="verdict-word">CONCENTRATED</span>
              <span>
                {formatPercent(largest.share)} of the top {data.universe} is{" "}
                {largest.industry}. Correlation clustering, not this, decides the risk
                limit — but a book drawn from here starts unbalanced.
              </span>
            </div>
          ) : null}

          <table className="grid">
            <thead>
              <tr>
                <th>industry</th>
                <th className="num">names</th>
                <th className="num">share</th>
                <th>&nbsp;</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={row.industry}>
                  <td className={row.industry === "unclassified" ? "text-secondary" : ""}>
                    {row.industry}
                  </td>
                  <td className="num mono">{row.names}</td>
                  <td className="num mono">{formatPercent(row.share)}</td>
                  <td className="sector-bar-cell">
                    <span
                      className={`sector-bar ${row.share >= CONCENTRATED ? "wide" : ""}`}
                      style={{ width: `${Math.min(100, row.share * 100)}%` }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <p className="block-note">
            {formatPercent(data.coverage)} of the top {data.universe} is classified by NSE.
            The unclassified remainder is not random: a company delisted before the
            classification was observed appears in no current constituent list, so what is
            missing here is weighted toward the names that failed.
          </p>
        </>
      ) : null}
    </div>
  );
}
