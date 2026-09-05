/**
 * The research workspace — MASTER_PLAN §12.6, §5.5.
 *
 * Six views of the same question — is there anything here worth trading —
 * behind tabs rather than behind six separate nav entries. The order is the
 * order of the research loop (§5.5): screen the universe, decompose one name,
 * check it is not the same bet as everything else, ask whether the factor
 * behind it survives its own costs, see what the book is really exposed to,
 * and finally trade it in simulation and put it through the gauntlet.
 *
 * **The symbol is the shell's, not this screen's.** Picking a name in the
 * screener moves the whole console to it, so the chart, the analysis and the
 * ticket cannot disagree about which security is under discussion.
 */

import { Analytics } from "./Analytics";
import { Factors } from "./Factors";
import { Lab } from "./Lab";
import { Research } from "./Research";
import { RiskModel } from "./RiskModel";
import { Screener } from "./Screener";
import type { ContractTerms } from "./Ticket";

export type ResearchTab =
  | "security"
  | "screener"
  | "crosssection"
  | "factors"
  | "riskmodel"
  | "lab";

export const RESEARCH_TABS: { id: ResearchTab; label: string; hint: string }[] = [
  { id: "security", label: "Security", hint: "One name, fully decomposed" },
  { id: "screener", label: "Screener", hint: "Which names, rather than what is this name" },
  { id: "crosssection", label: "Cross-section", hint: "Correlation, clusters and weights" },
  { id: "factors", label: "Factors", hint: "Signal scores across the universe" },
  { id: "riskmodel", label: "Risk model", hint: "What the book is actually exposed to" },
  { id: "lab", label: "Lab", hint: "Run a backtest, or put one through the gauntlet" },
];

export interface ResearchDeskProps {
  tab: ResearchTab;
  onTab: (tab: ResearchTab) => void;
  symbol: string;
  venue: string;
  onSymbol: (symbol: string) => void;
  onVenue: (venue: string) => void;
  onTrade: (symbol: string) => void;
  onTradeContract: (symbol: string, terms: ContractTerms) => void;
}

export function ResearchDesk({
  tab,
  onTab,
  symbol,
  venue,
  onSymbol,
  onVenue,
  onTrade,
  onTradeContract,
}: ResearchDeskProps) {
  return (
    <div className="deskspace">
      <nav className="tabstrip" aria-label="Research views">
        {RESEARCH_TABS.map((entry) => (
          <button
            key={entry.id}
            type="button"
            className={tab === entry.id ? "tab on" : "tab"}
            onClick={() => onTab(entry.id)}
            title={entry.hint}
          >
            {entry.label}
          </button>
        ))}
      </nav>

      <div className="deskspace-body">
        {tab === "security" && (
          <Research
            symbol={symbol}
            venue={venue}
            onSymbolChange={onSymbol}
            onVenueChange={onVenue}
            onTrade={onTrade}
            onTradeContract={onTradeContract}
          />
        )}
        {tab === "screener" && <Screener onPick={(picked) => {
          onSymbol(picked);
          onTab("security");
        }} />}
        {tab === "crosssection" && <Analytics key={symbol || "none"} initialSymbols={symbol} />}
        {tab === "factors" && <Factors />}
        {tab === "riskmodel" && <RiskModel />}
        {tab === "lab" && <Lab />}
      </div>
    </div>
  );
}
