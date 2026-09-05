/**
 * The trading desk — MASTER_PLAN §8, §12.7.
 *
 * The ticket and the venue's own book, side by side. They were two screens,
 * which is one more than the task has: you place an order and then you look at
 * what the exchange is holding, and having to navigate between those means the
 * moment after sending is the moment you can see least.
 *
 * **The book is the venue's, not this process's.** Between submitting and
 * looking sit rejections, partial fills and cancellations nobody here
 * initiated. A local list of what was sent is a record of intent; only the
 * broker knows the state.
 *
 * **The journal is the third view, behind a tab rather than beside them.** It
 * answers a question asked after the session rather than during it — what the
 * fills actually cost against what the model charged — and putting it in the
 * live pane would trade the thing you need mid-order for the thing you need
 * afterwards.
 */

import { useState } from "react";
import { Journal } from "./Journal";
import { Ticket, type ContractTerms } from "./Ticket";
import { OrderBook } from "./OrderBook";

export interface TradeDeskProps {
  symbol: string;
  onSymbolChange: (symbol: string) => void;
  contract: ContractTerms | null;
  onContractChange: (contract: ContractTerms | null) => void;
}

export function TradeDesk({
  symbol,
  onSymbolChange,
  contract,
  onContractChange,
}: TradeDeskProps) {
  const [view, setView] = useState<"live" | "journal">("live");

  return (
    <div className="deskspace">
      <nav className="tabstrip" aria-label="Trading views">
        <button
          type="button"
          className={view === "live" ? "tab on" : "tab"}
          onClick={() => setView("live")}
          title="Order entry and the venue's book"
        >
          Desk
        </button>
        <button
          type="button"
          className={view === "journal" ? "tab on" : "tab"}
          onClick={() => setView("journal")}
          title="What the fills actually cost, against what the model charged"
        >
          Journal
        </button>
      </nav>

      <div className="deskspace-body">
        {view === "live" ? (
          <div className="desk">
            <section className="desk-ticket" aria-label="Order entry">
              <Ticket
                symbol={symbol}
                onSymbolChange={onSymbolChange}
                contract={contract}
                onContractChange={onContractChange}
              />
            </section>
            <section className="desk-book" aria-label="Venue order book">
              <OrderBook />
            </section>
          </div>
        ) : (
          <Journal />
        )}
      </div>
    </div>
  );
}
