"""Corporate actions from Yahoo Finance — MASTER_PLAN §9, §13.4.

**Why this exists.** NSE bhavcopy gives prices and nothing else. Without a
corporate-actions source the backtester treats a 2:1 split as a -50% day, and
that error is invisible: it produces a plausible-looking return series rather
than an exception. Every result computed on unadjusted NSE data is wrong by the
size of its splits.

**Free, and unofficial.** Yahoo publishes this without an API key. It is not a
sanctioned feed, the terms restrict redistribution, and the data occasionally
disagrees with the exchange record. Treat it as the *default* source, not the
authoritative one — §9 is explicit that a corporate-actions error should surface
as an alert, which is why `reconcile_against_prices` exists below.

**Split ratios follow the plan's convention** (§9): new shares per existing
share. Yahoo uses the same convention, so a 1:1 bonus arrives as 2.0 and a 2:1
split as 2.0 — indistinguishable, and correctly so, because their effect on a
position is identical.

**No announcement dates.** Yahoo publishes ex-dates only. `CorporateAction`
falls back to the ex-date when no announcement is recorded, which is the
conservative choice: assuming earlier knowledge would grant look-ahead (§3.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

from core.clock import UTC
from core.instruments import InstrumentId
from data.corpactions.actions import ActionType, CorporateAction, CorporateActionBook

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "ActionFetchResult",
    "YahooActionsLoader",
    "YahooError",
    "nse_yahoo_symbol",
]

logger = logging.getLogger(__name__)

#: A split factor this far from 1.0 is almost certainly a data error rather
#: than a corporate action. 1:20 splits exist; 1:1000 do not.
MAX_PLAUSIBLE_RATIO = Decimal(100)
MIN_PLAUSIBLE_RATIO = Decimal("0.01")

#: A price series needs two points before a jump can exist.
MIN_BARS_FOR_RECONCILIATION = 2


class YahooError(RuntimeError):
    """Yahoo returned something unusable. Never swallowed (§14.1.5)."""


@dataclass(frozen=True)
class ActionFetchResult:
    """A book plus the names it could not cover.

    **`fetch_book` returns this rather than a bare book on purpose.** A ticker
    that 404s produces an instrument with zero actions, which is
    indistinguishable from an instrument that genuinely had none — so the
    backtest runs unadjusted for that name and reports nothing. Handing the
    caller a book alone makes that failure unrepresentable in the output;
    handing back the failures forces the decision to be made explicitly.
    """

    book: CorporateActionBook
    failures: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.failures


class TickerFactory(Protocol):
    """Just enough of yfinance to be substitutable in tests.

    Injected rather than imported directly, so no test in this repo reaches the
    network and a future source swap touches one line.
    """

    def __call__(self, symbol: str) -> Any: ...


def nse_yahoo_symbol(symbol: str) -> str:
    """NSE ticker in Yahoo's namespace. RELIANCE -> RELIANCE.NS"""
    return f"{symbol.upper()}.NS"


class YahooActionsLoader:
    """Loads splits, bonuses and dividends for NSE-listed names."""

    def __init__(self, ticker_factory: TickerFactory | None = None) -> None:
        self._factory = ticker_factory

    @property
    def factory(self) -> TickerFactory:
        if self._factory is None:
            try:
                import yfinance  # noqa: PLC0415 - optional dependency, resolved on first use
            except ImportError as exc:  # pragma: no cover - install-time only
                raise YahooError(
                    "yfinance is not installed. pip install yfinance, or inject a "
                    "ticker_factory to use a different source."
                ) from exc
            self._factory = yfinance.Ticker
        return self._factory

    def fetch(self, symbol: str, instrument_id: InstrumentId) -> list[CorporateAction]:
        """Every recorded action for one instrument.

        Args:
            symbol: NSE trading symbol, without the `.NS` suffix.
            instrument_id: Internal id, so actions bind to the stable identity
                rather than to a symbol that may later be renamed (§1.1).

        Raises:
            YahooError: if the response cannot be interpreted. An empty result
                is *not* an error — most instruments have no actions in a given
                window — but an unreadable one is.
        """
        try:
            frame = self.factory(nse_yahoo_symbol(symbol)).actions
        except Exception as exc:
            raise YahooError(f"could not fetch actions for {symbol}: {exc}") from exc

        if frame is None or len(frame) == 0:
            return []

        actions: list[CorporateAction] = []
        for stamp, row in frame.iterrows():
            ex_date = self._to_utc(stamp)
            actions.extend(self._row_to_actions(instrument_id, symbol, ex_date, row))
        return sorted(actions, key=lambda a: a.ex_date)

    def _row_to_actions(
        self,
        instrument_id: InstrumentId,
        symbol: str,
        ex_date: datetime,
        row: Any,
    ) -> list[CorporateAction]:
        """One Yahoo row may carry both a dividend and a split."""
        out: list[CorporateAction] = []

        split = self._decimal(row.get("Stock Splits", 0))
        if split and split != 1:
            if not MIN_PLAUSIBLE_RATIO <= split <= MAX_PLAUSIBLE_RATIO:
                # Implausible ratios are dropped loudly rather than applied. A
                # bad split factor silently rewrites a position by orders of
                # magnitude.
                logger.error(
                    "implausible split ratio %s for %s on %s — dropped",
                    split,
                    symbol,
                    ex_date.date(),
                )
            else:
                out.append(
                    CorporateAction(
                        instrument_id=instrument_id,
                        action_type=ActionType.SPLIT,
                        ex_date=ex_date,
                        ratio=split,
                        note=f"yahoo:{symbol}",
                    )
                )

        dividend = self._decimal(row.get("Dividends", 0))
        if dividend > 0:
            out.append(
                CorporateAction(
                    instrument_id=instrument_id,
                    action_type=ActionType.DIVIDEND,
                    ex_date=ex_date,
                    cash_per_share=dividend,
                    note=f"yahoo:{symbol}",
                )
            )
        return out

    def fetch_book(
        self, symbols: dict[InstrumentId, str], skip_failures: bool = True
    ) -> ActionFetchResult:
        """Build a book for a whole universe.

        Args:
            symbols: instrument_id -> NSE trading symbol.
            skip_failures: Collect and continue when one name fails. A single
                delisted ticker should not abort a hundred-name backfill. Set
                False when completeness matters more than progress.

        Returns:
            The book *and* the symbols that failed. Callers must decide what an
            uncovered name means for them — see `ActionFetchResult`.
        """
        collected: list[CorporateAction] = []
        failures: list[str] = []
        for instrument_id, symbol in sorted(symbols.items()):
            try:
                collected.extend(self.fetch(symbol, instrument_id))
            except YahooError:
                if not skip_failures:
                    raise
                logger.exception("corporate actions unavailable for %s", symbol)
                failures.append(symbol)
        return ActionFetchResult(CorporateActionBook(collected), tuple(failures))

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        """Coerce a pandas scalar to Decimal, treating anything unusable as zero."""
        if value is None:
            return Decimal(0)
        try:
            return Decimal(str(float(value)))
        except (TypeError, ValueError, ArithmeticError):
            return Decimal(0)

    @staticmethod
    def _to_utc(stamp: Any) -> datetime:
        """Yahoo stamps ex-dates in exchange-local time; normalise to UTC."""
        moment = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
        if not isinstance(moment, datetime):
            raise YahooError(f"unreadable ex-date: {stamp!r}")
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
        return moment.astimezone(UTC)


def reconcile_against_prices(
    book: CorporateActionBook,
    instrument_id: InstrumentId,
    bars: pl.DataFrame,
    threshold: float = 0.35,
) -> list[str]:
    """Cross-check the action book against unexplained price jumps.

    Yahoo is unofficial and occasionally incomplete. A large overnight move with
    *no* recorded action is the signature of a missing split — the exact error
    this module exists to prevent, so it is worth detecting rather than assuming
    the source is complete (§9).

    Returns:
        Human-readable descriptions of unexplained jumps. Empty means every
        large move is accounted for.
    """
    import polars as pl  # noqa: PLC0415 - keeps polars out of the import path for callers

    if bars.height < MIN_BARS_FOR_RECONCILIATION:
        return []

    moves = (
        bars.sort("event_time")
        .select(
            pl.col("event_time"),
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret"),
        )
        .drop_nulls()
        .filter(pl.col("ret").abs() > threshold)
    )

    known = {a.ex_date.date() for a in book.for_instrument(instrument_id)}
    unexplained = []
    for row in moves.to_dicts():
        when = row["event_time"].date()
        if when not in known:
            unexplained.append(
                f"{instrument_id} {when}: {row['ret']:+.1%} with no recorded action "
                "— check for a missing split or bonus"
            )
    return unexplained
