"""Corporate actions loader (§9). Stubbed source — no test touches the network.

Without this loader the backtester treats a 2:1 split as a -50% day, and the
error is invisible: it produces a plausible return series rather than an
exception. These tests exist to keep it visible.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import polars as pl
import pytest

from core.clock import UTC
from core.instruments import InstrumentId
from data.corpactions.actions import ActionType
from data.feeds.yahoo import (
    YahooActionsLoader,
    YahooError,
    nse_yahoo_symbol,
    reconcile_against_prices,
)

IID = InstrumentId("NSE:INE002A01018")


class FakeFrame:
    """Minimal stand-in for the pandas frame yfinance returns."""

    def __init__(self, rows: list[tuple[datetime, dict[str, float]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def iterrows(self):
        yield from self._rows


class FakeTicker:
    def __init__(self, actions: object) -> None:
        self._actions = actions

    @property
    def actions(self) -> object:
        if isinstance(self._actions, Exception):
            raise self._actions
        return self._actions


def loader_for(rows: list[tuple[datetime, dict[str, float]]]) -> YahooActionsLoader:
    return YahooActionsLoader(lambda _symbol: FakeTicker(FakeFrame(rows)))


def ts(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


class TestSymbolMapping:
    def test_nse_suffix(self):
        assert nse_yahoo_symbol("reliance") == "RELIANCE.NS"


class TestSplits:
    def test_split_becomes_an_action(self):
        actions = loader_for([(ts(2024, 10, 28), {"Stock Splits": 2.0, "Dividends": 0.0})]).fetch(
            "RELIANCE", IID
        )
        assert len(actions) == 1
        assert actions[0].action_type is ActionType.SPLIT
        assert actions[0].ratio == Decimal(2)

    def test_ratio_convention_matches_the_plan(self):
        """New shares per existing share (§9).

        A 1:1 bonus and a 2:1 split both arrive as 2.0 — indistinguishable, and
        correctly so: their effect on a position is identical.
        """
        actions = loader_for([(ts(2024, 10, 28), {"Stock Splits": 2.0, "Dividends": 0.0})]).fetch(
            "RELIANCE", IID
        )
        assert actions[0].quantity_multiplier == Decimal(2)
        assert actions[0].price_multiplier == Decimal("0.5")

    def test_reverse_split(self):
        actions = loader_for([(ts(2023, 5, 1), {"Stock Splits": 0.1, "Dividends": 0.0})]).fetch(
            "X", IID
        )
        assert actions[0].ratio == Decimal("0.1")

    def test_ratio_of_one_is_not_an_action(self):
        assert (
            loader_for([(ts(2024, 1, 1), {"Stock Splits": 1.0, "Dividends": 0.0})]).fetch("X", IID)
            == []
        )

    @pytest.mark.parametrize("ratio", [1000.0, 0.0001])
    def test_implausible_ratio_is_dropped(self, ratio):
        """A bad split factor silently rewrites a position by orders of
        magnitude, so it is dropped loudly rather than applied."""
        assert (
            loader_for([(ts(2024, 1, 1), {"Stock Splits": ratio, "Dividends": 0.0})]).fetch(
                "X", IID
            )
            == []
        )


class TestDividends:
    def test_dividend_becomes_an_action(self):
        actions = loader_for([(ts(2024, 8, 19), {"Stock Splits": 0.0, "Dividends": 5.0})]).fetch(
            "RELIANCE", IID
        )
        assert actions[0].action_type is ActionType.DIVIDEND
        assert actions[0].cash_per_share == Decimal(5)

    def test_zero_dividend_ignored(self):
        assert (
            loader_for([(ts(2024, 1, 1), {"Stock Splits": 0.0, "Dividends": 0.0})]).fetch("X", IID)
            == []
        )

    def test_one_row_can_carry_both(self):
        actions = loader_for([(ts(2024, 6, 1), {"Stock Splits": 2.0, "Dividends": 3.0})]).fetch(
            "X", IID
        )
        assert {a.action_type for a in actions} == {ActionType.SPLIT, ActionType.DIVIDEND}


class TestOrderingAndTimezones:
    def test_actions_are_sorted_by_ex_date(self):
        actions = loader_for(
            [
                (ts(2024, 8, 1), {"Stock Splits": 0.0, "Dividends": 5.0}),
                (ts(2022, 3, 1), {"Stock Splits": 2.0, "Dividends": 0.0}),
                (ts(2023, 5, 1), {"Stock Splits": 0.0, "Dividends": 4.0}),
            ]
        ).fetch("X", IID)
        assert [a.ex_date.year for a in actions] == [2022, 2023, 2024]

    def test_naive_stamps_are_made_aware(self):
        actions = loader_for(
            [(datetime(2024, 6, 1), {"Stock Splits": 2.0, "Dividends": 0.0})]
        ).fetch("X", IID)
        assert actions[0].ex_date.tzinfo is not None

    def test_no_announcement_date_falls_back_conservatively(self):
        """Yahoo publishes ex-dates only. Assuming earlier knowledge would grant
        look-ahead (§3.3)."""
        action = loader_for([(ts(2024, 6, 1), {"Stock Splits": 2.0, "Dividends": 0.0})]).fetch(
            "X", IID
        )[0]
        assert action.announcement_date is None
        assert not action.known_at(ts(2024, 5, 31))
        assert action.known_at(ts(2024, 6, 1))


class TestFailures:
    def test_source_failure_raises(self):
        loader = YahooActionsLoader(lambda _s: FakeTicker(RuntimeError("network down")))
        with pytest.raises(YahooError, match="could not fetch"):
            loader.fetch("X", IID)

    def test_empty_result_is_not_an_error(self):
        # Most instruments have no actions in a given window.
        assert loader_for([]).fetch("X", IID) == []

    def mixed_loader(self) -> YahooActionsLoader:
        def factory(symbol: str) -> FakeTicker:
            if symbol.startswith("BAD"):
                return FakeTicker(RuntimeError("delisted"))
            return FakeTicker(
                FakeFrame([(ts(2024, 1, 1), {"Stock Splits": 2.0, "Dividends": 0.0})])
            )

        return YahooActionsLoader(factory)

    def test_book_skips_one_bad_name_by_default(self):
        result = self.mixed_loader().fetch_book({IID: "GOOD", InstrumentId("NSE:BAD"): "BAD"})
        assert len(result.book) == 1

    def test_failed_name_is_reported_not_swallowed(self):
        """A 404 ticker yields zero actions, which is indistinguishable from a
        name that genuinely had none. The distinction has to survive."""
        result = self.mixed_loader().fetch_book({IID: "GOOD", InstrumentId("NSE:BAD"): "BAD"})
        assert result.failures == ("BAD",)
        assert not result.complete
        assert InstrumentId("NSE:BAD") not in result.book.instruments

    def test_full_coverage_reports_complete(self):
        result = self.mixed_loader().fetch_book({IID: "GOOD"})
        assert result.complete
        assert result.failures == ()

    def test_book_can_be_strict(self):
        loader = YahooActionsLoader(lambda _s: FakeTicker(RuntimeError("boom")))
        with pytest.raises(YahooError):
            loader.fetch_book({IID: "X"}, skip_failures=False)


class TestReconciliation:
    """Yahoo is unofficial. A large jump with no recorded action is the
    signature of a missing split (§9)."""

    def bars(self, closes: list[float]) -> pl.DataFrame:
        times = [ts(2024, 1, 1 + i) for i in range(len(closes))]
        return pl.DataFrame(
            {"event_time": times, "close": closes},
            schema_overrides={"event_time": pl.Datetime("us", "UTC")},
        )

    def test_unexplained_halving_is_reported(self):
        from data.corpactions.actions import CorporateActionBook

        bars = self.bars([100.0, 100.0, 50.0, 50.0])
        findings = reconcile_against_prices(CorporateActionBook([]), IID, bars)
        assert len(findings) == 1
        assert "missing split" in findings[0]

    def test_explained_halving_is_silent(self):
        actions = loader_for([(ts(2024, 1, 3), {"Stock Splits": 2.0, "Dividends": 0.0})]).fetch(
            "X", IID
        )
        from data.corpactions.actions import CorporateActionBook

        bars = self.bars([100.0, 100.0, 50.0, 50.0])
        assert reconcile_against_prices(CorporateActionBook(actions), IID, bars) == []

    def test_normal_moves_are_silent(self):
        from data.corpactions.actions import CorporateActionBook

        bars = self.bars([100.0, 101.0, 102.0, 103.0])
        assert reconcile_against_prices(CorporateActionBook([]), IID, bars) == []

    def test_short_series_is_silent(self):
        from data.corpactions.actions import CorporateActionBook

        assert reconcile_against_prices(CorporateActionBook([]), IID, self.bars([100.0])) == []
