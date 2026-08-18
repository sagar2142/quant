"""Corporate actions (§9). Position-level application, no back-adjusted storage."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from pydantic import ValidationError

from core.clock import UTC
from core.instruments import InstrumentId
from data.corpactions.actions import (
    ActionType,
    CorporateAction,
    CorporateActionBook,
    back_adjust,
)

IID = InstrumentId("NSE:INE002A01018")
EX = datetime(2020, 6, 15, tzinfo=UTC)


def split(ratio: str = "2", **kw) -> CorporateAction:
    defaults = dict(
        instrument_id=IID, action_type=ActionType.SPLIT, ex_date=EX, ratio=Decimal(ratio)
    )
    return CorporateAction(**{**defaults, **kw})


class TestRatios:
    def test_two_for_one_split(self):
        action = split("2")
        assert action.quantity_multiplier == Decimal(2)
        assert action.price_multiplier == Decimal("0.5")

    def test_one_for_one_bonus_doubles_shares(self):
        action = split("2", action_type=ActionType.BONUS)
        assert action.quantity_multiplier == Decimal(2)

    def test_three_for_one_bonus(self):
        # 3:1 bonus = three free shares per share = 4x total.
        action = split("4", action_type=ActionType.BONUS)
        assert action.quantity_multiplier == Decimal(4)
        assert action.price_multiplier == Decimal("0.25")

    def test_reverse_split(self):
        action = split("0.1")
        assert action.quantity_multiplier == Decimal("0.1")
        assert action.price_multiplier == Decimal(10)

    def test_position_value_preserved(self):
        action = split("2")
        qty, price = Decimal(100), Decimal(1000)
        before = qty * price
        after = (qty * action.quantity_multiplier) * (price * action.price_multiplier)
        assert before == after

    def test_dividend_leaves_share_count_alone(self):
        action = CorporateAction(
            instrument_id=IID,
            action_type=ActionType.DIVIDEND,
            ex_date=EX,
            cash_per_share=Decimal("12.50"),
        )
        assert action.quantity_multiplier == Decimal(1)
        assert action.price_multiplier == Decimal(1)


class TestValidation:
    def test_split_with_ratio_one_rejected(self):
        with pytest.raises(ValidationError, match="no effect"):
            split("1")

    def test_dividend_without_cash_rejected(self):
        with pytest.raises(ValidationError, match="cash_per_share"):
            CorporateAction(instrument_id=IID, action_type=ActionType.DIVIDEND, ex_date=EX)

    def test_announcement_after_ex_date_rejected(self):
        with pytest.raises(ValidationError, match="cannot take effect"):
            split("2", announcement_date=EX + timedelta(days=1))

    def test_naive_ex_date_rejected(self):
        with pytest.raises(ValidationError):
            CorporateAction(
                instrument_id=IID,
                action_type=ActionType.SPLIT,
                ex_date=datetime(2020, 6, 15),
                ratio=Decimal(2),
            )

    def test_action_is_frozen(self):
        with pytest.raises(ValidationError):
            split("2").ratio = Decimal(3)


class TestKnowledgeTiming:
    """You learn on announcement, it takes effect on ex-date (§3.3)."""

    def test_unknown_before_announcement(self):
        action = split("2", announcement_date=EX - timedelta(days=30))
        assert not action.known_at(EX - timedelta(days=31))

    def test_known_after_announcement_before_ex_date(self):
        action = split("2", announcement_date=EX - timedelta(days=30))
        assert action.known_at(EX - timedelta(days=10))

    def test_without_announcement_falls_back_to_ex_date(self):
        # Conservative: assuming earlier knowledge would grant look-ahead.
        action = split("2")
        assert not action.known_at(EX - timedelta(days=1))
        assert action.known_at(EX)


class TestBook:
    @pytest.fixture
    def book(self) -> CorporateActionBook:
        return CorporateActionBook(
            [
                split("2", ex_date=datetime(2019, 3, 1, tzinfo=UTC)),
                split("2", ex_date=datetime(2021, 9, 1, tzinfo=UTC), action_type=ActionType.BONUS),
                CorporateAction(
                    instrument_id=InstrumentId("NSE:OTHER"),
                    action_type=ActionType.SPLIT,
                    ex_date=EX,
                    ratio=Decimal(5),
                ),
            ]
        )

    def test_len(self, book):
        assert len(book) == 3

    def test_isolated_per_instrument(self, book):
        assert len(book.for_instrument(IID)) == 2
        assert len(book.for_instrument(InstrumentId("NSE:OTHER"))) == 1

    def test_unknown_instrument_is_empty(self, book):
        assert book.for_instrument(InstrumentId("NSE:NOPE")) == []

    def test_effective_between_is_half_open(self, book):
        # Exactly-on-ex-date is included at the upper bound, excluded at lower,
        # so stepping bar by bar applies each action once.
        found = book.effective_between(
            IID, datetime(2019, 3, 1, tzinfo=UTC), datetime(2021, 9, 1, tzinfo=UTC)
        )
        assert len(found) == 1
        assert found[0].ex_date == datetime(2021, 9, 1, tzinfo=UTC)

    def test_cumulative_factor_compounds(self, book):
        factor = book.cumulative_quantity_factor(
            IID, datetime(2018, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)
        )
        assert factor == Decimal(4)  # 2x split then 2x bonus

    def test_cumulative_factor_is_one_when_no_actions(self, book):
        factor = book.cumulative_quantity_factor(
            IID, datetime(2022, 1, 1, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)
        )
        assert factor == Decimal(1)

    def test_known_at_filters(self, book):
        assert book.known_at(IID, datetime(2020, 1, 1, tzinfo=UTC)) == [
            a for a in book.for_instrument(IID) if a.ex_date.year == 2019
        ]


class TestBackAdjust:
    """Charting and vendor reconciliation only — never a backtest."""

    def frame(self) -> pl.DataFrame:
        days = [datetime(2020, 6, 10, tzinfo=UTC) + timedelta(days=i) for i in range(10)]
        # Price halves at the ex-date, as a 2:1 split does mechanically.
        closes = [1000.0] * 5 + [500.0] * 5
        return pl.DataFrame(
            {
                "event_time": days,
                "receive_time": days,
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": [1000.0] * 10,
                "trades": [10] * 10,
            },
            schema_overrides={
                "event_time": pl.Datetime("us", "UTC"),
                "receive_time": pl.Datetime("us", "UTC"),
            },
        )

    def test_split_jump_removed(self):
        adjusted = back_adjust(self.frame(), [split("2")])
        assert adjusted["close"].n_unique() == 1

    def test_volume_scales_inversely(self):
        adjusted = back_adjust(self.frame(), [split("2")])
        # Pre-split volume doubles: same value, more shares.
        assert adjusted["volume"][0] == pytest.approx(2000.0)
        assert adjusted["volume"][-1] == pytest.approx(1000.0)

    def test_no_actions_is_identity(self):
        original = self.frame()
        assert back_adjust(original, []).equals(original)

    def test_empty_frame_is_safe(self):
        empty = self.frame().head(0)
        assert back_adjust(empty, [split("2")]).is_empty()

    def test_dividend_does_not_adjust_price(self):
        action = CorporateAction(
            instrument_id=IID,
            action_type=ActionType.DIVIDEND,
            ex_date=EX,
            cash_per_share=Decimal(10),
        )
        original = self.frame()
        assert back_adjust(original, [action]).equals(original)
