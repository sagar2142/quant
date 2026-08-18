"""Market data events (§1.3). Validation here is the last line of defence
before bad venue data reaches a strategy."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

import core.events
from core.clock import UTC, as_event_time, as_receive_time
from core.events import Bar, CausalityError, EventType, Tick, Timeframe
from core.instruments import InstrumentId

T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
IID = InstrumentId("NSE:INE002A01018")


def bar(**overrides) -> Bar:
    defaults = dict(
        instrument_id=IID,
        source="nse",
        event_time=as_event_time(T0),
        receive_time=as_receive_time(T0 + timedelta(milliseconds=50)),
        timeframe=Timeframe.D1,
        open=Decimal(100),
        high=Decimal(110),
        low=Decimal(95),
        close=Decimal(105),
        volume=Decimal(1000),
    )
    return Bar(**{**defaults, **overrides})


class TestBarValidation:
    def test_valid_bar(self):
        assert bar().close == Decimal(105)

    def test_high_below_low_rejected(self):
        with pytest.raises(ValidationError, match="below low"):
            bar(high=Decimal(90), low=Decimal(95))

    def test_open_outside_range_rejected(self):
        with pytest.raises(ValidationError, match="open"):
            bar(open=Decimal(120))

    def test_close_outside_range_rejected(self):
        with pytest.raises(ValidationError, match="close"):
            bar(close=Decimal(80))

    def test_negative_price_rejected(self):
        with pytest.raises(ValidationError):
            bar(low=Decimal(-1))

    def test_zero_price_rejected(self):
        # A zero price becomes a -100% return becomes an enormous position.
        with pytest.raises(ValidationError):
            bar(low=Decimal(0))

    def test_negative_volume_rejected(self):
        with pytest.raises(ValidationError):
            bar(volume=Decimal(-1))

    def test_zero_volume_allowed(self):
        # Legitimate in illiquid names and halted sessions.
        assert bar(volume=Decimal(0)).volume == 0

    def test_bar_is_frozen(self):
        with pytest.raises(ValidationError):
            bar().close = Decimal(1)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            bar(unexpected_field=1)


class TestCausality:
    def test_receive_before_event_rejected(self):
        with pytest.raises((CausalityError, ValidationError)):
            bar(
                event_time=as_event_time(T0),
                receive_time=as_receive_time(T0 - timedelta(seconds=1)),
            )

    def test_simultaneous_allowed(self):
        assert bar(event_time=as_event_time(T0), receive_time=as_receive_time(T0))


class TestBarProperties:
    def test_typical_price(self):
        assert bar(high=Decimal(110), low=Decimal(100), close=Decimal(105)).typical_price == 105

    def test_doji_detection(self):
        flat = bar(open=Decimal(100), high=Decimal(100), low=Decimal(100), close=Decimal(100))
        assert flat.is_doji
        assert not bar().is_doji

    def test_decimal_precision_preserved(self):
        # A price that cannot round-trip through float64.
        b = bar(close=Decimal("105.10"))
        assert str(b.close) == "105.10"


class TestTimeframe:
    @pytest.mark.parametrize(
        ("tf", "secs"),
        [
            (Timeframe.M1, 60),
            (Timeframe.M15, 900),
            (Timeframe.H1, 3600),
            (Timeframe.H4, 14400),
            (Timeframe.D1, 86400),
        ],
    )
    def test_seconds(self, tf, secs):
        assert tf.seconds == secs


class TestTick:
    def _tick(self, **overrides) -> Tick:
        defaults = dict(
            instrument_id=IID,
            source="nse",
            event_time=as_event_time(T0),
            receive_time=as_receive_time(T0),
        )
        return Tick(**{**defaults, **overrides})

    def test_crossed_book_rejected(self):
        with pytest.raises(ValidationError, match="crossed book"):
            self._tick(bid=Decimal(101), ask=Decimal(100))

    def test_spread_and_mid(self):
        t = self._tick(bid=Decimal(100), ask=Decimal(102))
        assert t.spread == Decimal(2)
        assert t.mid == Decimal(101)

    def test_spread_none_without_quotes(self):
        t = self._tick(price=Decimal(100))
        assert t.spread is None
        assert t.mid is None


class TestEventVocabularyIsEquitiesOnly:
    """§0.0 — the system trades stock exchanges.

    There is no funding event, because no equity instrument pays funding. An
    unused event type is an invitation to write a code path nobody tests.
    """

    def test_no_funding_event_type(self):
        assert not hasattr(core.events, "FundingRate")
        assert "FUNDING" not in {member.value for member in EventType}

    def test_event_types_are_bar_trade_quote(self):
        assert {member.value for member in EventType} == {"BAR", "TRADE", "QUOTE"}
