"""Instrument master (§1.1) — identity, lifecycle, and point-in-time symbols."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from core.clock import UTC
from core.instruments import (
    AssetClass,
    Currency,
    Exchange,
    Instrument,
    InstrumentId,
    OptionType,
    SymbolAlias,
    SymbolResolver,
    UnknownSymbolError,
)


def equity(**overrides) -> Instrument:
    defaults = dict(
        instrument_id=InstrumentId("NSE:RELIANCE"),
        symbol="RELIANCE",
        asset_class=AssetClass.EQUITY,
        exchange=Exchange.NSE,
        currency=Currency.INR,
        tick_size=Decimal("0.05"),
    )
    return Instrument(**{**defaults, **overrides})


class TestValidation:
    def test_tick_size_must_be_positive(self):
        with pytest.raises(ValidationError):
            equity(tick_size=Decimal(0))

    def test_future_requires_expiry(self):
        with pytest.raises(ValidationError, match="requires an expiry"):
            equity(asset_class=AssetClass.FUTURE)

    def test_option_requires_strike_and_type(self):
        with pytest.raises(ValidationError, match="strike and option_type"):
            equity(
                asset_class=AssetClass.OPTION,
                expiry=datetime(2024, 12, 26, tzinfo=UTC),
            )

    def test_equity_must_not_carry_option_fields(self):
        with pytest.raises(ValidationError, match="must not carry"):
            equity(strike=Decimal(2500))

    def test_valid_option(self):
        opt = equity(
            asset_class=AssetClass.OPTION,
            expiry=datetime(2024, 12, 26, tzinfo=UTC),
            strike=Decimal(2500),
            option_type=OptionType.CALL,
        )
        assert opt.option_type is OptionType.CALL

    def test_delisting_before_listing_rejected(self):
        with pytest.raises(ValidationError, match="precedes"):
            equity(
                listed_on=datetime(2020, 1, 1, tzinfo=UTC),
                delisted_on=datetime(2019, 1, 1, tzinfo=UTC),
            )

    def test_instrument_is_frozen(self):
        with pytest.raises(ValidationError):
            equity().symbol = "OTHER"


class TestSurvivorshipLifecycle:
    """The M2 gate: a 2019 backtest must hold companies delisted in 2022."""

    def test_not_tradable_before_listing(self):
        inst = equity(listed_on=datetime(2020, 1, 1, tzinfo=UTC))
        assert not inst.is_tradable_on(datetime(2019, 6, 1, tzinfo=UTC))
        assert inst.is_tradable_on(datetime(2020, 6, 1, tzinfo=UTC))

    def test_not_tradable_after_delisting(self):
        inst = equity(delisted_on=datetime(2022, 3, 15, tzinfo=UTC))
        assert inst.is_tradable_on(datetime(2022, 3, 14, tzinfo=UTC))
        assert not inst.is_tradable_on(datetime(2022, 3, 15, tzinfo=UTC))

    def test_delisted_name_still_tradable_in_its_own_era(self):
        inst = equity(
            listed_on=datetime(2015, 1, 1, tzinfo=UTC),
            delisted_on=datetime(2022, 1, 1, tzinfo=UTC),
        )
        assert inst.is_tradable_on(datetime(2019, 6, 1, tzinfo=UTC))

    def test_expired_derivative_not_tradable(self):
        fut = equity(
            asset_class=AssetClass.FUTURE,
            expiry=datetime(2024, 12, 26, tzinfo=UTC),
        )
        assert fut.is_tradable_on(datetime(2024, 12, 25, tzinfo=UTC))
        assert not fut.is_tradable_on(datetime(2024, 12, 26, tzinfo=UTC))


class TestPricing:
    def test_round_to_tick(self):
        inst = equity(tick_size=Decimal("0.05"))
        assert inst.round_to_tick(Decimal("100.03")) == Decimal("100.05")
        assert inst.round_to_tick(Decimal("100.02")) == Decimal("100.00")

    def test_notional_uses_multiplier(self):
        fut = equity(
            asset_class=AssetClass.FUTURE,
            expiry=datetime(2024, 12, 26, tzinfo=UTC),
            multiplier=Decimal(50),
        )
        assert fut.notional(Decimal(100), 2) == Decimal(10_000)

    def test_notional_is_exact_decimal(self):
        inst = equity()
        # 0.1 * 3 in float is 0.30000000000000004; Decimal keeps it exact.
        assert inst.notional(Decimal("0.1"), 3) == Decimal("0.3")


class TestAssetClassTraits:
    def test_cash_equity_flag(self):
        # Decides which cost model applies: cash equities attract STT on both
        # legs and DP charges on exit; derivatives do not.
        assert AssetClass.EQUITY.is_cash_equity
        assert AssetClass.ETF.is_cash_equity
        assert not AssetClass.FUTURE.is_cash_equity
        assert not AssetClass.OPTION.is_cash_equity

    def test_derivative_flag(self):
        assert AssetClass.OPTION.is_derivative
        assert AssetClass.FUTURE.is_derivative
        assert not AssetClass.EQUITY.is_derivative


class TestSymbolResolver:
    """Resolution is always as-of. Today's symbol table must not answer 2010."""

    @pytest.fixture
    def resolver(self) -> SymbolResolver:
        iid = InstrumentId("NSE:INFY")
        return SymbolResolver(
            [
                SymbolAlias(
                    instrument_id=iid,
                    symbol="INFOSYSTCH",
                    exchange=Exchange.NSE,
                    valid_from=datetime(2000, 1, 1, tzinfo=UTC),
                    valid_to=datetime(2015, 6, 1, tzinfo=UTC),
                ),
                SymbolAlias(
                    instrument_id=iid,
                    symbol="INFY",
                    exchange=Exchange.NSE,
                    valid_from=datetime(2015, 6, 1, tzinfo=UTC),
                ),
            ]
        )

    def test_old_symbol_resolves_in_its_era(self, resolver):
        got = resolver.resolve("INFOSYSTCH", Exchange.NSE, datetime(2010, 1, 1, tzinfo=UTC))
        assert got == InstrumentId("NSE:INFY")

    def test_new_symbol_resolves_today(self, resolver):
        got = resolver.resolve("INFY", Exchange.NSE, datetime(2024, 1, 1, tzinfo=UTC))
        assert got == InstrumentId("NSE:INFY")

    def test_new_symbol_unknown_before_it_existed(self, resolver):
        with pytest.raises(UnknownSymbolError):
            resolver.resolve("INFY", Exchange.NSE, datetime(2010, 1, 1, tzinfo=UTC))

    def test_old_symbol_unknown_after_window_closed(self, resolver):
        with pytest.raises(UnknownSymbolError):
            resolver.resolve("INFOSYSTCH", Exchange.NSE, datetime(2024, 1, 1, tzinfo=UTC))

    def test_resolution_is_case_insensitive(self, resolver):
        assert resolver.resolve("infy", Exchange.NSE, datetime(2024, 1, 1, tzinfo=UTC))

    def test_unknown_symbol_raises_never_returns_none(self, resolver):
        # §14.1.5: a silently unresolved symbol becomes a silently skipped position.
        with pytest.raises(UnknownSymbolError):
            resolver.resolve("NOSUCH", Exchange.NSE, datetime(2024, 1, 1, tzinfo=UTC))

    def test_symbols_for_instrument_at_time(self, resolver):
        iid = InstrumentId("NSE:INFY")
        assert resolver.symbols_for(iid, datetime(2010, 1, 1, tzinfo=UTC)) == ["INFOSYSTCH"]
        assert resolver.symbols_for(iid, datetime(2024, 1, 1, tzinfo=UTC)) == ["INFY"]

    def test_alias_window_must_be_ordered(self):
        with pytest.raises(ValidationError, match="after valid_from"):
            SymbolAlias(
                instrument_id=InstrumentId("X"),
                symbol="X",
                exchange=Exchange.NSE,
                valid_from=datetime(2020, 1, 1, tzinfo=UTC),
                valid_to=datetime(2019, 1, 1, tzinfo=UTC),
            )
