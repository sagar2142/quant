"""The independent risk engine (§8, §21).

Safety-critical, so §14.5 requires every limit tested at the boundary and on
both sides of it. `TestFailClosed` is the class that matters most: an engine
that permits what it cannot evaluate is not a risk engine.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.instruments import InstrumentId
from trading.risk.engine import KillSwitchEngagedError, RiskEngine
from trading.risk.limits import (
    DrawdownLadder,
    LadderRung,
    PortfolioState,
    ProposedOrder,
    RiskDecision,
    RiskLimits,
)

A = InstrumentId("NSE:A")
B = InstrumentId("NSE:B")

EQUITY = Decimal(1_000_000)


def state(**overrides) -> PortfolioState:
    defaults = dict(
        equity=EQUITY,
        cash=EQUITY,
        peak_equity=EQUITY,
        day_start_equity=EQUITY,
        positions={},
        clusters={},
        open_orders=0,
        orders_this_minute=0,
        last_prices={A: Decimal(100), B: Decimal(100)},
        adv={A: Decimal(10_000_000), B: Decimal(10_000_000)},
    )
    return PortfolioState(**{**defaults, **overrides})


def order(**overrides) -> ProposedOrder:
    defaults = dict(
        strategy_id="s1",
        instrument_id=A,
        quantity=Decimal(100),
        price=Decimal(100),
    )
    return ProposedOrder(**{**defaults, **overrides})


def check_named(verdict, name: str):
    return next(c for c in verdict.checks if c.name == name)


class TestKillSwitch:
    """The mechanism that stops everything must not itself be complicated."""

    def test_blocks_every_order(self):
        engine = RiskEngine()
        engine.engage_kill("data staleness", "operator")
        verdict = engine.check(order(), state())
        assert verdict.decision is RiskDecision.BLOCK
        assert verdict.scale == 0

    def test_reason_and_operator_recorded(self):
        engine = RiskEngine()
        engine.engage_kill("feed gap", "sagar")
        assert "sagar" in check_named(engine.check(order(), state()), "kill_switch").message

    def test_requires_a_reason(self):
        with pytest.raises(ValueError, match="reason"):
            RiskEngine().engage_kill("  ", "operator")

    def test_requires_an_operator(self):
        with pytest.raises(ValueError, match="operator"):
            RiskEngine().engage_kill("something broke", "")

    def test_release_is_manual(self):
        engine = RiskEngine()
        engine.engage_kill("incident", "operator")
        assert engine.is_killed
        engine.release_kill("operator")
        assert not engine.is_killed
        assert engine.check(order(), state()).allowed

    def test_release_requires_an_operator(self):
        engine = RiskEngine()
        engine.engage_kill("incident", "operator")
        with pytest.raises(ValueError, match="operator"):
            engine.release_kill("")

    def test_raise_if_killed(self):
        engine = RiskEngine()
        engine.engage_kill("halt", "operator")
        with pytest.raises(KillSwitchEngagedError, match="halt"):
            engine.raise_if_killed()

    def test_raise_is_silent_when_live(self):
        RiskEngine().raise_if_killed()


class TestFailClosed:
    """What cannot be evaluated must never be allowed."""

    def test_missing_last_price_blocks(self):
        verdict = RiskEngine().check(order(), state(last_prices={}))
        assert verdict.decision is RiskDecision.BLOCK
        assert "cannot verify" in check_named(verdict, "price_band").message

    def test_zero_equity_blocks(self):
        verdict = RiskEngine().check(order(), state(equity=Decimal(0)))
        assert verdict.decision is RiskDecision.BLOCK

    def test_negative_equity_blocks(self):
        assert not RiskEngine().check(order(), state(equity=Decimal(-1))).allowed

    def test_unknown_adv_is_reported_not_failed(self):
        # A fresh listing has no history; blocking every order in it is wrong.
        verdict = RiskEngine().check(order(), state(adv={}))
        liquidity = check_named(verdict, "liquidity")
        assert liquidity.passed
        assert "not checked" in liquidity.message

    def test_zero_quantity_order_rejected_at_construction(self):
        with pytest.raises(ValueError, match="zero quantity"):
            order(quantity=Decimal(0))

    def test_non_positive_price_rejected_at_construction(self):
        with pytest.raises(ValueError, match="price"):
            order(price=Decimal(0))


class TestPreTradeLimits:
    def test_order_notional_boundary(self):
        limits = RiskLimits(max_order_notional=Decimal(10_000))
        engine = RiskEngine(limits)
        # Exactly at the limit passes; a rupee over does not.
        assert check_named(
            engine.check(order(quantity=Decimal(100), price=Decimal(100)), state()),
            "order_notional",
        ).passed
        assert not check_named(
            engine.check(order(quantity=Decimal(101), price=Decimal(100)), state()),
            "order_notional",
        ).passed

    def test_fat_finger_price_blocked(self):
        # A decimal-point slip: 1000 against a last price of 100.
        verdict = RiskEngine().check(order(price=Decimal(1000)), state())
        assert not check_named(verdict, "price_band").passed
        assert verdict.decision is RiskDecision.BLOCK

    def test_price_within_band_allowed(self):
        assert check_named(
            RiskEngine().check(order(price=Decimal("102")), state()), "price_band"
        ).passed

    def test_price_band_boundary(self):
        engine = RiskEngine(RiskLimits(price_band_pct=Decimal("0.05")))
        assert check_named(engine.check(order(price=Decimal(105)), state()), "price_band").passed
        assert not check_named(
            engine.check(order(price=Decimal("105.01")), state()), "price_band"
        ).passed

    def test_order_rate_limit(self):
        engine = RiskEngine(RiskLimits(max_orders_per_minute=5))
        assert check_named(engine.check(order(), state(orders_this_minute=4)), "order_rate").passed
        assert not check_named(
            engine.check(order(), state(orders_this_minute=5)), "order_rate"
        ).passed

    def test_open_order_cap(self):
        engine = RiskEngine(RiskLimits(max_open_orders=3))
        assert not check_named(engine.check(order(), state(open_orders=3)), "open_orders").passed

    def test_position_cap_counts_existing_holding(self):
        engine = RiskEngine(RiskLimits(max_position_pct=Decimal("0.10")))
        # Already holding 9% of NAV; another 2% would breach.
        verdict = engine.check(
            order(quantity=Decimal(200), price=Decimal(100)),
            state(positions={A: Decimal(90_000)}),
        )
        assert not check_named(verdict, "position_size").passed

    def test_position_cap_allows_reduction(self):
        engine = RiskEngine(RiskLimits(max_position_pct=Decimal("0.10")))
        # Selling out of an oversized position must be permitted.
        verdict = engine.check(
            order(quantity=Decimal(-500), price=Decimal(100)),
            state(positions={A: Decimal(90_000)}),
        )
        assert check_named(verdict, "position_size").passed

    def test_liquidity_participation_cap(self):
        engine = RiskEngine(RiskLimits(max_adv_participation=Decimal("0.05")))
        verdict = engine.check(
            order(quantity=Decimal(10_000), price=Decimal(100)),
            state(adv={A: Decimal(1_000_000)}),
        )
        assert not check_named(verdict, "liquidity").passed


class TestPortfolioLimits:
    def test_gross_exposure_cap(self):
        engine = RiskEngine(RiskLimits(max_gross_exposure_pct=Decimal("1.0")))
        verdict = engine.check(
            order(quantity=Decimal(1000), price=Decimal(100)),
            state(positions={B: Decimal(950_000)}),
        )
        assert not check_named(verdict, "gross_exposure").passed

    def test_net_exposure_cap(self):
        engine = RiskEngine(RiskLimits(max_net_exposure_pct=Decimal("0.5")))
        verdict = engine.check(
            order(quantity=Decimal(1000), price=Decimal(100)),
            state(positions={B: Decimal(450_000)}),
        )
        assert not check_named(verdict, "net_exposure").passed

    def test_market_neutral_book_passes_net_but_counts_gross(self):
        engine = RiskEngine(
            RiskLimits(max_net_exposure_pct=Decimal("0.1"), max_gross_exposure_pct=Decimal("2.0"))
        )
        # Long 400k, short 400k: net zero, gross 800k.
        verdict = engine.check(
            order(quantity=Decimal(100), price=Decimal(100)),
            state(positions={A: Decimal(400_000), B: Decimal(-400_000)}),
        )
        assert check_named(verdict, "net_exposure").passed

    def test_daily_loss_limit_halts(self):
        engine = RiskEngine(RiskLimits(daily_loss_limit_pct=Decimal("-0.03")))
        verdict = engine.check(order(), state(equity=Decimal(960_000)))
        assert not check_named(verdict, "daily_loss").passed
        assert verdict.decision is RiskDecision.BLOCK

    def test_daily_loss_within_limit_allowed(self):
        engine = RiskEngine(RiskLimits(daily_loss_limit_pct=Decimal("-0.03")))
        assert check_named(
            engine.check(order(), state(equity=Decimal(985_000))), "daily_loss"
        ).passed

    def test_correlation_cluster_cap(self):
        """Ten correlated PSU banks is one bet, not ten."""
        engine = RiskEngine(RiskLimits(max_cluster_pct=Decimal("0.30")))
        verdict = engine.check(
            order(quantity=Decimal(500), price=Decimal(100), cluster="psu_banks"),
            state(clusters={"psu_banks": Decimal(280_000)}),
        )
        assert not check_named(verdict, "cluster_concentration").passed

    def test_no_cluster_check_when_ungrouped(self):
        verdict = RiskEngine().check(order(), state())
        assert all(c.name != "cluster_concentration" for c in verdict.checks)


class TestDrawdownLadder:
    """Pre-committed de-risking — the operator is removed from the decision."""

    def test_no_scaling_above_the_first_rung(self):
        assert DrawdownLadder().scale_for(Decimal("-0.02")) == Decimal(1)

    def test_first_rung_halves(self):
        assert DrawdownLadder().scale_for(Decimal("-0.05")) == Decimal("0.50")

    def test_second_rung_quarters(self):
        assert DrawdownLadder().scale_for(Decimal("-0.08")) == Decimal("0.25")

    def test_third_rung_flattens_and_halts(self):
        ladder = DrawdownLadder()
        assert ladder.scale_for(Decimal("-0.10")) == Decimal(0)
        assert ladder.halts_at(Decimal("-0.10"))

    def test_deeper_than_the_last_rung_still_halts(self):
        assert DrawdownLadder().halts_at(Decimal("-0.35"))

    def test_engine_blocks_at_halt_depth(self):
        engine = RiskEngine()
        verdict = engine.check(order(), state(equity=Decimal(890_000)))
        assert verdict.decision is RiskDecision.BLOCK
        assert not check_named(verdict, "drawdown_ladder").passed

    def test_engine_reports_scale_without_blocking(self):
        """A -6% drawdown scales positions to half but keeps trading.

        The drawdown is set to have accumulated over prior sessions, so the
        daily loss limit is not also in play. Both firing together is correct
        behaviour — this test isolates the ladder.
        """
        engine = RiskEngine()
        verdict = engine.check(
            order(),
            state(
                equity=Decimal(940_000),
                peak_equity=Decimal(1_000_000),
                day_start_equity=Decimal(945_000),
            ),
        )
        assert verdict.allowed
        assert verdict.scale == Decimal("0.50")

    def test_same_day_drawdown_trips_both_limits(self):
        """A drawdown taken entirely today breaches the daily loss limit too."""
        engine = RiskEngine()
        verdict = engine.check(order(), state(equity=Decimal(940_000)))
        assert verdict.decision is RiskDecision.BLOCK
        assert not check_named(verdict, "daily_loss").passed
        # The ladder itself has not reached a halting rung at -6%.
        assert check_named(verdict, "drawdown_ladder").passed

    def test_rungs_must_be_ordered(self):
        with pytest.raises(ValueError, match="shallowest to deepest"):
            DrawdownLadder(
                rungs=(
                    LadderRung(Decimal("-0.10"), Decimal("0.25")),
                    LadderRung(Decimal("-0.05"), Decimal("0.50")),
                )
            )

    def test_deeper_rungs_cannot_scale_up(self):
        with pytest.raises(ValueError, match="never larger"):
            DrawdownLadder(
                rungs=(
                    LadderRung(Decimal("-0.05"), Decimal("0.25")),
                    LadderRung(Decimal("-0.10"), Decimal("0.50")),
                )
            )

    def test_positive_rung_rejected(self):
        with pytest.raises(ValueError, match="must be negative"):
            LadderRung(Decimal("0.05"), Decimal("0.5"))

    def test_scale_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="scale_to"):
            LadderRung(Decimal("-0.05"), Decimal("1.5"))

    def test_empty_ladder_rejected(self):
        with pytest.raises(ValueError, match="not a ladder"):
            DrawdownLadder(rungs=())


class TestVerdictReporting:
    def test_every_check_runs_even_after_a_breach(self):
        """Diagnosing at 2am is easier with the complete list."""
        engine = RiskEngine(RiskLimits(max_order_notional=Decimal(1)))
        verdict = engine.check(order(price=Decimal(1000)), state())
        # Both the size breach and the price-band breach are reported.
        assert len(verdict.breaches) >= 2

    def test_reasons_are_human_readable(self):
        engine = RiskEngine(RiskLimits(max_order_notional=Decimal(1)))
        reasons = engine.check(order(), state()).reasons
        assert any("order_notional" in r for r in reasons)

    def test_clean_order_has_no_breaches(self):
        verdict = RiskEngine().check(order(), state())
        assert verdict.allowed
        assert verdict.breaches == ()

    def test_format_renders(self):
        text = RiskEngine().check(order(), state()).format()
        assert "ALLOW" in text
        assert "gross_exposure" in text


class TestLimitValidation:
    def test_positive_daily_loss_limit_rejected(self):
        with pytest.raises(ValueError, match="must be negative"):
            RiskLimits(daily_loss_limit_pct=Decimal("0.03"))

    def test_non_positive_thresholds_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            RiskLimits(max_order_notional=Decimal(0))


class TestStateArithmetic:
    def test_gross_and_net(self):
        s = state(positions={A: Decimal(300_000), B: Decimal(-200_000)})
        assert s.gross_exposure == Decimal(500_000)
        assert s.net_exposure == Decimal(100_000)

    def test_drawdown_from_peak(self):
        s = state(equity=Decimal(900_000), peak_equity=Decimal(1_000_000))
        assert s.drawdown == Decimal("-0.1")

    def test_drawdown_zero_at_peak(self):
        assert state().drawdown == Decimal(0)

    def test_day_pnl(self):
        s = state(equity=Decimal(980_000), day_start_equity=Decimal(1_000_000))
        assert s.day_pnl_pct == Decimal("-0.02")

    def test_undefined_ratios_return_zero_not_infinity(self):
        # A zero denominator means "not measurable yet", not "infinitely bad".
        assert state(peak_equity=Decimal(0)).drawdown == Decimal(0)
        assert state(day_start_equity=Decimal(0)).day_pnl_pct == Decimal(0)

    def test_decision_flag(self):
        assert RiskDecision.BLOCK.is_blocked
        assert not RiskDecision.ALLOW.is_blocked


class TestFormatting:
    def test_check_without_numbers_renders(self):
        verdict = RiskEngine().check(order(), state(adv={}))
        # The liquidity check carries a message but no observed value.
        assert "not checked" in check_named(verdict, "liquidity").format()

    def test_verdict_reports_ladder_scale(self):
        verdict = RiskEngine().check(
            order(),
            state(
                equity=Decimal(940_000),
                peak_equity=Decimal(1_000_000),
                day_start_equity=Decimal(945_000),
            ),
        )
        assert "ladder scale" in verdict.format()
