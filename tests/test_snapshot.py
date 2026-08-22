"""Derived paper-book state — MASTER_PLAN §8, §12.7.

Every test here exists because a monitoring surface was reporting a number it
had not measured. `/vitals` returned literal zeros and the console's Risk
screen hardcoded `observed: 0, passed: true`, so a book three days stale with a
position over its cap displayed as live, flat and entirely within limits.

The distinction under test throughout is **null versus zero**. Zero is a
measurement. Null is the absence of one. Collapsing them is what made the bug
invisible, so most of these assert `is None` rather than a value.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from apps.api.limits import _verdict, limit_rows
from apps.api.snapshot import book_snapshot
from core.clock import UTC, utc_now
from core.instruments import InstrumentId
from core.orders import Side
from engine.accounting import CostBreakdown, Fill, Portfolio, Position
from trading.paper.state import PaperState, PaperStateStore
from trading.risk.limits import RiskLimits

RELIANCE = InstrumentId("NSE:INE002A01018")
TCS = InstrumentId("NSE:INE467B01029")


def write_state(  # noqa: PLR0913 - a state fixture has as many knobs as the state
    root,
    *,
    cash: Decimal = Decimal(100_000),
    positions: dict[InstrumentId, Position] | None = None,
    peak_equity: Decimal = Decimal(1_000_000),
    cycles: int = 1,
    cycle_age: timedelta = timedelta(hours=1),
    equity_rows: list[Decimal] | None = None,
) -> PaperStateStore:
    store = PaperStateStore(root)
    portfolio = Portfolio(cash=cash)
    for instrument_id, position in (positions or {}).items():
        portfolio.positions[instrument_id] = position
    store.save(
        PaperState(
            strategy_id="test",
            portfolio=portfolio,
            peak_equity=peak_equity,
            cycles=cycles,
            last_cycle_at=utc_now() - cycle_age,
            last_session=date(2026, 7, 31),
        )
    )
    for index, equity in enumerate(equity_rows or []):
        store.append_equity(date(2026, 7, 1) + timedelta(days=index), equity, cash, Decimal(0))
    return store


def held(instrument_id: InstrumentId, quantity: int, price: str) -> Position:
    return Position(
        instrument_id=instrument_id,
        quantity=Decimal(quantity),
        average_price=Decimal(price),
    )


class TestAbsentBook:
    def test_no_state_is_absent_not_flat(self, tmp_path):
        """ "Not started" and "started and flat" call for different actions."""
        snapshot = book_snapshot(tmp_path)
        assert snapshot.present is False
        assert snapshot.equity is None
        assert snapshot.drawdown is None

    def test_an_absent_book_reports_no_staleness_rather_than_zero(self, tmp_path):
        """Zero seconds stale would mean a cycle just finished."""
        assert book_snapshot(tmp_path).staleness_seconds is None

    def test_corrupt_state_reads_as_absent(self, tmp_path):
        store = PaperStateStore(tmp_path)
        store.state_path.parent.mkdir(parents=True, exist_ok=True)
        store.state_path.write_text("{not json", encoding="utf-8")
        assert book_snapshot(tmp_path).present is False


class TestDrawdown:
    def test_drawdown_is_negative_like_the_ladder_rungs(self, tmp_path):
        """`LadderRung.drawdown_pct` is validated negative, and both the engine
        and the console compare against it directly. A positive convention here
        leaves every rung unlit through an arbitrarily deep drawdown."""
        write_state(tmp_path, cash=Decimal(900_000), peak_equity=Decimal(1_000_000))
        snapshot = book_snapshot(tmp_path)
        assert snapshot.drawdown is not None
        assert snapshot.drawdown < 0
        assert snapshot.drawdown == pytest.approx(Decimal("-0.10"))

    def test_a_book_at_its_high_is_flat_not_positive(self, tmp_path):
        write_state(tmp_path, cash=Decimal(1_000_000), peak_equity=Decimal(1_000_000))
        assert book_snapshot(tmp_path).drawdown == Decimal(0)

    def test_a_book_above_its_recorded_peak_does_not_report_a_gain(self, tmp_path):
        """Clamped: a new high is a zero drawdown, never a positive one."""
        write_state(tmp_path, cash=Decimal(1_200_000), peak_equity=Decimal(1_000_000))
        assert book_snapshot(tmp_path).drawdown == Decimal(0)


class TestDayChange:
    def test_one_cycle_has_no_day_change(self, tmp_path):
        """A single cycle has nothing to difference against. Comparing it to
        starting capital would report the whole book's P&L as one session's."""
        write_state(tmp_path, equity_rows=[Decimal(998_000)])
        snapshot = book_snapshot(tmp_path)
        assert snapshot.day_pnl is None
        assert snapshot.day_pnl_pct is None

    def test_two_cycles_difference_the_last_two(self, tmp_path):
        write_state(tmp_path, equity_rows=[Decimal(1_000_000), Decimal(990_000)])
        snapshot = book_snapshot(tmp_path)
        assert snapshot.day_pnl == Decimal(-10_000)
        assert snapshot.day_pnl_pct == pytest.approx(Decimal("-0.01"))

    def test_a_flat_session_is_zero_not_null(self, tmp_path):
        """Zero here is a real measurement and must not read as absent."""
        write_state(tmp_path, equity_rows=[Decimal(500_000), Decimal(500_000)])
        assert book_snapshot(tmp_path).day_pnl == Decimal(0)


class TestExposure:
    def test_exposures_are_fractions_of_equity(self, tmp_path):
        write_state(
            tmp_path,
            cash=Decimal(50_000),
            positions={RELIANCE: held(RELIANCE, 100, "500"), TCS: held(TCS, 100, "500")},
        )
        marks = {RELIANCE: Decimal(500), TCS: Decimal(500)}
        snapshot = book_snapshot(tmp_path, marks)
        # 100,000 of positions against 150,000 of equity.
        assert snapshot.gross_exposure == pytest.approx(Decimal("0.666666"), abs=1e-4)
        assert snapshot.largest_position_pct == pytest.approx(Decimal("0.333333"), abs=1e-4)

    def test_a_flat_book_reports_zero_exposure_not_null(self, tmp_path):
        write_state(tmp_path, cash=Decimal(100_000))
        assert book_snapshot(tmp_path).gross_exposure == Decimal(0)

    def test_non_positive_equity_reports_no_exposure(self, tmp_path):
        """Dividing by it yields a number with no meaning; an infinite
        exposure on screen is less informative than admitting it cannot say."""
        write_state(tmp_path, cash=Decimal(0))
        assert book_snapshot(tmp_path).gross_exposure is None


class TestStaleness:
    def test_staleness_counts_from_the_last_cycle(self, tmp_path):
        write_state(tmp_path, cycle_age=timedelta(hours=3))
        snapshot = book_snapshot(tmp_path)
        assert snapshot.staleness_seconds == pytest.approx(3 * 3600, abs=30)

    def test_a_future_cycle_does_not_report_negative_staleness(self, tmp_path):
        write_state(tmp_path, cycle_age=timedelta(hours=-2))
        staleness = book_snapshot(tmp_path).staleness_seconds
        assert staleness is not None
        assert staleness >= 0


class TestLimitRows:
    def test_per_order_limits_report_no_observation(self, tmp_path):
        """No book at rest has a value for a fat-finger price band. Reporting
        zero would say the budget is untouched when the question does not
        apply."""
        write_state(tmp_path, cash=Decimal(100_000))
        rows = {r.name: r for r in limit_rows(RiskLimits(), book_snapshot(tmp_path))}
        for name in ("order_notional", "price_band", "order_rate", "open_orders", "liquidity"):
            assert rows[name].observed is None, name
            assert rows[name].passed is None, name

    def test_portfolio_limits_report_real_observations(self, tmp_path):
        write_state(
            tmp_path,
            cash=Decimal(50_000),
            positions={RELIANCE: held(RELIANCE, 100, "500")},
        )
        rows = {
            r.name: r
            for r in limit_rows(RiskLimits(), book_snapshot(tmp_path, {RELIANCE: Decimal(500)}))
        }
        assert rows["gross_exposure"].observed == pytest.approx(0.5)
        assert rows["gross_exposure"].passed is True

    def test_an_absent_book_observes_nothing_at_all(self, tmp_path):
        rows = limit_rows(RiskLimits(), book_snapshot(tmp_path))
        assert all(r.observed is None for r in rows)
        assert all(r.passed is None for r in rows)
        # The thresholds are still reported: the engine enforces them whether
        # or not a book exists to measure against.
        assert all(r.threshold for r in rows)

    def test_every_limit_is_still_listed(self, tmp_path):
        assert len(limit_rows(RiskLimits())) == 10


class TestVerdicts:
    def test_a_breach_of_a_ceiling_fails(self):
        assert _verdict("gross_exposure", 1.6, 1.5) is False

    def test_daily_loss_is_a_floor_not_a_ceiling(self):
        """-2% passes a -3% limit. Comparing it as a ceiling would fail the
        limit on every losing day and pass it on every winning one."""
        assert _verdict("daily_loss", -0.02, -0.03) is True
        assert _verdict("daily_loss", -0.04, -0.03) is False

    def test_a_profitable_session_passes_the_loss_limit(self):
        assert _verdict("daily_loss", 0.05, -0.03) is True

    def test_net_exposure_is_bounded_in_both_directions(self):
        """A large short book breaches the tilt limit exactly as a long one
        does; comparing the signed value would let it through."""
        assert _verdict("net_exposure", -1.2, 1.0) is False
        assert _verdict("net_exposure", 0.9, 1.0) is True

    def test_an_unmeasured_limit_has_no_verdict(self):
        assert _verdict("gross_exposure", None, 1.5) is None

    def test_exactly_at_the_threshold_passes(self):
        assert _verdict("position_size", 0.10, 0.10) is True

    def test_a_hair_over_the_threshold_fails(self):
        """The real book carries a name at 10.002% after post-entry drift.
        Widening the comparison to hide it would restore the bug."""
        assert _verdict("position_size", 0.10002, 0.10) is False


class TestEquityLog:
    def test_history_is_read_oldest_first(self, tmp_path):
        write_state(tmp_path, equity_rows=[Decimal(100), Decimal(200), Decimal(300)])
        rows = PaperStateStore(tmp_path).equity_history()
        assert [r["equity"] for r in rows] == ["100", "200", "300"]

    def test_a_blank_line_does_not_break_the_log(self, tmp_path):
        store = write_state(tmp_path, equity_rows=[Decimal(100)])
        with store.log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        assert len(store.equity_history()) == 1

    def test_rows_carry_the_fields_the_console_plots(self, tmp_path):
        store = write_state(tmp_path, equity_rows=[Decimal(100)])
        row = store.equity_history()[0]
        assert set(row) >= {"session", "equity", "cash", "fees_paid"}
        assert json.loads(json.dumps(row)) == row


class TestClockSanity:
    def test_last_cycle_is_stored_in_utc(self, tmp_path):
        write_state(tmp_path)
        snapshot = book_snapshot(tmp_path)
        assert snapshot.last_cycle_at is not None
        assert snapshot.last_cycle_at.tzinfo is not None
        assert snapshot.last_cycle_at.astimezone(UTC) == snapshot.last_cycle_at

    def test_a_naive_cycle_time_is_refused_by_the_state(self):
        with pytest.raises(Exception):  # noqa: B017 - require_utc's own type
            PaperState(
                strategy_id="t",
                portfolio=Portfolio(cash=Decimal(0)),
                peak_equity=Decimal(1),
                last_cycle_at=datetime(2026, 1, 1),
            )


class TestFillLog:
    """The blotter had no source at all. `/fills` served nothing because
    nothing wrote it, and the console answered "What did I trade?" with "No
    trades today." after eleven real fills.
    """

    def fill(self, instrument_id=RELIANCE, side=Side.BUY, quantity="10", price="500"):
        return Fill(
            instrument_id=instrument_id,
            side=side,
            quantity=Decimal(quantity),
            price=Decimal(price),
            costs=CostBreakdown(brokerage=Decimal("1.50")),
            event_time=utc_now(),
        )

    def test_a_fill_survives_to_the_log(self, tmp_path):
        store = PaperStateStore(tmp_path)
        store.append_fill(date(2026, 7, 31), self.fill())
        rows = store.fill_history()
        assert len(rows) == 1
        assert rows[0]["instrument_id"] == str(RELIANCE)
        assert rows[0]["side"] == "BUY"

    def test_money_round_trips_as_a_string(self, tmp_path):
        """Decimal-as-string, never float: the blotter is what a reconciliation
        dispute is argued from, and 0.1 + 0.2 has no place in it."""
        store = PaperStateStore(tmp_path)
        store.append_fill(date(2026, 7, 31), self.fill(price="1234.55"))
        assert store.fill_history()[0]["price"] == "1234.55"
        assert Decimal(store.fill_history()[0]["price"]) == Decimal("1234.55")

    def test_fills_accumulate_across_cycles(self, tmp_path):
        store = PaperStateStore(tmp_path)
        for _ in range(3):
            store.append_fill(date(2026, 7, 31), self.fill())
        assert len(store.fill_history()) == 3

    def test_the_limit_keeps_the_most_recent(self, tmp_path):
        store = PaperStateStore(tmp_path)
        for i in range(5):
            store.append_fill(date(2026, 7, 31), self.fill(price=str(100 + i)))
        recent = store.fill_history(limit=2)
        assert [r["price"] for r in recent] == ["103", "104"]

    def test_a_malformed_line_is_skipped_not_fatal(self, tmp_path):
        """A blotter missing one row beats a screen that will not render."""
        store = PaperStateStore(tmp_path)
        store.append_fill(date(2026, 7, 31), self.fill())
        with store.fills_path.open("a", encoding="utf-8") as handle:
            handle.write("{ this is not json\n")
        store.append_fill(date(2026, 7, 31), self.fill())
        assert len(store.fill_history()) == 2

    def test_no_log_is_an_empty_blotter_not_an_error(self, tmp_path):
        assert PaperStateStore(tmp_path).fill_history() == []

    def test_the_log_stores_identity_not_a_ticker(self, tmp_path):
        """A symbol is not identity (§3.3). The console resolves today's ticker
        at render time, so a rename cannot make the blotter disagree with every
        other screen."""
        store = PaperStateStore(tmp_path)
        store.append_fill(date(2026, 7, 31), self.fill())
        assert store.fill_history()[0]["symbol"] == ""
        assert store.fill_history()[0]["instrument_id"] == str(RELIANCE)

    def test_the_log_is_separate_from_the_equity_log(self, tmp_path):
        """One is per-cycle, the other per-fill; sharing a file would mean
        rewriting history to append either."""
        store = PaperStateStore(tmp_path)
        assert store.fills_path != store.log_path
