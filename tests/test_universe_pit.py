"""Point-in-time universe construction — the M2 gate.

The decisive test is `TestSurvivorshipBias`: a company that was liquid in 2019
and delisted in 2022 must appear in the 2019 universe and be absent from the
2023 one. If that fails, every backtest built on this system is fiction.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from core.clock import UTC, as_decision_time
from data.store.panel import PANEL_SCHEMA, PanelStore
from data.universe.pit import UniverseBuilder, UniverseSpec

CLOSE_UTC = 10  # 15:30 IST
PUBLISH_UTC = 12  # ~18:00 IST, 2.5h later


def session_rows(
    session_date: date,
    names: dict[str, tuple[float, float]],
) -> pl.DataFrame:
    """One cross-section. `names` maps instrument_id -> (close, volume)."""
    event = datetime(session_date.year, session_date.month, session_date.day, CLOSE_UTC, tzinfo=UTC)
    receive = event + timedelta(hours=2, minutes=30)
    ids = list(names)
    closes = [names[i][0] for i in ids]
    volumes = [names[i][1] for i in ids]
    return pl.DataFrame(
        {
            "event_time": [event] * len(ids),
            "receive_time": [receive] * len(ids),
            "instrument_id": ids,
            "symbol": [i.split(":")[-1] for i in ids],
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
            "trades": [100] * len(ids),
        },
        schema_overrides=PANEL_SCHEMA,
    )


def business_days(start: date, count: int) -> list[date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


@pytest.fixture
def panel(tmp_path) -> PanelStore:
    return PanelStore(tmp_path, venue="NSE")


def decision_on(day: date, hour: int = PUBLISH_UTC + 1):
    return as_decision_time(datetime(day.year, day.month, day.day, hour, tzinfo=UTC))


class TestSurvivorshipBias:
    """The M2 gate. Everything else in the data layer is secondary to this."""

    @pytest.fixture
    def historical_panel(self, panel) -> PanelStore:
        """SURVIVOR trades throughout. DOOMED is liquid, then delists in 2021."""
        for day in business_days(date(2019, 1, 1), 120):
            names = {
                "NSE:SURVIVOR": (500.0, 100_000.0),
                "NSE:DOOMED": (400.0, 120_000.0),
                "NSE:TINY": (5.0, 100.0),
            }
            panel.write_session(day, session_rows(day, names))

        # DOOMED stops trading — it simply never appears again, exactly as a
        # delisted name behaves in the bhavcopy archive.
        for day in business_days(date(2022, 1, 3), 120):
            names = {
                "NSE:SURVIVOR": (700.0, 100_000.0),
                "NSE:NEWLISTING": (300.0, 90_000.0),
                "NSE:TINY": (6.0, 100.0),
            }
            panel.write_session(day, session_rows(day, names))
        return panel

    def test_delisted_name_present_in_its_own_era(self, historical_panel):
        universe = UniverseBuilder(historical_panel).build(
            decision_on(date(2019, 6, 1)), UniverseSpec(top_n=10, min_sessions=20)
        )
        assert "NSE:DOOMED" in universe

    def test_delisted_name_absent_later(self, historical_panel):
        universe = UniverseBuilder(historical_panel).build(
            decision_on(date(2022, 6, 1)), UniverseSpec(top_n=10, min_sessions=20)
        )
        assert "NSE:DOOMED" not in universe

    def test_later_listing_absent_from_earlier_universe(self, historical_panel):
        universe = UniverseBuilder(historical_panel).build(
            decision_on(date(2019, 6, 1)), UniverseSpec(top_n=10, min_sessions=20)
        )
        assert "NSE:NEWLISTING" not in universe

    def test_membership_genuinely_differs_across_eras(self, historical_panel):
        builder = UniverseBuilder(historical_panel)
        spec = UniverseSpec(top_n=10, min_sessions=20)
        early = builder.build(decision_on(date(2019, 6, 1)), spec)
        late = builder.build(decision_on(date(2022, 6, 1)), spec)
        assert set(early.members) != set(late.members)
        assert early.turnover_against(late) > 0


class TestPointInTimeDiscipline:
    def test_unpublished_session_not_visible(self, panel):
        for day in business_days(date(2024, 1, 1), 60):
            panel.write_session(day, session_rows(day, {"NSE:A": (100.0, 50_000.0)}))

        last = business_days(date(2024, 1, 1), 60)[-1]
        spec = UniverseSpec(top_n=5, min_sessions=1, min_median_value=0.0)

        # Before publication the final session is invisible...
        before = UniverseBuilder(panel).build(decision_on(last, hour=PUBLISH_UTC - 1), spec)
        # ...and after it, it counts.
        after = UniverseBuilder(panel).build(decision_on(last, hour=PUBLISH_UTC + 1), spec)
        assert before.as_of < after.as_of

        rows_before = panel.view(as_of=decision_on(last, hour=PUBLISH_UTC - 1))
        rows_after = panel.view(as_of=decision_on(last, hour=PUBLISH_UTC + 1))
        assert rows_after.height == rows_before.height + 1

    def test_empty_history_yields_empty_universe(self, panel):
        panel.write_session(
            date(2024, 1, 2), session_rows(date(2024, 1, 2), {"NSE:A": (100.0, 5000.0)})
        )
        universe = UniverseBuilder(panel).build(decision_on(date(2020, 1, 1)))
        assert len(universe) == 0

    def test_schedule_uses_prior_sessions_only(self, panel):
        days = business_days(date(2024, 1, 1), 80)
        for day in days:
            panel.write_session(day, session_rows(day, {"NSE:A": (100.0, 50_000.0)}))
        spec = UniverseSpec(top_n=5, min_sessions=10, min_median_value=0.0)
        schedule = UniverseBuilder(panel).build_schedule([days[40], days[70]], spec)
        assert len(schedule) == 2
        assert all(len(u) == 1 for u in schedule.values())


class TestSelectionRules:
    @pytest.fixture
    def liquid_panel(self, panel) -> PanelStore:
        for day in business_days(date(2024, 1, 1), 80):
            panel.write_session(
                day,
                session_rows(
                    day,
                    {
                        "NSE:BIG": (1000.0, 100_000.0),  # value 1e8
                        "NSE:MID": (500.0, 50_000.0),  # value 2.5e7
                        "NSE:SMALL": (100.0, 20_000.0),  # value 2e6
                        "NSE:PENNY": (3.0, 900_000.0),  # liquid but sub-₹20
                    },
                ),
            )
        return panel

    def test_ranked_by_traded_value(self, liquid_panel):
        universe = UniverseBuilder(liquid_panel).build(
            decision_on(date(2024, 4, 1)), UniverseSpec(top_n=2, min_sessions=20)
        )
        assert universe.members == ("NSE:BIG", "NSE:MID")

    def test_top_n_respected(self, liquid_panel):
        universe = UniverseBuilder(liquid_panel).build(
            decision_on(date(2024, 4, 1)), UniverseSpec(top_n=1, min_sessions=20)
        )
        assert len(universe) == 1

    def test_penny_stock_excluded(self, liquid_panel):
        universe = UniverseBuilder(liquid_panel).build(
            decision_on(date(2024, 4, 1)),
            UniverseSpec(top_n=10, min_sessions=20, min_price=20.0),
        )
        assert "NSE:PENNY" not in universe

    def test_illiquid_excluded_by_value_floor(self, liquid_panel):
        universe = UniverseBuilder(liquid_panel).build(
            decision_on(date(2024, 4, 1)),
            UniverseSpec(top_n=10, min_sessions=20, min_median_value=1e7),
        )
        assert "NSE:SMALL" not in universe
        assert "NSE:BIG" in universe

    def test_recent_listing_excluded_by_min_sessions(self, panel):
        days = business_days(date(2024, 1, 1), 80)
        for i, day in enumerate(days):
            names = {"NSE:OLD": (100.0, 50_000.0)}
            if i >= 75:  # lists near the end
                names["NSE:JUSTLISTED"] = (100.0, 90_000.0)
            panel.write_session(day, session_rows(day, names))
        universe = UniverseBuilder(panel).build(
            decision_on(days[-1]), UniverseSpec(top_n=10, min_sessions=40)
        )
        assert "NSE:JUSTLISTED" not in universe
        assert "NSE:OLD" in universe

    def test_liquidity_reported_per_member(self, liquid_panel):
        universe = UniverseBuilder(liquid_panel).build(
            decision_on(date(2024, 4, 1)), UniverseSpec(top_n=2, min_sessions=20)
        )
        assert universe.liquidity["NSE:BIG"] > universe.liquidity["NSE:MID"]

    def test_selection_is_deterministic(self, liquid_panel):
        # Ties must not reorder between runs (§14.1.1).
        builder = UniverseBuilder(liquid_panel)
        spec = UniverseSpec(top_n=3, min_sessions=20)
        runs = {builder.build(decision_on(date(2024, 4, 1)), spec).members for _ in range(5)}
        assert len(runs) == 1


class TestSpecValidation:
    def test_top_n_must_be_positive(self):
        with pytest.raises(ValueError, match="top_n"):
            UniverseSpec(top_n=0)

    def test_min_sessions_cannot_exceed_lookback(self):
        with pytest.raises(ValueError, match="min_sessions"):
            UniverseSpec(lookback_days=30, min_sessions=60)


class TestTurnover:
    def test_identical_membership_has_zero_turnover(self, panel):
        for day in business_days(date(2024, 1, 1), 80):
            panel.write_session(day, session_rows(day, {"NSE:A": (100.0, 50_000.0)}))
        builder = UniverseBuilder(panel)
        spec = UniverseSpec(top_n=5, min_sessions=20, min_median_value=0.0)
        a = builder.build(decision_on(date(2024, 4, 1)), spec)
        b = builder.build(decision_on(date(2024, 4, 2)), spec)
        assert a.turnover_against(b) == 0.0

    def test_disjoint_membership_has_full_turnover(self, panel):
        for day in business_days(date(2024, 1, 1), 80):
            panel.write_session(day, session_rows(day, {"NSE:A": (100.0, 50_000.0)}))
        spec = UniverseSpec(top_n=5, min_sessions=20, min_median_value=0.0)
        current = UniverseBuilder(panel).build(decision_on(date(2024, 4, 1)), spec)
        empty = UniverseBuilder(panel).build(decision_on(date(2020, 1, 1)), spec)
        assert current.turnover_against(empty) == 1.0
