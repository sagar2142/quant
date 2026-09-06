"""The announced board-meeting calendar — MASTER_PLAN §1.1, §3.3.

Named for the calendar, not for `core.events`, which `tests/test_events.py`
covers.

The failure this feed is exposed to is not a wrong number, it is a false quiet.
A calendar that cannot be read, or one read from a fortnight-old file, reports
"nothing announced" — which is indistinguishable from a genuinely empty week
and is exactly the answer that gets a position held through a print.
"""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from core.clock import UTC
from data.feeds.nse_events import (
    EventFormatError,
    is_results_event,
    parse_events,
    resolve,
)
from data.store.events import EventStore
from ops.alerts import Severity
from ops.watch import Rule, RuleKind, WatchContext, evaluate

OBSERVED = date(2026, 9, 5)

EVENT = {
    "symbol": "RELIANCE",
    "company": "Reliance Industries Limited",
    "date": "12-Sep-2026",
    "purpose": "Financial Results",
    "bm_desc": "To consider and approve the quarterly results",
}
OTHER = {
    "symbol": "AGROPHOS",
    "company": "Agro Phos India Limited",
    "date": "07-Sep-2026",
    "purpose": "Other business matters",
    "bm_desc": "To consider other business matters",
}


def payload(*rows: dict) -> bytes:
    return json.dumps(list(rows)).encode()


class TestParsing:
    def test_it_reads_an_event(self) -> None:
        frame = parse_events(payload(EVENT), OBSERVED)
        assert frame.height == 1
        assert frame["event_date"][0] == date(2026, 9, 12)
        assert frame["observed_at"][0] == OBSERVED

    def test_a_results_meeting_is_marked(self) -> None:
        frame = parse_events(payload(EVENT), OBSERVED)
        assert bool(frame["is_results"][0]) is True

    def test_an_unrelated_meeting_is_not(self) -> None:
        frame = parse_events(payload(OTHER), OBSERVED)
        assert bool(frame["is_results"][0]) is False

    def test_a_compound_purpose_still_counts(self) -> None:
        """NSE writes 'Financial Results/Other business matters'. Matching the
        purpose exactly would miss most of the ones that matter."""
        assert is_results_event("Financial Results/Other business matters")
        assert is_results_event("Dividend/Financial Results")

    def test_a_row_without_a_date_is_dropped(self) -> None:
        assert parse_events(payload({**EVENT, "date": ""}), OBSERVED).is_empty()

    def test_a_row_without_a_symbol_is_dropped(self) -> None:
        assert parse_events(payload({**EVENT, "symbol": ""}), OBSERVED).is_empty()

    def test_html_is_refused(self) -> None:
        with pytest.raises(EventFormatError, match="HTML"):
            parse_events(b"<html>blocked</html>", OBSERVED)

    def test_a_changed_layout_is_refused_not_emptied(self) -> None:
        """An empty calendar reads as 'nobody is reporting', which is the one
        answer that must never be produced by a parsing failure."""
        with pytest.raises(EventFormatError, match="missing every one"):
            parse_events(payload({"companyId": 1}), OBSERVED)

    def test_symbols_are_upper_cased(self) -> None:
        frame = parse_events(payload({**EVENT, "symbol": "reliance"}), OBSERVED)
        assert frame["symbol"][0] == "RELIANCE"


class TestResolution:
    """The calendar carries no ISIN, so this join is the weak point."""

    def test_it_attaches_an_instrument(self) -> None:
        events = parse_events(payload(EVENT), OBSERVED)
        found = resolve(events, {"RELIANCE": "NSE:INE002A01018"})
        assert found.rows["instrument_id"][0] == "NSE:INE002A01018"
        assert found.unresolved == ()

    def test_an_unknown_symbol_is_reported_not_dropped(self) -> None:
        """An event nobody can join to a position is still an event. A
        calendar that hides its gaps stops being trusted."""
        events = parse_events(payload(EVENT, OTHER), OBSERVED)
        found = resolve(events, {"RELIANCE": "NSE:INE002A01018"})
        assert found.unresolved == ("AGROPHOS",)
        assert found.rows.height == 1

    def test_coverage_is_the_resolved_share(self) -> None:
        events = parse_events(payload(EVENT, OTHER), OBSERVED)
        found = resolve(events, {"RELIANCE": "NSE:INE002A01018"})
        assert found.coverage == pytest.approx(0.5)

    def test_an_empty_calendar_resolves_to_nothing(self) -> None:
        found = resolve(pl.DataFrame(schema=parse_events(b"[]", OBSERVED).schema), {})
        assert found.rows.is_empty()
        assert found.unresolved == ()


class TestStore:
    def stored(self, tmp_path) -> EventStore:
        store = EventStore(tmp_path)
        events = parse_events(payload(EVENT, OTHER), OBSERVED)
        store.write(OBSERVED, resolve(events, {"RELIANCE": "NSE:INE002A01018"}).rows)
        return store

    def test_it_returns_events_in_the_window(self, tmp_path) -> None:
        window = self.stored(tmp_path).upcoming(OBSERVED, within_days=14)
        assert window.rows.height == 1
        assert window.observed_at == OBSERVED

    def test_an_event_beyond_the_horizon_is_excluded(self, tmp_path) -> None:
        assert self.stored(tmp_path).upcoming(OBSERVED, within_days=3).rows.is_empty()

    def test_a_past_event_is_excluded(self, tmp_path) -> None:
        assert self.stored(tmp_path).upcoming(date(2026, 9, 20), within_days=14).rows.is_empty()

    def test_results_only_filters(self, tmp_path) -> None:
        store = EventStore(tmp_path)
        events = parse_events(payload(EVENT, OTHER), OBSERVED)
        store.write(OBSERVED, resolve(events, {"RELIANCE": "X", "AGROPHOS": "Y"}).rows)
        assert store.upcoming(OBSERVED, 14).rows.height == 2
        assert store.upcoming(OBSERVED, 14, results_only=True).rows.height == 1

    def test_it_reports_its_own_age(self, tmp_path) -> None:
        """A calendar is only as good as its age, and a stale one reports
        quiet where there is none."""
        window = self.stored(tmp_path).upcoming(date(2026, 9, 9), within_days=14)
        assert window.age_days == 4

    def test_an_empty_store_holds_no_calendar(self, tmp_path) -> None:
        window = EventStore(tmp_path).upcoming(OBSERVED)
        assert window.observed_at is None
        assert window.rows.is_empty()

    def test_it_answers_for_held_instruments(self, tmp_path) -> None:
        window = self.stored(tmp_path).upcoming(OBSERVED, within_days=14)
        assert window.for_instruments({"NSE:INE002A01018"}).height == 1
        assert window.for_instruments({"NSE:OTHER"}).height == 0

    def test_a_frame_missing_columns_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            EventStore(tmp_path).write(OBSERVED, pl.DataFrame({"symbol": ["X"]}))


class TestReportsRule:
    """A rule that cannot read the calendar must not report quiet."""

    def rule(self, threshold: int = 3) -> Rule:
        from decimal import Decimal

        return Rule(
            rule_id="reports-soon",
            kind=RuleKind.REPORTS_WITHIN_DAYS,
            subject="RELIANCE",
            threshold=Decimal(threshold),
            severity=Severity.WARN,
        )

    def context(self, **kw) -> WatchContext:
        from datetime import datetime

        base = {
            "as_of": datetime(2026, 9, 5, tzinfo=UTC),
            "days_to_results": {"RELIANCE": 2},
            "days_to_results_known": True,
            # The calendar was read far enough to answer a 3-day rule. Without
            # this the rule is unevaluable rather than quiet, which is correct
            # and is what `TestHorizonIsNotSilentlyCapped` covers.
            "results_horizon_days": 30,
        }
        return WatchContext(**{**base, **kw})

    def test_it_fires_inside_the_window(self) -> None:
        found = evaluate([self.rule()], self.context())
        assert found[0].fired

    def test_it_does_not_fire_outside(self) -> None:
        found = evaluate([self.rule()], self.context(days_to_results={"RELIANCE": 20}))
        assert not found[0].fired
        assert found[0].evaluable

    def test_no_calendar_is_unevaluable_not_quiet(self) -> None:
        """The distinction the whole feed turns on: 'nothing is announced' and
        'I could not find out' are opposite answers."""
        found = evaluate([self.rule()], self.context(days_to_results_known=False))
        assert not found[0].evaluable
        assert "no calendar" in found[0].reason

    def test_a_name_with_no_meeting_is_evaluable(self) -> None:
        """The calendar was read and said nothing about this name. That is a
        measurement, unlike the case above."""
        found = evaluate([self.rule()], self.context(days_to_results={}))
        assert found[0].evaluable
        assert not found[0].fired
        assert "no announced results meeting" in found[0].reason

    def test_the_kind_needs_a_subject(self) -> None:
        from decimal import Decimal

        with pytest.raises(ValueError, match="needs a subject"):
            Rule(
                rule_id="r",
                kind=RuleKind.REPORTS_WITHIN_DAYS,
                threshold=Decimal(3),
            )

    def test_every_kind_still_has_an_evaluator(self) -> None:
        from ops.watch import _EVALUATORS

        assert set(_EVALUATORS) == set(RuleKind)


class TestHorizonIsNotSilentlyCapped:
    """A rule asking further ahead than the calendar was read.

    The tempting answer — "no meeting announced" — is a measurement nobody
    took, and it is the same false quiet an unreadable calendar produces,
    reached from the other direction.
    """

    def rule(self, threshold: int):
        from decimal import Decimal

        return Rule(
            rule_id="far",
            kind=RuleKind.REPORTS_WITHIN_DAYS,
            subject="FARCO",
            threshold=Decimal(threshold),
        )

    def context(self, horizon: int):
        from datetime import datetime

        return WatchContext(
            as_of=datetime(2026, 9, 5, tzinfo=UTC),
            days_to_results={},
            days_to_results_known=True,
            results_horizon_days=horizon,
        )

    def test_a_rule_beyond_the_horizon_is_unevaluable(self) -> None:
        found = evaluate([self.rule(60)], self.context(30))
        assert not found[0].evaluable
        assert "read only 30 days ahead" in found[0].reason

    def test_a_rule_inside_the_horizon_is_answered(self) -> None:
        found = evaluate([self.rule(10)], self.context(30))
        assert found[0].evaluable
        assert not found[0].fired

    def test_the_boundary_is_inclusive(self) -> None:
        assert evaluate([self.rule(30)], self.context(30))[0].evaluable

    def test_a_zero_horizon_answers_nothing(self) -> None:
        """No calendar was read, so no calendar rule is quiet."""
        assert not evaluate([self.rule(3)], self.context(0))[0].evaluable

    def test_the_runner_reads_far_enough_for_its_rules(self) -> None:
        from apps.cli.watch import results_horizon

        assert results_horizon([self.rule(3), self.rule(12)]) == 12

    def test_it_never_asks_beyond_the_cap(self) -> None:
        from apps.cli.watch import MAX_RESULTS_HORIZON, results_horizon

        assert results_horizon([self.rule(999)]) == MAX_RESULTS_HORIZON

    def test_no_calendar_rules_means_no_calendar_read(self) -> None:
        """Reading a calendar nothing asks about is a wasted fetch."""
        from decimal import Decimal

        from apps.cli.watch import results_horizon

        other = Rule(rule_id="d", kind=RuleKind.DRAWDOWN_BEYOND, threshold=Decimal("-0.05"))
        assert results_horizon([other]) == 0

    def test_a_disabled_rule_does_not_widen_the_horizon(self) -> None:
        from decimal import Decimal

        from apps.cli.watch import results_horizon

        off = Rule(
            rule_id="off",
            kind=RuleKind.REPORTS_WITHIN_DAYS,
            subject="X",
            threshold=Decimal(25),
            enabled=False,
        )
        assert results_horizon([off, self.rule(4)]) == 4
