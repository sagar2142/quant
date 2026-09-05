"""Alert rules — MASTER_PLAN §12.7, §20.

Delivery existed; nothing ever decided that something was worth saying. A
staleness light that only shows while you are looking at it is not a warning.

The three properties below are what separate an alerting system from a
decorative one, and each has a way of failing that looks like working:

  - a rule that **cannot be evaluated** must not look like one that passed
  - a condition that **persists** must not page every evening until it clears
  - a rule that **could not run** must not be recorded as having cleared
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import pytest

from core.clock import UTC
from ops.alerts import Severity
from ops.watch import Rule, RuleKind, WatchContext, evaluate

NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)


def context(**overrides) -> WatchContext:
    base = {
        "as_of": NOW,
        "closes": {"RELIANCE": Decimal("2500"), "NSE:INE002A01018": Decimal("2500")},
        "previous_closes": {"RELIANCE": Decimal("2400")},
        "drawdown": Decimal("-0.03"),
        "feed_age_hours": {"nse": 20.0},
        "risk_observations": {"gross_exposure": Decimal("0.85")},
    }
    return WatchContext(**{**base, **overrides})


def rule(kind: RuleKind, **overrides) -> Rule:
    base = {"rule_id": f"r-{kind.value}", "kind": kind, "threshold": Decimal(0)}
    return Rule(**{**base, **overrides})


class TestRuleValidation:
    def test_a_rule_needs_an_id(self) -> None:
        with pytest.raises(ValueError, match="needs an id"):
            Rule(rule_id="  ", kind=RuleKind.DRAWDOWN_BEYOND)

    def test_a_price_rule_needs_a_subject(self) -> None:
        """A close_below with no name is not a rule, it is a typo that would
        sit in the file looking like coverage."""
        with pytest.raises(ValueError, match="needs a subject"):
            Rule(rule_id="r", kind=RuleKind.CLOSE_BELOW, threshold=Decimal(100))

    def test_a_book_wide_rule_needs_none(self) -> None:
        assert Rule(rule_id="r", kind=RuleKind.DRAWDOWN_BEYOND).subject == ""

    def test_every_kind_has_an_evaluator(self) -> None:
        """A kind added without one would be a rule that silently never fires."""
        from ops.watch import _EVALUATORS

        assert set(_EVALUATORS) == set(RuleKind)


class TestPriceRules:
    def test_close_below_fires(self) -> None:
        found = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="RELIANCE", threshold=Decimal(2600))], context()
        )
        assert found[0].fired
        assert found[0].observed == Decimal("2500")

    def test_close_below_does_not_fire_when_above(self) -> None:
        found = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="RELIANCE", threshold=Decimal(2000))], context()
        )
        assert not found[0].fired
        assert found[0].evaluable

    def test_close_above_fires(self) -> None:
        found = evaluate(
            [rule(RuleKind.CLOSE_ABOVE, subject="RELIANCE", threshold=Decimal(2400))], context()
        )
        assert found[0].fired

    def test_an_instrument_id_works_as_well_as_a_ticker(self) -> None:
        """A rule is written by a person who thinks in tickers; the system keys
        on ISIN. Accepting one and silently failing the other would make half
        the rules never fire."""
        found = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="NSE:INE002A01018", threshold=Decimal(2600))],
            context(),
        )
        assert found[0].fired

    def test_a_session_move_is_measured_in_percent(self) -> None:
        # 2400 -> 2500 is +4.17%.
        found = evaluate(
            [rule(RuleKind.MOVED_MORE_THAN, subject="RELIANCE", threshold=Decimal(4))], context()
        )
        assert found[0].fired
        assert found[0].observed == pytest.approx(Decimal("4.1666"), abs=Decimal("0.01"))

    def test_a_move_fires_in_either_direction(self) -> None:
        found = evaluate(
            [rule(RuleKind.MOVED_MORE_THAN, subject="RELIANCE", threshold=Decimal(2))],
            context(
                closes={"RELIANCE": Decimal("2200")}, previous_closes={"RELIANCE": Decimal("2400")}
            ),
        )
        assert found[0].fired
        assert found[0].observed < 0


class TestBookRules:
    def test_drawdown_fires_when_beyond(self) -> None:
        """Both numbers are negative, so 'beyond' is deeper, not larger."""
        found = evaluate([rule(RuleKind.DRAWDOWN_BEYOND, threshold=Decimal("-0.02"))], context())
        assert found[0].fired

    def test_drawdown_does_not_fire_when_shallower(self) -> None:
        found = evaluate([rule(RuleKind.DRAWDOWN_BEYOND, threshold=Decimal("-0.10"))], context())
        assert not found[0].fired

    def test_a_stale_feed_fires(self) -> None:
        found = evaluate(
            [rule(RuleKind.FEED_STALE_HOURS, subject="nse", threshold=Decimal(12))], context()
        )
        assert found[0].fired

    def test_a_fresh_feed_does_not(self) -> None:
        found = evaluate(
            [rule(RuleKind.FEED_STALE_HOURS, subject="nse", threshold=Decimal(96))], context()
        )
        assert not found[0].fired

    def test_a_risk_limit_is_compared_on_magnitude(self) -> None:
        """`daily_loss` is a negative floor and `gross_exposure` a positive
        ceiling; comparing signed values would make one of them never fire."""
        found = evaluate(
            [rule(RuleKind.RISK_LIMIT_AT, subject="gross_exposure", threshold=Decimal("0.8"))],
            context(),
        )
        assert found[0].fired


class TestUnevaluable:
    """The failure an alerting system actually dies of."""

    def test_a_missing_name_is_not_a_pass(self) -> None:
        found = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="GHOST", threshold=Decimal(100))], context()
        )
        assert not found[0].fired
        assert not found[0].evaluable
        assert "no observable close" in found[0].reason

    def test_no_book_makes_a_drawdown_rule_unevaluable(self) -> None:
        found = evaluate(
            [rule(RuleKind.DRAWDOWN_BEYOND, threshold=Decimal("-0.02"))], context(drawdown=None)
        )
        assert not found[0].evaluable
        assert "no book" in found[0].reason

    def test_an_unmeasured_limit_is_not_a_cleared_one(self) -> None:
        """The console renders an unmeasured limit as an em dash for the same
        reason: zero is a measurement, absence is not."""
        found = evaluate(
            [
                rule(
                    RuleKind.RISK_LIMIT_AT,
                    subject="cluster_concentration",
                    threshold=Decimal("0.3"),
                )
            ],
            context(),
        )
        assert not found[0].evaluable

    def test_a_name_with_no_prior_close_cannot_have_moved(self) -> None:
        found = evaluate(
            [rule(RuleKind.MOVED_MORE_THAN, subject="RELIANCE", threshold=Decimal(1))],
            context(previous_closes={}),
        )
        assert not found[0].evaluable

    def test_an_unevaluable_rule_reads_differently_from_a_quiet_one(self) -> None:
        quiet = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="RELIANCE", threshold=Decimal(1))], context()
        )[0]
        missing = evaluate(
            [rule(RuleKind.CLOSE_BELOW, subject="GHOST", threshold=Decimal(1))], context()
        )[0]
        assert quiet.fired is missing.fired is False
        assert quiet.evaluable and not missing.evaluable


class TestDisabledAndOrder:
    def test_a_disabled_rule_is_not_evaluated(self) -> None:
        found = evaluate(
            [rule(RuleKind.DRAWDOWN_BEYOND, threshold=Decimal("-0.01"), enabled=False)], context()
        )
        assert found == []

    def test_results_follow_the_order_given(self) -> None:
        rules = [
            rule(RuleKind.CLOSE_ABOVE, rule_id="a", subject="RELIANCE", threshold=Decimal(1)),
            rule(RuleKind.CLOSE_BELOW, rule_id="b", subject="RELIANCE", threshold=Decimal(1)),
        ]
        assert [t.rule.rule_id for t in evaluate(rules, context())] == ["a", "b"]


class TestSeverityIsTheRules:
    def test_the_alert_carries_the_rule_severity(self) -> None:
        """Whether a 3% move is worth waking someone is a judgement made once,
        by the person writing the rule."""
        one = rule(
            RuleKind.CLOSE_BELOW,
            subject="RELIANCE",
            threshold=Decimal(2600),
            severity=Severity.CRITICAL,
        )
        assert evaluate([one], context())[0].to_alert().severity is Severity.CRITICAL

    def test_the_note_reaches_the_message(self) -> None:
        one = rule(
            RuleKind.CLOSE_BELOW,
            subject="RELIANCE",
            threshold=Decimal(2600),
            note="stop level",
        )
        assert "stop level" in evaluate([one], context())[0].to_alert().body


class TestRuleFile:
    def loaded(self, tmp_path, payload):
        from apps.cli.watch import load_rules

        path = tmp_path / "rules.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_rules(path)

    def test_it_reads_a_rule(self, tmp_path) -> None:
        rules = self.loaded(
            tmp_path,
            [{"rule_id": "r1", "kind": "close_below", "subject": "TCS", "threshold": "3000"}],
        )
        assert rules[0].kind is RuleKind.CLOSE_BELOW
        assert rules[0].threshold == Decimal("3000")

    def test_a_threshold_is_read_as_a_decimal(self) -> None:
        """These are compared against prices. A float threshold would make
        2400.1 not equal to itself somewhere down the line."""
        assert Rule(
            rule_id="r", kind=RuleKind.CLOSE_BELOW, subject="X", threshold=Decimal("0.1")
        ).threshold == Decimal("0.1")

    def test_a_missing_file_is_no_rules_not_an_error(self, tmp_path) -> None:
        from apps.cli.watch import load_rules

        assert load_rules(tmp_path / "absent.json") == []

    def test_a_malformed_rule_refuses_the_whole_file(self, tmp_path) -> None:
        """Half a rule set looks exactly like a working one, and the rules that
        vanished are the ones nobody notices are gone."""
        with pytest.raises(ValueError, match="rule #2"):
            self.loaded(
                tmp_path,
                [
                    {"rule_id": "ok", "kind": "drawdown_beyond", "threshold": "-0.05"},
                    {"rule_id": "bad", "kind": "not_a_kind"},
                ],
            )

    def test_a_file_that_is_not_a_list_is_refused(self, tmp_path) -> None:
        with pytest.raises(TypeError, match="list of rules"):
            self.loaded(tmp_path, {"rule_id": "r"})

    def test_invalid_json_is_refused(self, tmp_path) -> None:
        from apps.cli.watch import load_rules

        path = tmp_path / "rules.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_rules(path)

    def test_the_examples_are_valid(self, tmp_path) -> None:
        """A starter file that does not load is worse than none."""
        from apps.cli.watch import EXAMPLE_RULES

        assert len(self.loaded(tmp_path, EXAMPLE_RULES)) == len(EXAMPLE_RULES)
