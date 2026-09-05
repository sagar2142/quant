"""Conditions worth being told about — MASTER_PLAN §12.7, §20.

**Delivery existed; rules did not.** `ops.alerts` could route a message to a
console and to Telegram, and the paper cycle used it for reconciliation breaks,
but nothing else ever decided that something was worth saying. A staleness
light that only shows while you are looking at it is not a warning.

This is the deciding half, and it is a pure function of a context the caller
assembles — the same split as the gauntlet's sample plan, and for the same
reason: what fires must depend only on its inputs, so it can be tested without
a lake, a broker or a clock.

**Three properties, none of them optional.**

*A rule that cannot be evaluated is reported, not skipped.* A price rule on a
name absent from the panel is not a rule that passed — it is a rule that did
not run, and the two must not look alike. Silent skipping is how an alerting
system decays into decoration.

*Firing is deduplicated by state, not by luck.* A drawdown that stays past a
rung would otherwise page every evening until it recovered, and an alert that
arrives daily is one nobody reads. A rule re-fires when its condition clears
and returns, not while it persists.

*Severity is the rule's, not the sender's.* Whether a 3% move is worth waking
someone is a judgement the person writing the rule makes once, rather than one
the delivery layer guesses at every time.

**Daily frequency is the honest limit.** The panel updates once a session, so a
price condition here is evaluated against a close and reaches you after it.
That makes these positional alerts — "this ended the day below your level" —
and not intraday triggers. The rule kinds are named so nobody expects
otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from ops.alerts import Alert, Severity

__all__ = [
    "Rule",
    "RuleKind",
    "Trigger",
    "WatchContext",
    "evaluate",
    "unevaluable",
]


class RuleKind(str, Enum):
    """What a rule watches.

    Named for what they actually are on a daily panel. `CLOSE_ABOVE` rather
    than `PRICE_ABOVE` because the second implies a tick this system never
    sees.
    """

    CLOSE_ABOVE = "close_above"
    CLOSE_BELOW = "close_below"
    #: Absolute session move, in percent, either direction.
    MOVED_MORE_THAN = "moved_more_than"
    #: Book drawdown at or past a level. Negative, matching `LadderRung`.
    DRAWDOWN_BEYOND = "drawdown_beyond"
    #: A feed has not produced a session in this many hours.
    FEED_STALE_HOURS = "feed_stale_hours"
    #: A named risk limit's observed value has reached its threshold.
    RISK_LIMIT_AT = "risk_limit_at"


@dataclass(frozen=True)
class Rule:
    """One condition, and how loudly it should be said."""

    rule_id: str
    kind: RuleKind
    #: The instrument, feed or limit this concerns. Empty for book-wide rules.
    subject: str = ""
    threshold: Decimal = Decimal(0)
    severity: Severity = Severity.WARN
    note: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("a rule needs an id")
        needs_subject = {
            RuleKind.CLOSE_ABOVE,
            RuleKind.CLOSE_BELOW,
            RuleKind.MOVED_MORE_THAN,
            RuleKind.FEED_STALE_HOURS,
            RuleKind.RISK_LIMIT_AT,
        }
        if self.kind in needs_subject and not self.subject.strip():
            raise ValueError(f"{self.kind.value} needs a subject")


@dataclass(frozen=True)
class WatchContext:
    """Everything the rules are evaluated against.

    Assembled by the caller so this module reads no files and no clock. Every
    field may be absent, and absence is what produces an *unevaluable* result
    rather than a passing one.
    """

    as_of: datetime
    #: instrument_id or symbol -> latest observable close.
    closes: dict[str, Decimal] = field(default_factory=dict)
    #: Same keys -> the close before that, for session moves.
    previous_closes: dict[str, Decimal] = field(default_factory=dict)
    #: Book drawdown, negative. `None` when there is no book.
    drawdown: Decimal | None = None
    #: Feed name -> hours since its newest session.
    feed_age_hours: dict[str, float] = field(default_factory=dict)
    #: Limit name -> observed value. Absent means unmeasured, not zero.
    risk_observations: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class Trigger:
    """A rule that fired, or could not be evaluated."""

    rule: Rule
    fired: bool
    #: What the condition was measured at. `None` when unevaluable.
    observed: Decimal | None
    reason: str

    @property
    def evaluable(self) -> bool:
        return self.observed is not None or self.fired

    def to_alert(self) -> Alert:
        """The message this becomes."""
        return Alert(
            severity=self.rule.severity,
            title=f"{self.rule.kind.value}: {self.rule.subject or 'book'}",
            body=self.reason + (f" — {self.rule.note}" if self.rule.note else ""),
        )


def unevaluable(rule: Rule, why: str) -> Trigger:
    """A rule that did not run.

    Distinct from one that ran and did not fire, and reported as such: a price
    rule on a name missing from the panel has not been checked, and treating
    that as "all clear" is how an alerting system quietly stops alerting.
    """
    return Trigger(rule=rule, fired=False, observed=None, reason=f"not evaluated: {why}")


def _price_of(context: WatchContext, subject: str) -> Decimal | None:
    """Latest close, by instrument id or symbol.

    Both, because a rule is written by a person who thinks in tickers while
    the system keys on ISIN (§1.1). Accepting one and silently failing the
    other would make half the rules never fire.
    """
    return context.closes.get(subject) or context.closes.get(subject.upper())


def _close_rule(rule: Rule, context: WatchContext) -> Trigger:
    close = _price_of(context, rule.subject)
    if close is None:
        return unevaluable(rule, f"{rule.subject} has no observable close")
    above = rule.kind is RuleKind.CLOSE_ABOVE
    hit = close > rule.threshold if above else close < rule.threshold
    word = "above" if above else "below"
    return Trigger(
        rule=rule,
        fired=hit,
        observed=close,
        reason=(
            f"{rule.subject} closed at {close} ({word} {rule.threshold})"
            if hit
            else f"{rule.subject} at {close}, not {word} {rule.threshold}"
        ),
    )


def _move_rule(rule: Rule, context: WatchContext) -> Trigger:
    close = _price_of(context, rule.subject)
    previous = context.previous_closes.get(rule.subject) or context.previous_closes.get(
        rule.subject.upper()
    )
    if close is None or previous is None or previous == 0:
        return unevaluable(rule, f"{rule.subject} has no prior close to compare")
    move = (close - previous) / previous * 100
    return Trigger(
        rule=rule,
        fired=abs(move) >= rule.threshold,
        observed=move,
        reason=f"{rule.subject} moved {move:+.2f}% (limit {rule.threshold}%)",
    )


def _drawdown_rule(rule: Rule, context: WatchContext) -> Trigger:
    if context.drawdown is None:
        return unevaluable(rule, "no book to measure a drawdown on")
    # Both are negative, so "beyond" is <=.
    return Trigger(
        rule=rule,
        fired=context.drawdown <= rule.threshold,
        observed=context.drawdown,
        reason=f"drawdown {context.drawdown:.2%} against {rule.threshold:.2%}",
    )


def _stale_rule(rule: Rule, context: WatchContext) -> Trigger:
    age = context.feed_age_hours.get(rule.subject)
    if age is None:
        return unevaluable(rule, f"no age known for feed {rule.subject}")
    return Trigger(
        rule=rule,
        fired=Decimal(str(age)) >= rule.threshold,
        observed=Decimal(str(round(age, 1))),
        reason=f"{rule.subject} last produced a session {age:.1f}h ago (limit {rule.threshold}h)",
    )


def _risk_rule(rule: Rule, context: WatchContext) -> Trigger:
    observed = context.risk_observations.get(rule.subject)
    if observed is None:
        return unevaluable(rule, f"{rule.subject} is unmeasured")
    return Trigger(
        rule=rule,
        fired=abs(observed) >= abs(rule.threshold),
        observed=observed,
        reason=f"{rule.subject} at {observed} against {rule.threshold}",
    )


#: One evaluator per kind. A dict rather than a chain of branches so that
#: adding a kind without an evaluator is a `KeyError` at the point of use
#: rather than a rule that silently never fires.
_EVALUATORS: dict[RuleKind, Callable[[Rule, WatchContext], Trigger]] = {
    RuleKind.CLOSE_ABOVE: _close_rule,
    RuleKind.CLOSE_BELOW: _close_rule,
    RuleKind.MOVED_MORE_THAN: _move_rule,
    RuleKind.DRAWDOWN_BEYOND: _drawdown_rule,
    RuleKind.FEED_STALE_HOURS: _stale_rule,
    RuleKind.RISK_LIMIT_AT: _risk_rule,
}


def evaluate(rules: list[Rule], context: WatchContext) -> list[Trigger]:
    """Evaluate every enabled rule.

    Returns:
        One trigger per enabled rule, in the order given — including the ones
        that did not fire and the ones that could not be evaluated. The caller
        decides what to send; this decides what is true.
    """
    return [_EVALUATORS[rule.kind](rule, context) for rule in rules if rule.enabled]
