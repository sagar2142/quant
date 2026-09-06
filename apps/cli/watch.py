"""Evaluate the alert rules and send what fired — MASTER_PLAN §12.7, §20.

    python -m apps.cli.watch              # evaluate and send
    python -m apps.cli.watch --dry-run    # say what would be sent
    python -m apps.cli.watch --example    # write a starter rule file

Rules live in `<lake>/alerts/rules.json` and are edited by hand. A single
operator does not need a CRUD screen to keep six conditions, and a file is
diffable, version-controllable and readable without the console running.

**A job, not a daemon**, like `apps.cli.paper` and for the same reasons: the
panel updates once a session, so there is nothing to watch between them. It is
called from `apps.cli.daily` after the ingest, which is the moment the answers
can actually have changed.

**Firing is deduplicated across runs.** A drawdown past a rung would otherwise
page every evening until it recovered, and an alert that arrives daily is one
nobody reads. The last state of each rule is kept in `alerts/state.json`, and a
rule announces itself when it *starts* firing and again when it clears — the
two transitions worth knowing about — rather than continuously while it holds.

**A rule that could not be evaluated is reported, never silently skipped.** A
price rule on a name that left the panel has not passed; it has stopped being
checked, and that is the failure mode an alerting system dies of.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from core.clock import UTC, as_decision_time, utc_now
from core.config import settings
from ops.alerts import Alert, AlertRouter, Severity
from ops.routing import build_router, describe_channels
from ops.watch import Rule, RuleKind, Trigger, WatchContext, evaluate

__all__ = ["assemble_context", "load_rules", "results_horizon", "run"]

RULES_FILE = "rules.json"
STATE_FILE = "state.json"

#: Sessions of panel read to find the latest two closes. Two is the minimum and
#: three is the margin for a name that did not trade yesterday.
RECENT_SESSIONS = 3

#: How far ahead the results calendar is read. A rule may set any threshold up
#: to this; beyond it the announcement is too far off to act on.
MAX_RESULTS_HORIZON = 30

#: A calendar older than this is treated as no calendar at all. Companies
#: announce meetings a few days ahead, so a fortnight-old fetch reporting
#: "nothing announced" is a false all-clear rather than useful quiet.
STALE_CALENDAR_DAYS = 7

EXAMPLE_RULES: list[dict[str, Any]] = [
    {
        "rule_id": "nse-feed-stale",
        "kind": "feed_stale_hours",
        "subject": "nse",
        "threshold": "96",
        "severity": "CRITICAL",
        "note": "the daily ingest has not run; everything on screen is stale",
    },
    {
        "rule_id": "book-drawdown-5pct",
        "kind": "drawdown_beyond",
        "threshold": "-0.05",
        "severity": "WARN",
        "note": "first ladder rung",
    },
    {
        "rule_id": "reliance-reports-soon",
        "kind": "reports_within_days",
        "subject": "RELIANCE",
        "threshold": "3",
        "severity": "WARN",
        "note": "earnings inside the week; size the position for a gap",
    },
    {
        "rule_id": "reliance-below-2400",
        "kind": "close_below",
        "subject": "RELIANCE",
        "threshold": "2400",
        "severity": "INFO",
    },
]


@dataclass(frozen=True)
class WatchPaths:
    """Where the rules and the firing state live."""

    root: Path

    @property
    def rules(self) -> Path:
        return self.root / "alerts" / RULES_FILE

    @property
    def state(self) -> Path:
        return self.root / "alerts" / STATE_FILE


def load_rules(path: Path) -> list[Rule]:
    """Read the rule file.

    Raises:
        ValueError: if a rule is malformed.
        TypeError: if the file does not hold a list.

    Refused rather than partially loaded: half a rule set looks exactly like a
    working one, and the rules that vanished are the ones nobody notices are
    gone.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise TypeError(f"{path} must hold a list of rules")

    rules: list[Rule] = []
    for index, entry in enumerate(raw):
        try:
            rules.append(
                Rule(
                    rule_id=str(entry["rule_id"]),
                    kind=RuleKind(entry["kind"]),
                    subject=str(entry.get("subject", "")),
                    threshold=Decimal(str(entry.get("threshold", "0"))),
                    severity=Severity(entry.get("severity", "WARN")),
                    note=str(entry.get("note", "")),
                    enabled=bool(entry.get("enabled", True)),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path} rule #{index + 1} is invalid: {exc}") from exc
    return rules


def _read_state(path: Path) -> dict[str, bool]:
    if not path.exists():
        return {}
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A corrupt state file means every rule looks new and fires once. Noisy
        # for one run, which is the safe direction — the alternative is
        # treating unknown state as "already fired" and staying silent.
        return {}
    return {str(k): bool(v) for k, v in found.items()} if isinstance(found, dict) else {}


def _write_state(path: Path, state: dict[str, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def results_horizon(rules: list[Rule]) -> int:
    """How far ahead the calendar must be read to answer these rules.

    Driven by the rules rather than fixed, because a horizon shorter than a
    rule's threshold cannot answer it — and the tempting answer, "nothing
    announced", is a measurement nobody took.
    """
    wanted = [
        int(rule.threshold)
        for rule in rules
        if rule.enabled and rule.kind is RuleKind.REPORTS_WITHIN_DAYS
    ]
    return min(max(wanted, default=0), MAX_RESULTS_HORIZON)


def assemble_context(lake: Path, venue: str = "NSE", horizon: int = 0) -> WatchContext:
    """Gather everything the rules are evaluated against.

    Args:
        lake: Where the stores live.
        venue: Exchange whose panel supplies the closes.
        horizon: Days ahead to read the results calendar. `results_horizon`
            derives it from the rules; zero reads no calendar at all, which
            leaves every calendar rule unevaluable rather than quiet.

    Kept here rather than in `ops.watch` so the evaluation stays a pure
    function of its inputs and can be tested without a lake, a broker or a
    clock.
    """
    from apps.api.snapshot import book_snapshot  # noqa: PLC0415 - heavy
    from data.store.panel import PanelStore  # noqa: PLC0415

    now = utc_now()
    closes: dict[str, Decimal] = {}
    previous: dict[str, Decimal] = {}
    feed_ages: dict[str, float] = {}

    store = PanelStore(lake, venue=venue)
    sessions = store.sessions()
    if sessions:
        feed_ages[venue.lower()] = (
            now - datetime(sessions[-1].year, sessions[-1].month, sessions[-1].day, tzinfo=UTC)
        ).total_seconds() / 3600.0

        recent = store.view(
            as_of=as_decision_time(now),
            start=sessions[max(0, len(sessions) - RECENT_SESSIONS)],
        )
        if not recent.is_empty():
            ordered = recent.sort("event_time")
            latest_time = ordered["event_time"].max()
            latest = ordered.filter(pl.col("event_time") == latest_time)
            earlier = ordered.filter(pl.col("event_time") < latest_time)
            prior_time = earlier["event_time"].max() if not earlier.is_empty() else None

            for symbol, instrument_id, close in zip(
                latest["symbol"], latest["instrument_id"], latest["close"], strict=True
            ):
                value = Decimal(str(close))
                closes[str(symbol)] = value
                closes[str(instrument_id)] = value

            if prior_time is not None:
                before = ordered.filter(pl.col("event_time") == prior_time)
                for symbol, instrument_id, close in zip(
                    before["symbol"], before["instrument_id"], before["close"], strict=True
                ):
                    value = Decimal(str(close))
                    previous[str(symbol)] = value
                    previous[str(instrument_id)] = value

    drawdown: Decimal | None = None
    try:
        from apps.api.book import DEFAULT_STATE_DIR, _latest_marks  # noqa: PLC0415

        snapshot = book_snapshot(DEFAULT_STATE_DIR, _latest_marks(None))
        drawdown = snapshot.drawdown
    except Exception:  # noqa: BLE001 - no book is a state, not a failure
        drawdown = None

    days_to_results, calendar_known = _announced_results(lake, now.date(), horizon)

    return WatchContext(
        as_of=now,
        closes=closes,
        previous_closes=previous,
        drawdown=drawdown,
        feed_age_hours=feed_ages,
        days_to_results=days_to_results,
        days_to_results_known=calendar_known,
        results_horizon_days=horizon if calendar_known else 0,
    )


def _announced_results(lake: Path, today: date, horizon: int) -> tuple[dict[str, int], bool]:
    """Days until each name's announced results meeting, and whether we know.

    The second half of the pair is what stops an unread calendar from looking
    like a quiet one. A stale calendar is treated as no calendar: one fetched
    a fortnight ago has no useful opinion about the coming week, and reporting
    "nothing announced" from it would be a false all-clear.
    """
    if horizon <= 0:
        return {}, False
    from data.store.events import EventStore  # noqa: PLC0415 - only when asked

    window = EventStore(lake).upcoming(today, within_days=horizon, results_only=True)
    if window.observed_at is None or window.age_days > STALE_CALENDAR_DAYS:
        return {}, False

    days: dict[str, int] = {}
    for row in window.rows.iter_rows(named=True):
        ahead = (row["event_date"] - today).days
        for key in (row["symbol"], row["instrument_id"]):
            if key and (key not in days or ahead < days[key]):
                days[key] = ahead
    return days, True


def _announce(trigger: Trigger, was_firing: bool, router: AlertRouter, dry_run: bool) -> str | None:
    """Send an alert for a transition, or nothing. Returns what was sent.

    Only transitions: a condition that has been true since yesterday is not
    news, and one that stops being true is.
    """
    if trigger.fired and not was_firing:
        alert = trigger.to_alert()
    elif was_firing and not trigger.fired and trigger.observed is not None:
        alert = Alert(
            severity=Severity.INFO,
            title=f"cleared: {trigger.rule.rule_id}",
            body=trigger.reason,
        )
    else:
        return None

    if not dry_run:
        router.send(alert)
    return f"{alert.severity.value}  {alert.title}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alert rules against the lake")
    parser.add_argument("--lake", default=None)
    parser.add_argument("--venue", choices=["NSE", "BSE"], default="NSE")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without sending")
    parser.add_argument("--example", action="store_true", help="Write a starter rule file and exit")
    return parser.parse_args(argv)


def _write_example(paths: WatchPaths) -> int:
    """Write a starter rule file, or refuse to clobber one."""
    if paths.rules.exists():
        print(f"{paths.rules} already exists; not overwriting it")
        return 1
    paths.rules.parent.mkdir(parents=True, exist_ok=True)
    paths.rules.write_text(json.dumps(EXAMPLE_RULES, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(EXAMPLE_RULES)} example rule(s) to {paths.rules}")
    return 0


def _dispatch(
    triggers: list[Trigger], state: dict[str, bool], router: AlertRouter, dry_run: bool
) -> tuple[list[str], list[Trigger]]:
    """Send what transitioned and collect what could not be evaluated."""
    sent: list[str] = []
    unevaluated: list[Trigger] = []
    for trigger in triggers:
        if not trigger.evaluable:
            unevaluated.append(trigger)
            # Left out of the state entirely: a rule that did not run has not
            # stopped firing, and recording it as cleared would send a false
            # all-clear the next time it can be read.
            continue
        message = _announce(trigger, state.get(trigger.rule.rule_id, False), router, dry_run)
        if message:
            sent.append(message)
        state[trigger.rule.rule_id] = trigger.fired
    return sent, unevaluated


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = WatchPaths(Path(args.lake) if args.lake else settings.lake)

    if args.example:
        return _write_example(paths)

    try:
        rules = load_rules(paths.rules)
    except (TypeError, ValueError) as exc:
        print(exc)
        return 1

    if not rules:
        print(f"no rules in {paths.rules}")
        print("Write a starter set with: python -m apps.cli.watch --example")
        return 0

    context = assemble_context(paths.root, args.venue, results_horizon(rules))
    triggers = evaluate(rules, context)
    state = _read_state(paths.state)
    router = build_router()
    if not args.dry_run:
        print(describe_channels())

    sent, unevaluated = _dispatch(triggers, state, router, args.dry_run)
    if not args.dry_run:
        _write_state(paths.state, state)

    verb = " would be" if args.dry_run else ""
    print(f"\n{len(triggers)} rule(s) evaluated, {len(sent)} alert(s){verb} sent")
    for line in sent:
        print(f"  {line}")

    if unevaluated:
        print(f"\n{len(unevaluated)} rule(s) COULD NOT BE EVALUATED:")
        for trigger in unevaluated:
            print(f"  {trigger.rule.rule_id:<28}{trigger.reason}")
        print("A rule that cannot run is not a rule that passed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
