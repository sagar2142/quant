"""Whether the paper cycle is running — MASTER_PLAN §M9, §12.7.

**The console could say which plane was armed, and not whether it was moving.**
Environment, the live flag and the four gates all describe what *would* happen
if an order were placed. None of them says whether the scheduled cycle actually
ran last night, and that is the one that fails silently: a workflow that stopped
firing looks exactly like a quiet market until somebody counts the sessions.

Kept out of `book.py` because they answer different questions. That module
reports the account — positions, cash, limits, the blotter. This one reports the
*process* that produces it, and the distinction matters most when the answer is
"there is no account yet, because nothing has run".
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from core.clock import utc_now
from quant.math.metrics.drift import MIN_SESSIONS
from trading.paper.state import PaperStateStore, StateCorruptError

__all__ = ["M9_SESSIONS", "PaperStatus", "paper_status_of"]


#: Sessions the M9 gate asks for: six weeks of trading, five days a week.
M9_SESSIONS = 30


class PaperStatus(BaseModel):
    """Whether the scheduled cycle is running, and what it has produced."""

    #: False when no cycle has ever written state. Distinct from a cycle that
    #: ran and did nothing.
    started: bool
    cycles: int
    first_session: str | None
    last_session: str | None
    #: Sessions since the last recorded cycle. The number that says a workflow
    #: has stopped firing — a stalled clock reads as a quiet market otherwise.
    days_since_last: int | None
    halted: bool
    halt_reason: str
    #: M9 progress: sessions recorded against the sessions the gate asks for.
    sessions_required: int
    #: The strategy the cycles were run under, from the log itself. Plural
    #: entries mean the parameters changed mid-run, and the curve either side
    #: of that is two experiments rather than one.
    strategies: list[str]
    #: Why drift is not reported, when it is not. Empty once it is computable.
    drift_reason: str


def paper_status_of(store: PaperStateStore) -> PaperStatus:
    """Read the cycle log and say what it shows.

    Separated from the route so it can be tested against a state directory
    without an HTTP client, the same split the alert rules use.
    """
    rows = store.equity_history()
    sessions = [str(r.get("session", "")) for r in rows if r.get("session")]
    strategies = sorted({str(r.get("strategy") or "") for r in rows} - {""})

    halted, halt_reason = False, ""
    if store.exists():
        try:
            state = store.restore()
            halted, halt_reason = state.halted, state.halt_reason
        except StateCorruptError as exc:
            halted, halt_reason = True, str(exc)

    since: int | None = None
    if sessions:
        try:
            since = (utc_now().date() - date.fromisoformat(sessions[-1])).days
        except ValueError:
            since = None

    reason = ""
    if len(sessions) < MIN_SESSIONS:
        reason = (
            f"{len(sessions)} cycle(s) recorded; {MIN_SESSIONS} needed before a "
            "tracking error against the backtest means anything"
        )
    elif not strategies:
        # Rows written before the spec was recorded. A drift number computed
        # against guessed parameters is a different strategy, not a small error.
        reason = "cycles carry no strategy spec, so there is nothing to compare against"

    return PaperStatus(
        started=bool(rows),
        cycles=len(rows),
        first_session=sessions[0] if sessions else None,
        last_session=sessions[-1] if sessions else None,
        days_since_last=since,
        halted=halted,
        halt_reason=halt_reason,
        sessions_required=M9_SESSIONS,
        strategies=strategies,
        drift_reason=reason,
    )
