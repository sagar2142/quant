"""Durable paper-account state — MASTER_PLAN §20, §M9.

Paper trading is a *six-week* exercise (§M9 gate), run as one cycle per NSE
session by a scheduled job. The process starts, trades once, and exits; every
piece of state that must survive between cycles lives in a file, and this
module owns that file.

**JSON with Decimal-as-string, never floats.** The state round-trips positions
and cash that reconciliation later compares to the paisa. `json.dump(float(...))`
would corrupt exactly the digits reconciliation exists to check — the classic
0.1 + 0.2 problem landing in an account balance.

**Writes are atomic.** The file is replaced via `os.replace` of a fully-written
sibling, so a crash mid-write leaves the previous state intact rather than a
truncated document. Losing one cycle is recoverable; a half-written account
state is not — it would have to be reconstructed from the equity log by hand.

**The equity log is append-only NDJSON**, one line per cycle, kept separate
from the state file. The state is *current* and small; the log is *history* and
grows. Mixing them would mean rewriting the whole history on every cycle, and
the log is what the M9 drift analysis reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from core.clock import require_utc, utc_now
from core.instruments import InstrumentId
from engine.accounting import Fill, Portfolio, Position

__all__ = ["PaperState", "PaperStateStore", "StateCorruptError"]

#: Bumped when the schema changes shape. A mismatched file refuses to load
#: rather than being reinterpreted.
STATE_VERSION = 1


class StateCorruptError(RuntimeError):
    """The state file exists but cannot be trusted.

    Never "recovered" by starting fresh silently: a fresh start mid-M9 resets
    the six-week clock and destroys the drift history, so the reset must be a
    human decision.
    """

    def __init__(self, path: Path, why: str) -> None:
        super().__init__(
            f"paper state at {path} is unusable: {why}. "
            "Inspect it; delete it only if restarting the paper run is intended."
        )


@dataclass
class PaperState:
    """Everything a cycle needs from the previous one."""

    strategy_id: str
    portfolio: Portfolio
    #: Highest equity ever marked. The drawdown ladder's denominator.
    peak_equity: Decimal
    #: Last broker fill already applied, so a re-run never double-applies.
    fill_marker: str | None = None
    cycles: int = 0
    last_cycle_at: datetime | None = None
    #: Session the last cycle traded, so the same bhavcopy is never traded twice.
    last_session: date | None = None
    #: Set on an unexplained reconciliation break. Survives restarts on
    #: purpose: a halt that a re-run resets is not a halt (§9).
    halted: bool = False
    halt_reason: str = ""

    def __post_init__(self) -> None:
        if self.last_cycle_at is not None:
            self.last_cycle_at = require_utc(self.last_cycle_at)

    def engage_halt(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("a halt requires a reason")
        self.halted = True
        self.halt_reason = reason

    def clear_halt(self) -> None:
        self.halted = False
        self.halt_reason = ""


@dataclass
class PaperStateStore:
    """Reads and writes one paper account's files under `root`."""

    root: Path
    state_name: str = "paper_state.json"
    log_name: str = "paper_equity.ndjson"
    fills_name: str = "paper_fills.ndjson"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def state_path(self) -> Path:
        return self.root / self.state_name

    @property
    def log_path(self) -> Path:
        return self.root / self.log_name

    @property
    def fills_path(self) -> Path:
        return self.root / self.fills_name

    # ── state ───────────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self.state_path.is_file()

    def save(self, state: PaperState) -> None:
        """Atomically replace the state file."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = _encode_state(state)
        scratch = self.state_path.with_suffix(".json.tmp")
        scratch.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        scratch.replace(self.state_path)

    def restore(self) -> PaperState:
        """Load, refusing anything malformed rather than guessing at it."""
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StateCorruptError(self.state_path, "file does not exist") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise StateCorruptError(self.state_path, str(exc)) from exc

        version = raw.get("version")
        if version != STATE_VERSION:
            raise StateCorruptError(
                self.state_path, f"schema version {version!r}, expected {STATE_VERSION}"
            )
        try:
            return _decode_state(raw)
        except (KeyError, ValueError, TypeError, ArithmeticError) as exc:
            raise StateCorruptError(self.state_path, f"unreadable field: {exc}") from exc

    # ── equity log ──────────────────────────────────────────────────────────

    def append_equity(self, session: date, equity: Decimal, cash: Decimal, fees: Decimal) -> None:
        """One NDJSON line per cycle. This is what drift analysis reads (§35)."""
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "at": utc_now().isoformat(),
                "session": session.isoformat(),
                "equity": str(equity),
                "cash": str(cash),
                "fees_paid": str(fees),
            }
        )
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def append_fill(self, session: date, fill: Fill, symbol: str = "") -> None:
        """One NDJSON line per applied fill — the blotter's only source.

        Written from the fill the *broker* reported, after it has moved the
        book, so the log records what happened rather than what was intended.
        Nothing persisted these before, and the console's Blotter answered
        "What did I trade?" with "No trades today." after eleven real fills.

        Args:
            symbol: Display ticker. Optional and never used as identity — the
                `instrument_id` is the key (§3.3), and a symbol resolved today
                may not be the one this fill traded under.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "at": fill.event_time.isoformat(),
                "session": session.isoformat(),
                "instrument_id": str(fill.instrument_id),
                "symbol": symbol,
                "side": fill.side.value,
                "quantity": str(fill.quantity),
                "price": str(fill.price),
                "costs": str(fill.costs.total),
            }
        )
        with self.fills_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def fill_history(self, limit: int = 0) -> list[dict[str, str]]:
        """Applied fills, oldest first. `limit` keeps only the most recent N.

        A malformed line is skipped rather than raising: a blotter missing one
        row is worth more than a screen that will not render.
        """
        if not self.fills_path.is_file():
            return []
        rows: list[dict[str, str]] = []
        for line in self.fills_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:] if limit > 0 else rows

    def equity_history(self) -> list[dict[str, str]]:
        if not self.log_path.is_file():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


# ── codec ────────────────────────────────────────────────────────────────────


def _encode_state(state: PaperState) -> dict[str, object]:
    portfolio = state.portfolio
    return {
        "version": STATE_VERSION,
        "strategy_id": state.strategy_id,
        "cash": str(portfolio.cash),
        "margin_allowance": str(portfolio.margin_allowance),
        "realised_pnl": str(portfolio.realised_pnl),
        "fees_paid": str(portfolio.fees_paid),
        "positions": [
            {
                "instrument_id": str(p.instrument_id),
                "quantity": str(p.quantity),
                "average_price": str(p.average_price),
                "realised_pnl": str(p.realised_pnl),
                "fees_paid": str(p.fees_paid),
                "multiplier": str(p.multiplier),
            }
            for p in sorted(portfolio.positions.values(), key=lambda p: p.instrument_id)
            if not p.is_flat
        ],
        "peak_equity": str(state.peak_equity),
        "fill_marker": state.fill_marker,
        "cycles": state.cycles,
        "last_cycle_at": state.last_cycle_at.isoformat() if state.last_cycle_at else None,
        "last_session": state.last_session.isoformat() if state.last_session else None,
        "halted": state.halted,
        "halt_reason": state.halt_reason,
    }


def _entries(raw: dict[str, object], key: str) -> list[dict[str, object]]:
    """A list-of-objects field, or a loud TypeError feeding StateCorruptError."""
    value = raw.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key!r} should be a list, found {type(value).__name__}")
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"{key!r} contains a non-object entry")
    return value


def _decode_position(entry: dict[str, object]) -> Position:
    instrument_id = InstrumentId(str(entry["instrument_id"]))
    return Position(
        instrument_id=instrument_id,
        quantity=Decimal(str(entry["quantity"])),
        average_price=Decimal(str(entry["average_price"])),
        realised_pnl=Decimal(str(entry["realised_pnl"])),
        fees_paid=Decimal(str(entry["fees_paid"])),
        multiplier=Decimal(str(entry["multiplier"])),
    )


def _decode_state(raw: dict[str, object]) -> PaperState:
    positions = {p.instrument_id: p for p in map(_decode_position, _entries(raw, "positions"))}

    portfolio = Portfolio(
        cash=Decimal(str(raw["cash"])),
        positions=positions,
        realised_pnl=Decimal(str(raw["realised_pnl"])),
        fees_paid=Decimal(str(raw["fees_paid"])),
        margin_allowance=Decimal(str(raw["margin_allowance"])),
    )

    marker = raw.get("fill_marker")
    last_cycle_raw = raw.get("last_cycle_at")
    last_session_raw = raw.get("last_session")
    return PaperState(
        strategy_id=str(raw["strategy_id"]),
        portfolio=portfolio,
        peak_equity=Decimal(str(raw["peak_equity"])),
        fill_marker=str(marker) if marker is not None else None,
        cycles=int(str(raw["cycles"])),
        last_cycle_at=(
            datetime.fromisoformat(str(last_cycle_raw)) if last_cycle_raw is not None else None
        ),
        last_session=(
            date.fromisoformat(str(last_session_raw)) if last_session_raw is not None else None
        ),
        halted=bool(raw.get("halted", False)),
        halt_reason=str(raw.get("halt_reason", "")),
    )
