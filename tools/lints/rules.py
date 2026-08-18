"""Project-specific AST lint rules — MASTER_PLAN §14.8.

Each rule guards one of the §14.1 correctness non-negotiables. Nothing
off-the-shelf catches these, and each one, when violated, silently produces a
wrong number rather than a crash.

Escape hatch: append ``# lint: allow-<rule-id>`` to the offending line. Use it
rarely and with a reason on the preceding line.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Modules where money is computed. Floats are banned here (§14.1.2): if a broker
# could disagree with you about the number, it must be Decimal.
MONEY_MODULES = (
    "trading/portfolio",
    "trading/execution",
    "trading/reconcile",
    # Risk limits are notional thresholds compared against real exposure. A
    # float rounding error here is the difference between a blocked order and
    # an allowed one.
    "trading/risk",
    "engine/costs",
    # Position and cash accounting is a shared primitive: the backtester needs
    # it, and §3.2 forbids engine/ importing trading/. It is reconciled against
    # broker statements, so it is exact.
    "engine/accounting",
)

# Junk-drawer names that grow forever and never shrink (§14.3).
JUNK_DRAWER_NAMES = frozenset({"utils", "helpers", "misc", "common", "stuff", "shared"})

MAX_FILE_LINES = 400  # §14.2

# Receivers whose bulk-read methods can return future data (§14.1.4).
STORE_HINTS = ("store", "feed", "loader", "reader")
UNBOUNDED_READS = frozenset({"load", "load_all", "read_all", "fetch_all", "all"})


@dataclass(frozen=True)
class Diagnostic:
    path: Path
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def _allowed(source_lines: list[str], line: int, rule: str) -> bool:
    """True if the line carries an explicit escape comment for this rule."""
    if not 1 <= line <= len(source_lines):
        return False
    return f"# lint: allow-{rule}" in source_lines[line - 1]


def _rel(path: Path) -> str:
    return path.as_posix()


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1 — naive datetimes (§14.1.3)
# ─────────────────────────────────────────────────────────────────────────────
def check_naive_datetime(path: Path, tree: ast.AST, lines: list[str]) -> list[Diagnostic]:
    """datetime.now() without tz=, and the always-naive datetime.utcnow()."""
    out: list[Diagnostic] = []
    rule = "naive-datetime"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        if attr == "utcnow":
            out.append(
                Diagnostic(
                    path,
                    node.lineno,
                    rule,
                    "datetime.utcnow() returns a naive datetime. Use datetime.now(timezone.utc).",
                )
            )
        elif attr == "now":
            has_tz = any(kw.arg == "tz" for kw in node.keywords) or bool(node.args)
            if not has_tz:
                out.append(
                    Diagnostic(
                        path,
                        node.lineno,
                        rule,
                        "datetime.now() without a timezone. Use datetime.now(timezone.utc).",
                    )
                )
    return [d for d in out if not _allowed(lines, d.line, rule)]


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2 — float in money modules (§14.1.2)
# ─────────────────────────────────────────────────────────────────────────────
def _is_float_annotation(node: ast.expr | None) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "float"
    if isinstance(node, ast.Subscript):  # Optional[float], list[float], ...
        return _is_float_annotation(node.slice)
    if isinstance(node, ast.BinOp):  # float | None
        return _is_float_annotation(node.left) or _is_float_annotation(node.right)
    if isinstance(node, ast.Tuple):
        return any(_is_float_annotation(e) for e in node.elts)
    return False


def check_money_float(path: Path, tree: ast.AST, lines: list[str]) -> list[Diagnostic]:
    """Ban float annotations and float() casts in modules that compute money."""
    rel = _rel(path)
    if not any(m in rel for m in MONEY_MODULES):
        return []

    out: list[Diagnostic] = []
    rule = "money-float"
    msg = (
        "float in a money module. Use Decimal (§14.1.2): if a broker could "
        "disagree with you about the number, it is Decimal."
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and _is_float_annotation(node.annotation):
            out.append(Diagnostic(path, node.lineno, rule, msg))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_float_annotation(node.returns):
                out.append(Diagnostic(path, node.lineno, rule, f"return {msg}"))
            for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                if _is_float_annotation(arg.annotation):
                    out.append(Diagnostic(path, arg.lineno, rule, f"parameter '{arg.arg}': {msg}"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
        ):
            out.append(Diagnostic(path, node.lineno, rule, f"float() cast: {msg}"))
    return [d for d in out if not _allowed(lines, d.line, rule)]


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3 — unseeded randomness (§14.1.1)
# ─────────────────────────────────────────────────────────────────────────────
def check_unseeded_rng(path: Path, tree: ast.AST, lines: list[str]) -> list[Diagnostic]:
    """np.random.<fn>() and random.<fn>() break reproducibility (an M3 gate).

    The permitted form is an explicit generator whose seed is recorded on the
    experiment row: rng = np.random.default_rng(seed).
    """
    out: list[Diagnostic] = []
    rule = "unseeded-rng"
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        # Attribute chain rooted at the numpy alias, e.g. the numpy random module.
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "random"
            and isinstance(value.value, ast.Name)
            and value.value.id in {"np", "numpy"}
            and node.func.attr != "default_rng"
        ):
            out.append(
                Diagnostic(
                    path,
                    node.lineno,
                    rule,
                    f"np.random.{node.func.attr}() is unseeded global state. "
                    "Use rng = np.random.default_rng(seed) and record the seed.",
                )
            )
        # Bare call against the stdlib random module, which is global state.
        elif isinstance(value, ast.Name) and value.id == "random" and node.func.attr != "Random":
            out.append(
                Diagnostic(
                    path,
                    node.lineno,
                    rule,
                    f"random.{node.func.attr}() is unseeded global state. "
                    "Use random.Random(seed) and record the seed.",
                )
            )
    return [d for d in out if not _allowed(lines, d.line, rule)]


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4 — unbounded data reads (§14.1.4)
# ─────────────────────────────────────────────────────────────────────────────
def check_unbounded_read(path: Path, tree: ast.AST, lines: list[str]) -> list[Diagnostic]:
    """Outside the data layer, the future must be unreachable by construction.

    Bans bulk reads on store/feed objects and any .view() call that omits as_of.
    """
    rel = _rel(path)
    in_data_layer = rel.startswith("data/") or "/data/" in rel
    out: list[Diagnostic] = []
    rule = "unbounded-read"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr

        if method == "view" and not any(kw.arg == "as_of" for kw in node.keywords):
            out.append(
                Diagnostic(
                    path,
                    node.lineno,
                    rule,
                    ".view() without as_of= can return future data. "
                    "Pass the decision time explicitly.",
                )
            )
            continue

        if in_data_layer or method not in UNBOUNDED_READS:
            continue
        receiver = node.func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", "")
        if any(hint in name.lower() for hint in STORE_HINTS):
            out.append(
                Diagnostic(
                    path,
                    node.lineno,
                    rule,
                    f"{name}.{method}() is an unbounded read outside the data layer. "
                    "Use .view(as_of=decision_time) so the future is unreachable.",
                )
            )
    return [d for d in out if not _allowed(lines, d.line, rule)]


# ─────────────────────────────────────────────────────────────────────────────
# Rule 5 — junk-drawer modules (§14.3)
# ─────────────────────────────────────────────────────────────────────────────
def check_junk_drawer(path: Path, _tree: ast.AST, _lines: list[str]) -> list[Diagnostic]:
    stem = path.stem
    if stem in JUNK_DRAWER_NAMES or path.parent.name in JUNK_DRAWER_NAMES:
        return [
            Diagnostic(
                path,
                1,
                "junk-drawer",
                f"'{stem}' is a junk drawer: it only ever grows. "
                "Name the module after its domain responsibility.",
            )
        ]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Rule 6 — file length (§14.2)
# ─────────────────────────────────────────────────────────────────────────────
def check_file_length(path: Path, _tree: ast.AST, lines: list[str]) -> list[Diagnostic]:
    if len(lines) <= MAX_FILE_LINES:
        return []
    return [
        Diagnostic(
            path,
            MAX_FILE_LINES,
            "file-length",
            f"{len(lines)} lines exceeds the {MAX_FILE_LINES}-line limit. "
            "The module is doing more than one thing; split by responsibility.",
        )
    ]


ALL_RULES = (
    check_naive_datetime,
    check_money_float,
    check_unseeded_rng,
    check_unbounded_read,
    check_junk_drawer,
    check_file_length,
)
