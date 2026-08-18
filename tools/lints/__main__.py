"""CLI for the project AST lints — MASTER_PLAN §14.8.

    python -m tools.lints              # lint the whole repo
    python -m tools.lints core data    # lint specific paths

Exits 1 on any diagnostic so pre-commit and CI block the change.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tools.lints.rules import ALL_RULES, Diagnostic

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
        "lake",
        "artifacts",
    }
)
DEFAULT_ROOTS = ("core", "data", "quant", "engine", "trading", "ai", "ops", "apps", "tools")


def python_files(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            found.append(root)
            continue
        for p in root.rglob("*.py"):
            if not any(part in SKIP_DIRS for part in p.parts):
                found.append(p)
    return sorted(found)


def lint_file(path: Path) -> list[Diagnostic]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [Diagnostic(path, exc.lineno or 1, "syntax", f"cannot parse: {exc.msg}")]

    diagnostics: list[Diagnostic] = []
    for rule in ALL_RULES:
        diagnostics.extend(rule(path, tree, lines))
    return diagnostics


def main(argv: list[str]) -> int:
    repo = Path(__file__).resolve().parents[2]
    targets = [Path(a) for a in argv] or [repo / r for r in DEFAULT_ROOTS]
    roots = [t for t in targets if t.exists()]

    diagnostics: list[Diagnostic] = []
    for path in python_files(roots):
        try:
            rel = path.relative_to(repo)
        except ValueError:
            rel = path
        diagnostics.extend(Diagnostic(rel, d.line, d.rule, d.message) for d in lint_file(path))

    for d in sorted(diagnostics, key=lambda x: (str(x.path), x.line)):
        print(d.format())

    if diagnostics:
        by_rule: dict[str, int] = {}
        for d in diagnostics:
            by_rule[d.rule] = by_rule.get(d.rule, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items()))
        print(f"\n{len(diagnostics)} violation(s): {summary}")
        return 1

    print("neutron lints: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
