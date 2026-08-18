"""Every custom lint must actually fire. A rule that never triggers is decoration.

MASTER_PLAN §14.8 — each rule guards a §14.1 non-negotiable, so each rule gets
a positive test (it catches the violation) and a negative test (it permits the
correct form).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tools.lints.rules import (
    check_file_length,
    check_junk_drawer,
    check_money_float,
    check_naive_datetime,
    check_unbounded_read,
    check_unseeded_rng,
)


def run(rule, source: str, path: str = "quant/features/momentum.py"):
    p = Path(path)
    return rule(p, ast.parse(source), source.splitlines())


# ── Rule 1: naive datetime (§14.1.3) ─────────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        "import datetime\nx = datetime.datetime.now()",
        "from datetime import datetime\nx = datetime.now()",
        "from datetime import datetime\nx = datetime.utcnow()",
    ],
)
def test_naive_datetime_caught(source):
    assert run(check_naive_datetime, source)


@pytest.mark.parametrize(
    "source",
    [
        "from datetime import datetime, timezone\nx = datetime.now(timezone.utc)",
        "from datetime import datetime, timezone\nx = datetime.now(tz=timezone.utc)",
    ],
)
def test_tz_aware_datetime_allowed(source):
    assert not run(check_naive_datetime, source)


def test_naive_datetime_escape_hatch():
    src = "from datetime import datetime\nx = datetime.now()  # lint: allow-naive-datetime"
    assert not run(check_naive_datetime, src)


# ── Rule 2: float in money modules (§14.1.2) ─────────────────────────────────

MONEY_PATH = "trading/portfolio/accounting.py"


@pytest.mark.parametrize(
    "source",
    [
        "def fee(x: float) -> None: ...",
        "def fee() -> float: ...",
        "cash: float = 0.0",
        "x = float('1.5')",
        "def fee(x: float | None) -> None: ...",
    ],
)
def test_money_float_caught(source):
    assert run(check_money_float, source, MONEY_PATH)


@pytest.mark.parametrize(
    "source",
    [
        "from decimal import Decimal\ndef fee(x: Decimal) -> Decimal: ...",
        "from decimal import Decimal\ncash: Decimal = Decimal('0')",
    ],
)
def test_decimal_allowed_in_money_modules(source):
    assert not run(check_money_float, source, MONEY_PATH)


def test_float_allowed_outside_money_modules():
    # Statistics are float64 by design (§14.1.2).
    src = "def sharpe(returns: list[float]) -> float: ..."
    assert not run(check_money_float, src, "quant/math/metrics/performance.py")


# ── Rule 3: unseeded RNG (§14.1.1) ───────────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nx = np.random.randn(10)",
        "import numpy as np\nx = np.random.permutation(10)",
        "import random\nx = random.shuffle([1, 2])",
    ],
)
def test_unseeded_rng_caught(source):
    assert run(check_unseeded_rng, source)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nrng = np.random.default_rng(42)\nx = rng.standard_normal(10)",
        "import random\nr = random.Random(42)\nx = r.random()",
    ],
)
def test_seeded_rng_allowed(source):
    assert not run(check_unseeded_rng, source)


# ── Rule 4: unbounded reads (§14.1.4) ────────────────────────────────────────


def test_view_without_as_of_caught():
    src = "bars = store.view('RELIANCE')"
    assert run(check_unbounded_read, src)


def test_view_with_as_of_allowed():
    src = "bars = store.view('RELIANCE', as_of=decision_time)"
    assert not run(check_unbounded_read, src)


def test_bulk_store_read_caught_outside_data_layer():
    src = "bars = bar_store.load('RELIANCE')"
    assert run(check_unbounded_read, src, "quant/strategies/momentum.py")


def test_bulk_store_read_allowed_inside_data_layer():
    src = "bars = bar_store.load('RELIANCE')"
    assert not run(check_unbounded_read, src, "data/store/parquet.py")


def test_unrelated_load_not_flagged():
    # json.load / config.load must not trip the rule.
    src = "import json\ncfg = json.load(fh)"
    assert not run(check_unbounded_read, src, "quant/strategies/momentum.py")


# ── Rule 5: junk drawers (§14.3) ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    ["quant/utils.py", "core/helpers.py", "data/misc.py", "engine/common/thing.py"],
)
def test_junk_drawer_caught(path):
    assert run(check_junk_drawer, "x = 1", path)


def test_domain_named_module_allowed():
    assert not run(check_junk_drawer, "x = 1", "quant/features/momentum.py")


# ── Rule 6: file length (§14.2) ──────────────────────────────────────────────


def test_long_file_caught():
    assert run(check_file_length, "x = 1\n" * 401)


def test_short_file_allowed():
    assert not run(check_file_length, "x = 1\n" * 399)
