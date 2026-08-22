"""Ops console API (§13.6, §26).

The kill switch is the only mutating endpoint, so it is the one that gets
tested hardest.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from core.config import Settings
from ops.alerts import Alert, AlertRouter
from trading.risk.engine import RiskEngine


class RecordingSink:
    """Captures alerts instead of sending them."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


#: The kill switch is unavailable until a token is configured — that is the
#: point of the guard, so these tests configure one rather than working around
#: it. Reads stay open either way, which is why the read tests need no header.
HARNESS_TOKEN = "harness-token"


@pytest.fixture
def harness():
    """A client whose kill switch is reachable.

    Authenticating here rather than disabling the guard: a harness that
    bypassed authentication would stop testing the endpoint that ships.

    Settings are restored *before* the auth module is reloaded, not by
    monkeypatch afterwards. Fixture teardown runs ahead of monkeypatch teardown,
    so reloading first would rebind the module to the still-patched token and
    leak it into every later test.
    """
    import importlib

    import core.config as config_module

    original = config_module.settings
    config_module.settings = Settings(api_token=HARNESS_TOKEN)
    import apps.api.auth as auth_module

    importlib.reload(auth_module)

    engine = RiskEngine()
    sink = RecordingSink()
    client = TestClient(create_app(engine, AlertRouter([sink])))
    client.headers.update({"Authorization": f"Bearer {HARNESS_TOKEN}"})
    yield client, engine, sink

    config_module.settings = original
    importlib.reload(auth_module)


class TestHealth:
    def test_reports_environment_and_kill_state(self, harness):
        client, _, _ = harness
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["kill_engaged"] is False

    def test_live_is_disabled_by_default(self, harness):
        """Default-off is deliberate (§21)."""
        client, _, _ = harness
        assert client.get("/health").json()["live_enabled"] is False


class TestVitals:
    def test_exposes_the_ladder_rungs(self, harness):
        client, _, _ = harness
        body = client.get("/vitals").json()
        # The console draws the same rungs the risk engine acts on.
        assert len(body["ladder_rungs"]) == 3
        assert all(float(r) < 0 for r in body["ladder_rungs"])

    def test_reflects_kill_state(self, harness):
        client, engine, _ = harness
        engine.engage_kill("test halt", "operator")
        assert client.get("/vitals").json()["kill_engaged"] is True


class TestKillSwitch:
    def test_engaging_halts_the_engine(self, harness):
        client, engine, _ = harness
        response = client.post("/kill", json={"reason": "feed gap", "operator": "sagar"})
        assert response.status_code == 200
        assert response.json()["engaged"] is True
        assert engine.is_killed

    def test_engaging_raises_an_alert(self, harness):
        """The console is never the only place a halt is recorded (§12.7)."""
        client, _, sink = harness
        client.post("/kill", json={"reason": "feed gap", "operator": "sagar"})
        assert len(sink.alerts) == 1
        assert "kill switch ENGAGED" in sink.alerts[0].title

    def test_reason_is_required(self, harness):
        client, engine, _ = harness
        response = client.post("/kill", json={"reason": "", "operator": "sagar"})
        assert response.status_code == 422
        assert not engine.is_killed

    def test_operator_is_required(self, harness):
        client, engine, _ = harness
        response = client.post("/kill", json={"reason": "a real reason", "operator": ""})
        assert response.status_code == 422
        assert not engine.is_killed

    def test_release_resumes_trading(self, harness):
        client, engine, _ = harness
        client.post("/kill", json={"reason": "feed gap", "operator": "sagar"})
        response = client.post("/kill/release", json={"reason": "cause found", "operator": "sagar"})
        assert response.status_code == 200
        assert not engine.is_killed

    def test_release_is_also_alerted(self, harness):
        client, _, sink = harness
        client.post("/kill", json={"reason": "feed gap", "operator": "sagar"})
        client.post("/kill/release", json={"reason": "fixed", "operator": "sagar"})
        assert "RELEASED" in sink.alerts[-1].title


class TestReadOnlySurface:
    def test_no_order_entry_endpoint(self, harness):
        """Only halting is exposed. Its failure mode is recoverable; placing
        orders over a network surface is not."""
        client, _, _ = harness
        paths = set(client.app.openapi()["paths"])
        assert "/orders" not in paths
        mutating = {p for p in paths if p.startswith("/kill")}
        assert mutating == {"/kill", "/kill/release"}


class TestHostValidation:
    """§13.7, §21 — binding to 127.0.0.1 is necessary and not sufficient.

    A browser is already inside the loopback boundary. A page the operator
    visits can re-resolve its own hostname to 127.0.0.1 after loading, at which
    point its requests are same-origin and CORS does not apply. The Host header
    is what still distinguishes them, because the attacking page must send its
    own name.
    """

    def client(self):
        from fastapi.testclient import TestClient

        from apps.api.main import create_app

        return TestClient(create_app())

    def test_the_local_console_is_served(self):
        assert self.client().get("/health").status_code == 200

    def test_a_foreign_host_header_is_refused(self):
        response = self.client().get("/health", headers={"host": "evil.example.com"})
        assert response.status_code == 400

    def test_the_kill_switch_cannot_be_released_cross_origin(self):
        """The direction that matters. Engaging a halt is fail-safe; releasing
        one is the last thing standing between a known-bad book and the
        market."""
        body = {"reason": "released by an attacker", "operator": "evil"}
        response = self.client().post(
            "/kill/release", json=body, headers={"host": "evil.example.com"}
        )
        assert response.status_code == 400

    def test_localhost_is_also_accepted(self):
        assert self.client().get("/health", headers={"host": "localhost"}).status_code == 200

    def test_the_allowed_hosts_carry_no_port(self):
        """Starlette strips the port before comparing, so pinning one would
        break every other port without adding protection."""
        from apps.api.main import ALLOWED_HOSTS

        assert all(":" not in host for host in ALLOWED_HOSTS)


class TestVitalsReadRealState:
    """`/vitals` used to return literal zeros with a docstring calling them
    placeholders. The paper daemon writes real state, and these assert the bar
    reads it rather than inventing a flat, fresh, healthy book.
    """

    def paper_book(self, monkeypatch, tmp_path, **kwargs):
        """Point every state-reading endpoint at an isolated directory."""
        from tests.test_snapshot import write_state

        write_state(tmp_path, **kwargs)
        monkeypatch.setattr("apps.api.main.DEFAULT_STATE_DIR", tmp_path)
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path)

    def test_drawdown_comes_from_the_state_not_a_constant(self, harness, monkeypatch, tmp_path):
        from decimal import Decimal

        client, _, _ = harness
        self.paper_book(
            monkeypatch,
            tmp_path,
            cash=Decimal(900_000),
            peak_equity=Decimal(1_000_000),
        )
        assert float(client.get("/vitals").json()["drawdown"]) == pytest.approx(-0.10)

    def test_an_unmeasured_quantity_is_null_not_zero(self, harness, monkeypatch, tmp_path):
        """One cycle cannot produce a day-over-day change, and reporting ₹0.00
        would claim a flat session that was never measured."""
        from decimal import Decimal

        client, _, _ = harness
        self.paper_book(monkeypatch, tmp_path, equity_rows=[Decimal(998_000)])
        body = client.get("/vitals").json()
        assert body["day_pnl"] is None
        assert body["day_pnl_pct"] is None

    def test_an_absent_book_is_reported_as_absent(self, harness, monkeypatch, tmp_path):
        client, _, _ = harness
        monkeypatch.setattr("apps.api.main.DEFAULT_STATE_DIR", tmp_path / "nothing")
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path / "nothing")
        body = client.get("/vitals").json()
        assert body["book_present"] is False
        assert body["staleness_seconds"] is None
        assert body["drawdown"] is None

    def test_a_book_that_never_ran_is_not_healthy(self, harness, monkeypatch, tmp_path):
        """Green on an unstarted system is the same lie as a zero drawdown on
        a losing one."""
        client, _, _ = harness
        monkeypatch.setattr("apps.api.main.DEFAULT_STATE_DIR", tmp_path / "nothing")
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path / "nothing")
        assert client.get("/vitals").json()["feeds"][0]["health"] == "down"

    def test_a_stale_cycle_degrades_the_feed(self, harness, monkeypatch, tmp_path):
        from datetime import timedelta

        client, _, _ = harness
        self.paper_book(monkeypatch, tmp_path, cycle_age=timedelta(hours=48))
        assert client.get("/vitals").json()["feeds"][0]["health"] == "degraded"

    def test_a_very_stale_cycle_marks_the_feed_down(self, harness, monkeypatch, tmp_path):
        from datetime import timedelta

        client, _, _ = harness
        self.paper_book(monkeypatch, tmp_path, cycle_age=timedelta(days=7))
        assert client.get("/vitals").json()["feeds"][0]["health"] == "down"

    def test_a_fresh_cycle_is_healthy(self, harness, monkeypatch, tmp_path):
        from datetime import timedelta

        client, _, _ = harness
        self.paper_book(monkeypatch, tmp_path, cycle_age=timedelta(minutes=5))
        assert client.get("/vitals").json()["feeds"][0]["health"] == "ok"

    def test_the_daily_cycle_is_not_judged_on_the_tick_feed_thresholds(self):
        """§12.7's 2s/10s describe a live tick feed. A once-a-session batch
        judged by them would sit red permanently, which teaches the operator to
        ignore the colour."""
        from apps.api.main import CYCLE_WARN_SECONDS, STALE_WARN_SECONDS

        assert CYCLE_WARN_SECONDS > STALE_WARN_SECONDS * 1000


class TestRiskLimitsObservations:
    def test_limits_are_listed_even_with_no_book(self, harness):
        """The engine enforces them whether or not an account exists."""
        client, _, _ = harness
        rows = client.get("/risk/limits").json()
        assert len(rows) == 10
        assert all(row["threshold"] for row in rows)

    def test_per_order_limits_are_null_rather_than_zero(self, harness, monkeypatch, tmp_path):
        client, _, _ = harness
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path / "nothing")
        rows = {row["name"]: row for row in client.get("/risk/limits").json()}
        assert rows["price_band"]["observed"] is None
        assert rows["price_band"]["passed"] is None


class TestPaperLogEndpoints:
    def test_fills_are_empty_rather_than_missing_before_a_cycle(
        self, harness, monkeypatch, tmp_path
    ):
        client, _, _ = harness
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path)
        assert client.get("/fills").json() == []

    def test_an_unreconciled_book_does_not_claim_agreement(self, harness, monkeypatch, tmp_path):
        """The console rendered "Broker and internal records agree." from an
        array nothing populated — an affirmative safety claim about a check
        that had not run."""
        client, _, _ = harness
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path / "nothing")
        body = client.get("/reconciliation").json()
        assert body["checked"] is False
        assert body["halted"] is False

    def test_a_reconciled_book_reports_its_cycle_count(self, harness, monkeypatch, tmp_path):
        from tests.test_snapshot import write_state

        client, _, _ = harness
        write_state(tmp_path, cycles=4)
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path)
        body = client.get("/reconciliation").json()
        assert body["checked"] is True
        assert body["cycles"] == 4

    def test_a_halt_survives_into_the_endpoint(self, harness, monkeypatch, tmp_path):
        from tests.test_snapshot import write_state
        from trading.paper.state import PaperStateStore

        client, _, _ = harness
        write_state(tmp_path)
        store = PaperStateStore(tmp_path)
        state = store.restore()
        state.engage_halt("broker disagrees on 3 names")
        store.save(state)
        monkeypatch.setattr("apps.api.book.DEFAULT_STATE_DIR", tmp_path)
        body = client.get("/reconciliation").json()
        assert body["halted"] is True
        assert "broker disagrees" in body["halt_reason"]


class TestEnvironmentIsNotAsserted:
    def test_health_reports_the_configured_environment(self, harness):
        """The console badge was hardcoded to DEV. It exists to tell the
        operator which environment they are acting in, so asserting one is the
        worst thing it could do — it now reads this field."""
        client, _, _ = harness
        assert client.get("/health").json()["environment"] in {"dev", "paper", "live"}
