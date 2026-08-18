"""Ops console API (§13.6, §26).

The kill switch is the only mutating endpoint, so it is the one that gets
tested hardest.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from ops.alerts import Alert, AlertRouter
from trading.risk.engine import RiskEngine


class RecordingSink:
    """Captures alerts instead of sending them."""

    def __init__(self) -> None:
        self.alerts: list[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


@pytest.fixture
def harness():
    engine = RiskEngine()
    sink = RecordingSink()
    client = TestClient(create_app(engine, AlertRouter([sink])))
    return client, engine, sink


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
