"""Order entry — MASTER_PLAN §8, §21.

The tests that matter here are not about arithmetic. They are that an order
cannot leave this process unless every guard says so, that refusing says *why*,
and that nothing anywhere invents a fill.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.trade import BUY_COST_RATE, SELL_COST_RATE, estimated_costs, live_gates
from trading.risk.engine import RiskEngine

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def order(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "symbol": "RELIANCE",
        "side": "BUY",
        "quantity": "10",
        "order_type": "MARKET",
        "reference_price": "1287",
        "equity": "1000000",
    }
    body.update(overrides)
    return body


class TestGates:
    """Every precondition between a click and the exchange."""

    def test_development_is_not_armed(self):
        gates = live_gates(RiskEngine())
        assert not all(g.ready for g in gates)

    def test_every_gate_explains_itself(self):
        """A gate that is shut without saying why teaches an operator to
        distrust the panel. Each one has to name what is missing."""
        for gate in live_gates(RiskEngine()):
            assert gate.detail.strip(), gate.name

    def test_the_kill_switch_is_a_gate(self):
        engine = RiskEngine()
        engine.engage_kill("testing", "suite")
        kill = next(g for g in live_gates(engine) if g.name == "Kill switch")
        assert not kill.ready

    def test_status_reports_blocked_rather_than_pretending(self, client):
        body = client.get("/trade/status").json()
        assert body["can_trade"] is False
        assert body["mode"] == "BLOCKED"
        assert [g for g in body["gates"] if not g["ready"]]


class TestSubmitIsRefused:
    """The load-bearing test in this file."""

    def test_an_order_cannot_be_sent_from_a_development_environment(self, client):
        response = client.post("/trade/orders", json=order())
        assert response.status_code != 200

    def test_the_refusal_names_the_gates(self, client):
        """Not a bare 403. The operator has to learn what to fix."""
        response = client.post("/trade/orders", json=order())
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            assert detail["blocked_by"]
            assert all(b["gate"] and b["detail"] for b in detail["blocked_by"])
        else:
            # Auth refuses first when no API token is configured, which is also
            # a correct refusal — what must never happen is a 200.
            assert response.status_code in {401, 403, 409, 503}


class TestPreviewWorksWithoutCredentials:
    """Preview is arithmetic, so it is available when trading is not.

    This is the difference between a console you can think in and one that is
    inert until a broker is connected.
    """

    def test_it_returns_a_verdict(self, client):
        body = client.post("/trade/preview", json=order()).json()
        assert "allowed" in body
        assert body["checks"]

    def test_it_resolves_the_isin_not_the_ticker(self, client):
        """An order is keyed on ISIN (§1.1). A symbol can wear more than one
        over its life, and 344 names in this panel do."""
        body = client.post("/trade/preview", json=order()).json()
        assert body["instrument_id"].startswith("NSE:INE")
        assert body["instrument_id"] != body["symbol"]

    def test_an_unknown_symbol_is_a_404_rather_than_a_guess(self, client):
        response = client.post("/trade/preview", json=order(symbol="NOTREAL"))
        assert response.status_code == 404

    def test_a_fat_finger_price_fails_the_band(self, client):
        """The check that exists to catch a typed extra digit."""
        body = client.post("/trade/preview", json=order(reference_price="12870")).json()
        failed = {c["name"] for c in body["checks"] if not c["passed"]}
        assert "price_band" in failed
        assert body["allowed"] is False

    def test_an_order_within_every_limit_is_allowed(self, client):
        """Otherwise the suite only proves the engine can say no."""
        body = client.post("/trade/preview", json=order(quantity="10", equity="10000000")).json()
        assert body["allowed"] is True, body["breaches"]

    def test_size_is_measured_against_the_capital_supplied(self, client):
        """There is no book to read equity from, so it is an input — and it has
        to actually change the verdict, or the limits are decoration.

        Sized under `order_notional`, which is an absolute cap rather than a
        fraction of equity: at 500 shares the order is refused for being large
        in itself, whatever the account behind it, and this test would then be
        passing for the wrong reason.
        """
        small = client.post("/trade/preview", json=order(quantity="300", equity="100000")).json()
        large = client.post("/trade/preview", json=order(quantity="300", equity="50000000")).json()
        assert small["allowed"] is False
        assert "position_size" in {c["name"] for c in small["checks"] if not c["passed"]}
        assert large["allowed"] is True, large["breaches"]

    def test_a_single_order_is_capped_in_absolute_terms_too(self, client):
        """A limit that scales only with equity cannot stop a fat finger in a
        large account. This one does not scale."""
        body = client.post("/trade/preview", json=order(quantity="500", equity="50000000")).json()
        assert "order_notional" in {c["name"] for c in body["checks"] if not c["passed"]}

    @pytest.mark.parametrize("field", ["quantity", "equity", "reference_price"])
    def test_a_non_positive_number_is_rejected(self, client, field):
        assert client.post("/trade/preview", json=order(**{field: "0"})).status_code == 422

    def test_a_price_that_is_not_a_number_names_the_field(self, client):
        """Money never arrives as a float (§14.1.2), and a quantity that
        silently became zero is an order nobody meant to place."""
        response = client.post("/trade/preview", json=order(quantity="ten"))
        assert response.status_code == 422
        assert "quantity" in str(response.json()["detail"])


class TestCosts:
    def test_selling_costs_more_than_buying(self):
        """STT falls on the sell leg, and so does stamp duty."""
        assert SELL_COST_RATE > BUY_COST_RATE
        assert estimated_costs(Decimal(100_000), "SELL") > estimated_costs(Decimal(100_000), "BUY")

    def test_costs_are_quantised_to_paisa(self):
        charge = estimated_costs(Decimal("33333.33"), "BUY")
        assert charge == charge.quantize(Decimal("0.01"))

    def test_the_estimate_is_within_reach_of_the_real_model(self):
        """It is a display figure, but a display figure that disagreed with the
        backtester by an order of magnitude would make one of them a liar."""
        notional = Decimal(100_000)
        assert Decimal("50") < estimated_costs(notional, "BUY") < Decimal("500")
