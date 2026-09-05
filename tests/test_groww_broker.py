"""The Groww adapter — MASTER_PLAN §8, §21.

No test here touches the network. What is worth pinning is not that a happy
path parses, but that the three ways this broker differs from Kite are handled
rather than papered over: the daily token expiry, the absent trades feed, and
symbol-versus-ISIN addressing.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import httpx
import pytest

from core.clock import UTC
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from core.orders import OrderType, Side
from core.secrets import BrokerCredentials, SecretValue
from trading.execution.broker import BrokerError
from trading.execution.orders import Order, TradingMode

RELIANCE = InstrumentId("NSE:INE002A01018")


def credentials(token: str = "a-live-token", secret: str = "a-secret") -> BrokerCredentials:
    """What Groww actually issues: a key and a secret. The token is derived."""
    return BrokerCredentials(
        broker="groww",
        api_key=SecretValue("a-key"),
        api_secret=SecretValue(secret),
        access_token=SecretValue(token),
    )


def instruments() -> dict[InstrumentId, Instrument]:
    return {
        RELIANCE: Instrument(
            instrument_id=RELIANCE,
            symbol="RELIANCE",
            asset_class=AssetClass.EQUITY,
            exchange=Exchange.NSE,
            currency=Currency.INR,
            tick_size=Decimal("0.05"),
        )
    }


def an_order(quantity: int = 1) -> Order:
    return Order(
        strategy_id="test",
        instrument_id=RELIANCE,
        side=Side.BUY,
        quantity=Decimal(quantity),
        order_type=OrderType.MARKET,
        mode=TradingMode.LIVE,
        decision_time=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
    )


def transport(handler) -> httpx.Client:
    """An httpx client backed by a handler instead of a socket."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture
def live(monkeypatch):
    """Pretend the environment permits live trading.

    The adapter refuses to construct otherwise, which is the correct default
    and is asserted separately below.
    """
    from core.config import settings

    monkeypatch.setattr(type(settings), "require_live_permission", lambda self: None)
    return settings


def broker(live, handler, credentials_=None, **kwargs):
    from trading.execution.groww import GrowwBroker

    _ = live
    return GrowwBroker(
        credentials=credentials_ or credentials(),
        instruments=instruments(),
        client=transport(handler),
        **kwargs,
    )


class TestItRefusesToExist:
    def test_construction_needs_live_permission(self, monkeypatch):
        """The guard fires at construction, so a misconfigured process dies at
        startup rather than on its first order."""
        from core.config import settings
        from trading.execution.groww import GrowwBroker

        def refuse(self) -> None:
            raise PermissionError("live trading blocked")

        monkeypatch.setattr(type(settings), "require_live_permission", refuse)
        with pytest.raises(PermissionError):
            GrowwBroker(credentials=credentials(), instruments=instruments())

    def test_neither_a_token_nor_a_secret_is_refused(self, live):
        """A key alone cannot trade: without a secret there is nothing to mint
        a token from, and without a token there is nothing to present."""
        from trading.execution.groww import GrowwBroker

        with pytest.raises(BrokerError, match="API key and secret"):
            GrowwBroker(credentials=credentials(token="", secret=""), instruments=instruments())

    def test_a_secret_alone_is_enough(self, live):
        """The ordinary case. Groww hands out a key and a secret; the token is
        minted from them on first use, so demanding one up front was demanding
        a value the operator never had."""
        from trading.execution.groww import GrowwBroker

        assert GrowwBroker(credentials=credentials(token=""), instruments=instruments())


class TestTheDailyToken:
    def test_a_401_mints_a_fresh_token_and_retries(self, live):
        """The token dies at 06:00 IST every morning, mid-session for anyone
        who started before it. One re-mint and a retry, rather than an error
        the operator has to act on."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url.path))
            if request.url.path.endswith("/token/api/access"):
                return httpx.Response(200, json={"token": "a-fresh-token"})
            # Unauthorised once, then accepted with the new token.
            if seen.count("/v1/order/list") == 1:
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(200, json={"payload": {"order_list": []}})

        assert broker(live, handler, credentials_=credentials(token="")).orders() == []
        assert any(path.endswith("/token/api/access") for path in seen), "should have minted"

    def test_a_second_401_is_a_real_failure(self, live):
        """One retry, not a loop. A key that is simply wrong must surface as
        wrong rather than as an endless re-mint."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/token/api/access"):
                return httpx.Response(200, json={"token": "a-fresh-token"})
            return httpx.Response(401, json={"error": "unauthorized"})

        with pytest.raises(BrokerError, match="Check the API key and secret"):
            broker(live, handler, credentials_=credentials(token="")).orders()

    def test_other_failures_are_not_called_a_token_problem(self, live):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream exploded")

        with pytest.raises(BrokerError, match="HTTP 500"):
            broker(live, handler).orders()


class TestFillsAreNotAvailable:
    def test_it_raises_rather_than_returning_an_empty_list(self, live):
        """The load-bearing test.

        Groww documents no trades endpoint. An empty list is the claim
        "nothing filled", which this adapter cannot establish, and
        reconciliation would believe it — which is how a real fill goes
        unnoticed (§9).
        """

        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
            return httpx.Response(200, json={})

        with pytest.raises(BrokerError, match="no trades endpoint"):
            broker(live, handler).fills_since(None)


class TestOrders:
    def test_it_sends_the_symbol_not_the_isin(self, live):
        """Groww addresses orders by trading symbol; this system keys positions
        on ISIN. Sending the id would be an order the venue cannot match."""
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"payload": {"groww_order_id": "GRW-1"}})

        assert broker(live, handler).submit(an_order(), Decimal(1300)) == "GRW-1"
        assert seen["trading_symbol"] == "RELIANCE"
        assert seen["segment"] == "CASH"
        assert seen["transaction_type"] == "BUY"

    def test_an_unknown_symbol_is_refused_rather_than_guessed(self, live):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={})

        from trading.execution.groww import GrowwBroker

        empty = GrowwBroker(credentials=credentials(), instruments={}, client=transport(handler))
        with pytest.raises(BrokerError, match="no trading symbol known"):
            empty.submit(an_order(), Decimal(1300))

    def test_a_response_without_an_id_is_an_error(self, live):
        """Never fabricate an order id. A caller that believes an order exists
        when it does not is worse off than one that saw an error."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"payload": {}})

        with pytest.raises(BrokerError, match="returned no id"):
            broker(live, handler).submit(an_order(), Decimal(1300))

    def test_the_session_budget_is_a_circuit_breaker(self, live):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            return httpx.Response(200, json={"payload": {"groww_order_id": "GRW-1"}})

        small = broker(live, handler, session_notional_budget=Decimal(1000))
        with pytest.raises(BrokerError, match="session budget"):
            small.submit(an_order(quantity=10), Decimal(1300))


class TestPositions:
    def test_positions_are_keyed_on_isin(self, live):
        """§1.1: a symbol can wear more than one ISIN over its life, so the
        ISIN is the identity wherever Groww supplies it."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "payload": {
                        "positions": [
                            {
                                "trading_symbol": "RELIANCE",
                                "symbol_isin": "INE002A01018",
                                "quantity": 5,
                                "net_price": 1287.5,
                            }
                        ]
                    }
                },
            )

        held = broker(live, handler).positions()
        assert len(held) == 1
        assert str(held[0].instrument_id) == "NSE:INE002A01018"
        assert held[0].average_price == Decimal("1287.5")

    def test_flat_positions_are_dropped(self, live):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"payload": {"positions": [{"trading_symbol": "X", "quantity": 0}]}},
            )

        assert broker(live, handler).positions() == []

    def test_holdings_are_not_positions(self, live):
        """A position is today's net exposure; a holding is what is in demat.
        They differ for anything bought today and unsettled."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "payload": {
                        "holdings": [
                            {"isin": "INE002A01018", "quantity": 12, "average_price": 1200.0}
                        ]
                    }
                },
            )

        held = broker(live, handler).holdings()
        assert held[0].quantity == Decimal(12)
        assert held[0].average_price == Decimal("1200.0")


class TestResponseShapes:
    def test_an_unrecognised_shape_yields_nothing_rather_than_raising(self, live):
        """A portfolio screen that cannot parse one endpoint should still
        render the rest."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"payload": {"unexpected": "shape"}})

        assert broker(live, handler).positions() == []

    def test_a_non_json_body_is_named(self, live):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>maintenance</html>")

        with pytest.raises(BrokerError, match="non-JSON"):
            broker(live, handler).orders()
