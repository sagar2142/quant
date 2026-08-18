"""Live-trading safety guards — MASTER_PLAN §21, §23.

The most important test file in the repository. Everything here exists to
answer one question: **can a real order leave this system by accident?**

Each guard is tested in isolation, because they are deliberately redundant and
a redundancy that only works collectively is not a redundancy.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal

import httpx
import pytest

from core.clock import UTC
from core.config import Environment, Settings
from core.instruments import AssetClass, Currency, Exchange, Instrument, InstrumentId
from core.orders import OrderType, Side
from core.secrets import (
    BrokerCredentials,
    MissingCredentialError,
    SecretValue,
    load_broker_credentials,
    load_market_data_credentials,
)
from trading.execution.broker import BrokerError
from trading.execution.orders import Order, TradingMode

A = InstrumentId("NSE:RELIANCE")
T0 = datetime(2024, 6, 3, 10, 0, tzinfo=UTC)

INSTRUMENT = Instrument(
    instrument_id=A,
    symbol="RELIANCE",
    asset_class=AssetClass.EQUITY,
    exchange=Exchange.NSE,
    currency=Currency.INR,
    tick_size=Decimal("0.05"),
)
INSTRUMENTS = {A: INSTRUMENT}

CREDENTIALS = BrokerCredentials(
    broker="kite",
    api_key=SecretValue("key"),
    api_secret=SecretValue("secret"),
    access_token=SecretValue("token"),
)


def live_order(**overrides) -> Order:
    defaults = dict(
        strategy_id="s1",
        instrument_id=A,
        side=Side.BUY,
        quantity=Decimal(10),
        order_type=OrderType.MARKET,
        mode=TradingMode.LIVE,
        decision_time=T0,
    )
    return Order(**{**defaults, **overrides})


@pytest.fixture
def live_env(monkeypatch):
    """A correctly configured live environment. Deliberately awkward to reach."""
    import core.config
    import trading.execution.kite

    enabled = Settings(env=Environment.LIVE, live_enabled=True)
    monkeypatch.setattr(core.config, "settings", enabled)
    monkeypatch.setattr(trading.execution.kite, "settings", enabled)
    return enabled


class TestSecretValue:
    """A credential that reaches a log line has leaked."""

    def test_repr_hides_the_value(self):
        assert "hunter2" not in repr(SecretValue("hunter2"))
        assert repr(SecretValue("hunter2")) == "SecretValue(***)"

    def test_str_hides_the_value(self):
        assert str(SecretValue("hunter2")) == "***"

    def test_fstring_interpolation_hides_the_value(self):
        secret = SecretValue("hunter2")
        assert "hunter2" not in f"token={secret}"

    def test_dataclass_repr_hides_the_value(self):
        # The mundane leak path: a config object echoed in a traceback.
        assert "token" not in repr(CREDENTIALS).replace("access_token", "")

    def test_reveal_returns_it(self):
        assert SecretValue("hunter2").reveal() == "hunter2"

    def test_empty_is_falsy(self):
        assert not SecretValue("")
        assert not SecretValue("   ")
        assert SecretValue("x")

    def test_comparison_to_plain_string_is_refused(self):
        assert SecretValue("a").__eq__("a") is NotImplemented


class TestCredentialLoading:
    def test_missing_credential_raises_not_empties(self, monkeypatch):
        monkeypatch.delenv("KITE_API_KEY", raising=False)
        with pytest.raises(MissingCredentialError, match="KITE_API_KEY"):
            load_market_data_credentials("kite")

    def test_error_says_how_to_fix_it(self, monkeypatch):
        monkeypatch.delenv("KITE_API_KEY", raising=False)
        with pytest.raises(MissingCredentialError, match="never commit"):
            load_market_data_credentials("kite")

    def test_alpaca_keys_are_required(self, monkeypatch):
        # Alpaca paper keys grant data and paper order entry, so they are
        # required rather than optional.
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        with pytest.raises(MissingCredentialError, match="ALPACA_API_KEY"):
            load_market_data_credentials("alpaca")

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="unknown market data provider"):
            load_market_data_credentials("nasdaq")


class TestBrokerCredentialsRefuseOutsideLive:
    """Guard 1, at the credential layer."""

    def test_dev_environment_cannot_load_broker_credentials(self):
        # settings defaults to env=dev, live_enabled=false.
        with pytest.raises(PermissionError, match="live trading blocked"):
            load_broker_credentials("kite")

    def test_live_env_without_explicit_enable_is_refused(self, monkeypatch):
        import core.config

        monkeypatch.setattr(
            core.config, "settings", Settings(env=Environment.LIVE, live_enabled=False)
        )
        with pytest.raises(PermissionError, match="live_enabled"):
            load_broker_credentials("kite")

    def test_enabled_flag_alone_is_not_enough(self, monkeypatch):
        import core.config

        monkeypatch.setattr(
            core.config, "settings", Settings(env=Environment.DEV, live_enabled=True)
        )
        with pytest.raises(PermissionError):
            load_broker_credentials("kite")


class TestKiteBrokerGuards:
    def test_construction_refused_outside_live(self):
        """Guard 1: a misconfigured process dies at startup, not mid-session."""
        from trading.execution.kite import KiteBroker

        with pytest.raises(PermissionError, match="live trading blocked"):
            KiteBroker(CREDENTIALS, INSTRUMENTS)

    def test_empty_access_token_refused(self, live_env):
        from trading.execution.kite import KiteBroker

        stale = BrokerCredentials(
            broker="kite",
            api_key=SecretValue("k"),
            api_secret=SecretValue("s"),
            access_token=SecretValue(""),
        )
        with pytest.raises(BrokerError, match="access token is empty"):
            KiteBroker(stale, INSTRUMENTS)

    def test_paper_order_refused(self, live_env):
        """Guard 2: a mode mismatch is how a paper order reaches a real venue."""
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(CREDENTIALS, INSTRUMENTS, client=self._stub())
        with pytest.raises(BrokerError, match="refuses a PAPER order"):
            broker.submit(live_order(mode=TradingMode.PAPER), Decimal(1000))

    def test_backtest_order_refused(self, live_env):
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(CREDENTIALS, INSTRUMENTS, client=self._stub())
        with pytest.raises(BrokerError, match="refuses a BACKTEST order"):
            broker.submit(live_order(mode=TradingMode.BACKTEST), Decimal(1000))

    def test_session_budget_exhausts_rather_than_resets(self, live_env):
        """Guard 4: a circuit breaker against a runaway loop."""
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            session_notional_budget=Decimal(25_000),
            client=self._stub(),
        )
        broker.submit(live_order(quantity=Decimal(10)), Decimal(1000))  # 10,000
        broker.submit(live_order(quantity=Decimal(10)), Decimal(1000))  # 20,000
        assert broker.remaining_budget == Decimal(5_000)
        with pytest.raises(BrokerError, match="budget exhausted"):
            broker.submit(live_order(quantity=Decimal(10)), Decimal(1000))

    def test_budget_message_says_a_restart_is_needed(self, live_env):
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(
            CREDENTIALS, INSTRUMENTS, session_notional_budget=Decimal(1), client=self._stub()
        )
        with pytest.raises(BrokerError, match="Restart the process deliberately"):
            broker.submit(live_order(), Decimal(1000))

    def test_lot_size_violation_refused(self, live_env):
        from trading.execution.kite import KiteBroker

        lot_instrument = Instrument(
            instrument_id=A,
            symbol="NIFTY",
            asset_class=AssetClass.EQUITY,
            exchange=Exchange.NSE,
            currency=Currency.INR,
            tick_size=Decimal("0.05"),
            lot_size=50,
        )
        broker = KiteBroker(CREDENTIALS, {A: lot_instrument}, client=self._stub())
        with pytest.raises(BrokerError, match="not a multiple of lot size"):
            broker.submit(live_order(quantity=Decimal(30)), Decimal(100))

    def test_unknown_instrument_refused(self, live_env):
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(CREDENTIALS, {}, client=self._stub())
        with pytest.raises(BrokerError, match="unknown instrument"):
            broker.submit(live_order(), Decimal(1000))

    def test_network_failure_says_unknown_not_failed(self, live_env):
        """§19: the order may or may not have reached the venue."""
        from trading.execution.kite import KiteBroker

        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host")

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(BrokerError, match="UNKNOWN and must be reconciled"):
            broker.submit(live_order(), Decimal(1000))

    def test_venue_rejection_is_loud(self, live_env):
        from trading.execution.kite import KiteBroker

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="insufficient funds")

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(BrokerError, match="insufficient funds"):
            broker.submit(live_order(), Decimal(1000))

    def test_idempotency_key_is_sent_as_a_tag(self, live_env):
        """A second line of defence behind our own UNIQUE constraint (§19)."""
        from trading.execution.kite import KiteBroker

        captured: list[bytes] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.content)
            return httpx.Response(200, json={"data": {"order_id": "251103000123456"}})

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        order = live_order()
        broker.submit(order, Decimal(1000))
        assert order.idempotency_key[:20] in captured[0].decode()

    def test_successful_submit_returns_venue_id(self, live_env):
        from trading.execution.kite import KiteBroker

        broker = KiteBroker(CREDENTIALS, INSTRUMENTS, client=self._stub())
        assert broker.submit(live_order(), Decimal(1000)) == "251103000123456"

    def test_no_bulk_liquidation_methods_exist(self, live_env):
        """Convenience on a live adapter is how a typo liquidates a book."""
        from trading.execution.kite import KiteBroker

        for forbidden in ("cancel_all", "close_all_positions", "flatten", "liquidate"):
            assert not hasattr(KiteBroker, forbidden)

    @staticmethod
    def _stub() -> httpx.Client:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": {"order_id": "251103000123456"}})

        return httpx.Client(transport=httpx.MockTransport(handler))


class TestPositionsAndFills:
    def test_zero_positions_are_omitted(self, live_env):
        from trading.execution.kite import KiteBroker

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "net": [
                            {"tradingsymbol": "RELIANCE", "quantity": 10, "average_price": 2900.0},
                            {"tradingsymbol": "TCS", "quantity": 0, "average_price": 3800.0},
                        ]
                    }
                },
            )

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        positions = broker.positions()
        assert len(positions) == 1
        assert positions[0].quantity == Decimal(10)

    def test_fills_parsed(self, live_env):
        from trading.execution.kite import KiteBroker

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "trade_id": "t1",
                            "order_id": "o1",
                            "tradingsymbol": "RELIANCE",
                            "transaction_type": "BUY",
                            "quantity": 10,
                            "average_price": 2900.5,
                        }
                    ]
                },
            )

        broker = KiteBroker(
            CREDENTIALS,
            INSTRUMENTS,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        fills = broker.fills_since(None)
        assert len(fills) == 1
        assert fills[0].price == Decimal("2900.5")
        assert fills[0].side is Side.BUY


def mock_broker(handler, live_env=None):
    """A KiteBroker whose HTTP goes to `handler` instead of the venue."""
    from trading.execution.kite import KiteBroker

    return KiteBroker(
        CREDENTIALS,
        INSTRUMENTS,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def responder(status: int = 200, payload: object = None):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload if payload is not None else {})

    return handler


def exploder(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("network down")


class TestNetworkFailureIsUnknownNotAssumed:
    """§19 — the one case where guessing is the expensive mistake.

    A submit that fails mid-flight may or may not have reached the venue.
    "Assume rejected" resends and doubles the position; "assume filled" leaves
    the account holding nothing while believing otherwise. The adapter must
    say UNKNOWN and force reconciliation.
    """

    def test_submit_network_failure_names_unknown(self, live_env):
        broker = mock_broker(exploder)
        with pytest.raises(BrokerError, match="UNKNOWN and must be reconciled"):
            broker.submit(live_order(), Decimal(2900))

    def test_budget_is_not_consumed_by_a_failed_submit(self, live_env):
        """A submit that never landed must not eat the session budget — the
        budget is a cap on what can reach the market, not on attempts."""
        broker = mock_broker(exploder)
        before = broker.remaining_budget
        with pytest.raises(BrokerError):
            broker.submit(live_order(), Decimal(2900))
        assert broker.remaining_budget == before

    def test_cancel_network_failure_is_reported(self, live_env):
        with pytest.raises(BrokerError, match="network failure cancelling"):
            mock_broker(exploder).cancel("o1")

    def test_positions_network_failure_is_reported(self, live_env):
        """Reconciliation that cannot reach the venue must fail loudly, never
        return an empty book — an empty book reads as "flat" (§9)."""
        with pytest.raises(BrokerError, match="could not fetch positions"):
            mock_broker(exploder).positions()

    def test_fills_network_failure_is_reported(self, live_env):
        with pytest.raises(BrokerError, match="could not fetch trades"):
            mock_broker(exploder).fills_since(None)


class TestVenueRejections:
    def test_non_200_submit_is_refused(self, live_env):
        broker = mock_broker(responder(400, {"message": "insufficient funds"}))
        with pytest.raises(BrokerError, match="Kite rejected order"):
            broker.submit(live_order(), Decimal(2900))

    def test_missing_order_id_is_refused(self, live_env):
        """A 200 with no order_id means we do not know what was placed."""
        broker = mock_broker(responder(200, {"data": {}}))
        with pytest.raises(BrokerError, match="no order_id"):
            broker.submit(live_order(), Decimal(2900))

    def test_non_200_cancel_is_refused(self, live_env):
        with pytest.raises(BrokerError, match="cancel of o1 failed"):
            mock_broker(responder(500)).cancel("o1")

    def test_non_200_positions_is_refused(self, live_env):
        with pytest.raises(BrokerError, match="positions fetch failed"):
            mock_broker(responder(503)).positions()

    def test_non_200_trades_is_refused(self, live_env):
        with pytest.raises(BrokerError, match="trades fetch failed"):
            mock_broker(responder(503)).fills_since(None)


class TestSuccessfulLivePath:
    def test_a_clean_submit_returns_the_venue_id(self, live_env):
        broker = mock_broker(responder(200, {"data": {"order_id": "251118000123"}}))
        assert broker.submit(live_order(), Decimal(2900)) == "251118000123"

    def test_budget_is_consumed_by_a_successful_submit(self, live_env):
        broker = mock_broker(responder(200, {"data": {"order_id": "x"}}))
        before = broker.remaining_budget
        broker.submit(live_order(quantity=Decimal(10)), Decimal(2900))
        assert broker.remaining_budget == before - Decimal(29_000)

    def test_a_limit_order_sends_a_tick_rounded_price(self, live_env):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(httpx.QueryParams(request.content.decode())))
            return httpx.Response(200, json={"data": {"order_id": "x"}})

        broker = mock_broker(handler)
        broker.submit(
            live_order(order_type=OrderType.LIMIT, limit_price=Decimal("2900.123")),
            Decimal(2900),
        )
        # NSE trades in 5-paisa ticks; an unrounded price is rejected outright.
        assert seen["price"] == "2900.10"

    def test_a_stop_order_sends_a_trigger_price(self, live_env):
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(httpx.QueryParams(request.content.decode())))
            return httpx.Response(200, json={"data": {"order_id": "x"}})

        mock_broker(handler).submit(
            live_order(order_type=OrderType.STOP, stop_price=Decimal("2850.07")),
            Decimal(2900),
        )
        assert seen["trigger_price"] == "2850.05"

    def test_the_idempotency_key_is_sent_as_the_tag(self, live_env):
        """Second line of defence behind our own UNIQUE constraint (§19)."""
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(httpx.QueryParams(request.content.decode())))
            return httpx.Response(200, json={"data": {"order_id": "x"}})

        placed = live_order()
        mock_broker(handler).submit(placed, Decimal(2900))
        assert seen["tag"] == placed.idempotency_key[:20]

    def test_a_clean_cancel_returns(self, live_env):
        mock_broker(responder(200, {"data": {"order_id": "o1"}})).cancel("o1")

    def test_mode_is_live(self, live_env):
        assert mock_broker(responder()).mode is TradingMode.LIVE


class TestFillMarker:
    def payload(self) -> object:
        return {
            "data": [
                {
                    "trade_id": f"t{i}",
                    "order_id": f"o{i}",
                    "tradingsymbol": "RELIANCE",
                    "transaction_type": "BUY" if i % 2 else "SELL",
                    "quantity": 10,
                    "average_price": 2900.0 + i,
                }
                for i in range(3)
            ]
        }

    def test_marker_returns_only_later_fills(self, live_env):
        broker = mock_broker(responder(200, self.payload()))
        assert [f.broker_fill_id for f in broker.fills_since("t0")] == ["t1", "t2"]

    def test_unknown_marker_returns_everything(self, live_env):
        """Fail safe: replaying a fill is caught downstream, skipping one is
        invisible."""
        broker = mock_broker(responder(200, self.payload()))
        assert len(broker.fills_since("t-not-ours")) == 3

    def test_sell_side_is_parsed(self, live_env):
        broker = mock_broker(responder(200, self.payload()))
        assert broker.fills_since(None)[0].side is Side.SELL


class TestConnectionsAreNotLeaked:
    """When the adapter creates its own client it must also close it.

    A leaked connection per call is invisible until the process has run for a
    session and hits the file-descriptor ceiling — at which point orders start
    failing for a reason that looks nothing like the cause.
    """

    def owned_client_broker(self, monkeypatch, handler):
        """A broker with `client=None`, so it constructs (and must close) its own."""
        import trading.execution.kite as kite_module

        created: list[httpx.Client] = []
        real_client = httpx.Client

        def factory(*args, **kwargs):
            kwargs.pop("timeout", None)
            client = real_client(transport=httpx.MockTransport(handler))
            created.append(client)
            return client

        monkeypatch.setattr(kite_module.httpx, "Client", factory)
        return kite_module.KiteBroker(CREDENTIALS, INSTRUMENTS), created

    def test_submit_closes_its_own_client(self, live_env, monkeypatch):
        broker, created = self.owned_client_broker(
            monkeypatch, responder(200, {"data": {"order_id": "x"}})
        )
        broker.submit(live_order(), Decimal(2900))
        assert created and all(c.is_closed for c in created)

    def test_cancel_closes_its_own_client(self, live_env, monkeypatch):
        broker, created = self.owned_client_broker(monkeypatch, responder(200, {"data": {}}))
        broker.cancel("o1")
        assert created and all(c.is_closed for c in created)

    def test_positions_closes_its_own_client(self, live_env, monkeypatch):
        broker, created = self.owned_client_broker(
            monkeypatch, responder(200, {"data": {"net": []}})
        )
        broker.positions()
        assert created and all(c.is_closed for c in created)

    def test_fills_closes_its_own_client(self, live_env, monkeypatch):
        broker, created = self.owned_client_broker(monkeypatch, responder(200, {"data": []}))
        broker.fills_since(None)
        assert created and all(c.is_closed for c in created)

    def test_an_injected_client_is_left_open(self, live_env):
        """The caller owns what the caller supplied — closing it would break
        the next call on a shared session."""
        client = httpx.Client(transport=httpx.MockTransport(responder(200, {"data": {"net": []}})))
        from trading.execution.kite import KiteBroker

        KiteBroker(CREDENTIALS, INSTRUMENTS, client=client).positions()
        assert not client.is_closed


class TestCredentialsNeverSerialised:
    def test_credentials_are_not_json_serialisable(self):
        """A credential must not slip into a log payload or an API response."""
        with pytest.raises(TypeError):
            json.dumps({"key": SecretValue("hunter2")})
