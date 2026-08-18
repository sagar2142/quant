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


class TestCredentialsNeverSerialised:
    def test_credentials_are_not_json_serialisable(self):
        """A credential must not slip into a log payload or an API response."""
        with pytest.raises(TypeError):
            json.dumps({"key": SecretValue("hunter2")})
