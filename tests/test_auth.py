"""Console API authentication (§13.7, §21).

The design under test: reads are open until a token is configured, mutations
are refused until one is. Both halves matter, and the second is the one that
protects money.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

import core.config as config_module
from core.config import Settings

KILL = {"reason": "testing the guard", "operator": "sagar"}
TOKEN = "a-token-of-sufficient-length"


@pytest.fixture
def client(request):
    """A client whose app was built under a given token setting.

    The token is read at request time from the process settings, so the module
    is reloaded to pick up the substitution rather than patched in place.
    """
    token = getattr(request, "param", "")
    original = config_module.settings
    config_module.settings = Settings(api_token=token)

    import apps.api.auth as auth_module

    importlib.reload(auth_module)
    import apps.api.main as main_module

    importlib.reload(main_module)
    yield TestClient(main_module.create_app())

    config_module.settings = original
    importlib.reload(auth_module)
    importlib.reload(main_module)


def bearer(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestUnconfigured:
    """A fresh checkout: browse freely, but you cannot touch the halt."""

    @pytest.mark.parametrize("client", [""], indirect=True)
    def test_reads_are_open(self, client):
        assert client.get("/health").status_code == 200

    @pytest.mark.parametrize("client", [""], indirect=True)
    def test_engaging_a_halt_is_unavailable(self, client):
        assert client.post("/kill", json=KILL).status_code == 503

    @pytest.mark.parametrize("client", [""], indirect=True)
    def test_releasing_a_halt_is_unavailable(self, client):
        """The direction that matters. Engaging a halt is recoverable;
        releasing one puts a book known to be wrong back in the market."""
        assert client.post("/kill/release", json=KILL).status_code == 503

    @pytest.mark.parametrize("client", [""], indirect=True)
    def test_the_refusal_says_how_to_enable_it(self, client):
        """A 503 with no remedy is a dead end."""
        detail = client.post("/kill", json=KILL).json()["detail"]
        assert "NEUTRON_API_TOKEN" in detail


class TestConfigured:
    """Once a token exists it is required everywhere, reads included."""

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_reads_now_require_the_token(self, client):
        """A deployment worth protecting is protected uniformly — otherwise it
        serves its own position book to anything that asks."""
        assert client.get("/health").status_code == 401
        assert client.get("/health", headers=bearer()).status_code == 200

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_the_kill_switch_accepts_the_token(self, client):
        assert client.post("/kill", json=KILL, headers=bearer()).status_code == 200

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_a_wrong_token_is_refused(self, client):
        assert client.post("/kill", json=KILL, headers=bearer("wrong")).status_code == 401

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_a_missing_token_is_refused(self, client):
        assert client.post("/kill", json=KILL).status_code == 401

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_missing_and_wrong_are_indistinguishable(self, client):
        """Different messages would confirm to a prober that a token exists
        and that theirs was merely wrong."""
        missing = client.get("/health").json()
        wrong = client.get("/health", headers=bearer("wrong")).json()
        assert missing == wrong

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_a_bare_token_is_accepted(self, client):
        """A curl one-liner is how this gets poked during operations, and a
        scheme that only works from the console is one that gets bypassed."""
        assert client.get("/health", headers={"Authorization": TOKEN}).status_code == 200

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_the_challenge_header_is_sent(self, client):
        assert client.get("/health").headers.get("www-authenticate") == "Bearer"

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_analytics_reads_are_guarded_too(self, client):
        """Not just the obvious endpoints: the book is position data."""
        assert client.get("/book").status_code == 401
        assert client.get("/risk/limits").status_code == 401


class TestTokenComparison:
    def test_comparison_is_constant_time(self):
        """A byte-at-a-time comparison leaks the token under timing analysis."""
        import inspect

        import apps.api.auth as auth_module

        assert "compare_digest" in inspect.getsource(auth_module)

    @pytest.mark.parametrize("client", [TOKEN], indirect=True)
    def test_a_token_prefix_is_not_enough(self, client):
        assert client.get("/health", headers=bearer(TOKEN[:-1])).status_code == 401
