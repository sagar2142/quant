"""The pairs screen — MASTER_PLAN §6.

Cointegration was built, tested and unreachable: `engle_granger`, `hedge_ratio`
and `spread_series` had no caller outside their own tests, so a whole strategy
family was library-only.

Searching every pair is also the textbook way to manufacture a false positive,
which is why the symbol list is capped and the trial count is stated.
"""

from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def client() -> TestClient:
    from apps.api.main import app

    return TestClient(app, raise_server_exceptions=False)


BANKS = "HDFCBANK,ICICIBANK,AXISBANK,KOTAKBANK,SBIN,INDUSINDBK"


class TestPairsScreen:
    def test_every_unordered_pair_is_tested(self, client: TestClient) -> None:
        """Which pair cointegrates is exactly what is not known in advance."""
        found = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()
        # 6 names -> 6*5/2 = 15 pairs.
        assert found["tested"] == 15

    def test_one_symbol_is_refused(self, client: TestClient) -> None:
        assert client.get("/pairs?symbols=SBIN").status_code == 422

    def test_unknown_symbols_are_refused_not_emptied(self, client: TestClient) -> None:
        response = client.get("/pairs?symbols=NOTREAL1,NOTREAL2&sessions=500")
        assert response.status_code == 404

    def test_the_verdict_needs_both_tests_to_agree(self, client: TestClient) -> None:
        """A spread can reject the unit root and still not be stationary.

        ADF alone would call that a pair. The report carries the verdict so a
        reader can see why a low p-value did not become a `true`.
        """
        rows = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()["rows"]
        for row in rows:
            if row["cointegrated"]:
                assert row["spread_verdict"] == "STATIONARY"
            else:
                assert row["spread_verdict"] != "STATIONARY"

    def test_tradable_is_stricter_than_cointegrated(self, client: TestClient) -> None:
        """A spread that takes a year to close is a directional position."""
        rows = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()["rows"]
        for row in rows:
            if row["tradable"]:
                assert row["cointegrated"]

    def test_tradable_pairs_sort_first(self, client: TestClient) -> None:
        rows = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()["rows"]
        flags = [r["tradable"] for r in rows]
        assert flags == sorted(flags, reverse=True)

    def test_the_note_declares_the_trial_count(self, client: TestClient) -> None:
        """Fifteen tests and one significant result is not one significant
        result — the count is what makes the p-value readable."""
        found = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()
        assert str(found["tested"]) in found["note"]

    def test_a_half_life_that_is_not_finite_is_null(self, client: TestClient) -> None:
        """Never zero: a spread that does not revert has no half-life, and
        zero would read as one that reverts instantly."""
        rows = client.get(f"/pairs?symbols={BANKS}&sessions=500").json()["rows"]
        for row in rows:
            assert row["half_life_bars"] is None or row["half_life_bars"] > 0
