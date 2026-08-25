"""Factor risk model — MASTER_PLAN §6, §8.

The system could score a signal and report pairwise factor correlation, and
could not say how much of a *portfolio's* risk came from a factor. That gap
mattered concretely: the four factors surviving costs are all momentum, two of
them correlate at 0.97, and every screen presented the resulting book as four
bets rather than one.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from core.clock import UTC
from quant.research.factors import Factor
from quant.research.riskmodel import (
    MARKET,
    MAX_CONDITION,
    MIN_SESSIONS,
    RiskDecomposition,
    RiskModel,
    build_risk_model,
    decompose,
)


def panel(series: dict[str, np.ndarray], volume: float = 1e8) -> pl.DataFrame:
    length = len(next(iter(series.values())))
    times = [datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(length)]
    return pl.concat(
        [
            pl.DataFrame(
                {
                    "event_time": times,
                    "symbol": [name] * length,
                    "instrument_id": [f"NSE:INE{name}"] * length,
                    "open": [closes[max(0, i - 1)] for i in range(length)],
                    "high": [c * 1.01 for c in closes],
                    "low": [c * 0.99 for c in closes],
                    "close": list(closes),
                    "volume": [volume] * length,
                    "trades": [1000.0] * length,
                },
                schema_overrides={"event_time": pl.Datetime("us", "UTC")},
            )
            for name, closes in series.items()
        ]
    )


def market(names: int = 40, sessions: int = 700, seed: int = 7) -> pl.DataFrame:
    """A market with one shared driver, so a factor structure genuinely exists.

    Without a common component every regression slope is noise and the model
    would be tested against a world it is not built for.
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0004, 0.010, sessions)
    series = {}
    for i in range(names):
        beta = 0.4 + 1.4 * (i / names)
        steps = beta * common + rng.normal(0.0002, 0.012, sessions)
        series[f"N{i:02d}"] = 100.0 * np.exp(np.cumsum(steps))
    return panel(series)


class TestEstimation:
    def test_a_model_is_produced_from_a_real_market(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_1M, Factor.REVERSAL_5D))
        assert model is not None
        assert model.sessions >= MIN_SESSIONS

    def test_too_little_history_yields_no_model(self):
        """A covariance from forty sessions would be reported as confidently as
        one from a thousand, and nothing downstream could tell them apart."""
        assert build_risk_model(market(sessions=120), (Factor.MOMENTUM_1M,)) is None

    def test_factor_volatility_is_positive(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_1M, Factor.REVERSAL_5D))
        assert model is not None
        assert all(v > 0 for v in model.factor_volatility.values())

    def test_the_covariance_is_square_and_symmetric(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_1M, Factor.REVERSAL_5D))
        assert model is not None
        # Market plus the styles: `factor_names` is the column order, and
        # `factors` is only the style half of it.
        size = len(model.factor_names)
        assert size == len(model.factors) + 1
        assert model.covariance.shape == (size, size)
        assert np.allclose(model.covariance, model.covariance.T)

    def test_specific_variance_is_never_negative(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_1M,))
        assert model is not None
        assert all(v >= 0 for v in model.specific_variance.values())


class TestDecomposition:
    def model(self) -> RiskModel:
        built = build_risk_model(market(), (Factor.MOMENTUM_1M, Factor.REVERSAL_5D))
        assert built is not None
        return built

    def equal_weights(self, model: RiskModel) -> dict[str, float]:
        return dict.fromkeys(model.exposures, 1.0 / len(model.exposures))

    def test_variance_shares_sum_to_one(self):
        model = self.model()
        result = decompose(model, self.equal_weights(model))
        assert result.factor_share + result.specific_share == pytest.approx(1.0)

    def test_factor_contributions_sum_to_the_factor_share(self):
        """Each factor's share is one row of the same quadratic form, so the
        parts add to the whole rather than being computed independently and
        left not to."""
        model = self.model()
        result = decompose(model, self.equal_weights(model))
        assert sum(result.contributions.values()) == pytest.approx(result.factor_share, abs=1e-9)

    def test_volatilities_compose(self):
        model = self.model()
        result = decompose(model, self.equal_weights(model))
        expected = np.sqrt(result.factor_volatility**2 + result.specific_volatility**2)
        assert result.total_volatility == pytest.approx(expected, rel=1e-9)

    def test_an_unknown_name_contributes_nothing(self):
        """A position the model cannot see contributes no *estimated* risk,
        which is not the same as contributing none."""
        assert decompose(self.model(), {"NSE:INEUNKNOWN": 1.0}).total_volatility == 0.0

    def test_an_empty_book_is_safe(self):
        assert decompose(self.model(), {}).total_volatility == 0.0

    def test_a_concentrated_book_carries_more_specific_risk(self):
        """One name is all idiosyncratic; forty diversify it away."""
        model = self.model()
        names = list(model.exposures)
        one = decompose(model, {names[0]: 1.0})
        many = decompose(model, self.equal_weights(model))
        assert one.specific_share > many.specific_share

    def test_an_equal_weight_book_still_carries_market_risk(self):
        """Style exposures are z-scored, so they sum to zero across the
        universe and an equal-weight book has no style tilt at all. Without a
        market factor its entire variance would read as name-specific, which
        for a long-only book is the opposite of the truth."""
        model = self.model()
        result = decompose(model, self.equal_weights(model))
        assert result.contributions[MARKET] > 0.0
        assert result.factor_share > 0.0

    def test_style_tilts_vanish_on_the_whole_universe(self):
        """The other half of the same fact, asserted directly."""
        model = self.model()
        result = decompose(model, self.equal_weights(model))
        styles = [v for k, v in result.contributions.items() if k != MARKET]
        assert all(abs(v) < 1e-6 for v in styles)


class TestConditioning:
    def test_near_collinear_factors_raise_the_condition_number(self):
        """12-1 and 6-1 momentum share most of their construction."""
        model = build_risk_model(market(), (Factor.MOMENTUM_12_1, Factor.MOMENTUM_6_1))
        assert model is not None
        assert model.condition > 1.0

    def test_the_threshold_is_a_correlation_not_a_scale_measure(self):
        """A covariance's condition number conflates collinearity with scale: a
        26%-volatility factor beside a 5% one is 28x in variance before any
        correlation exists. Flagging a genuinely diversified set would train
        the reader to ignore the warning."""
        assert MAX_CONDITION == 100.0

    def test_a_well_conditioned_model_is_not_flagged(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_1M, Factor.REVERSAL_5D))
        assert model is not None
        assert not model.is_ill_conditioned

    def test_the_warning_reaches_the_report(self):
        model = build_risk_model(market(), (Factor.MOMENTUM_12_1, Factor.MOMENTUM_6_1))
        assert model is not None
        if model.is_ill_conditioned:
            assert "ILL-CONDITIONED" in model.format()


class TestEmptyStates:
    def test_no_factors_yields_no_model(self):
        assert build_risk_model(market(), ()) is None

    def test_an_empty_panel_yields_no_model(self):
        empty = pl.DataFrame(
            schema={
                "event_time": pl.Datetime("us", "UTC"),
                "symbol": pl.String,
                "instrument_id": pl.String,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "trades": pl.Float64,
            }
        )
        assert build_risk_model(empty, (Factor.MOMENTUM_1M,)) is None

    def test_a_zero_decomposition_reports_no_shares(self):
        assert RiskDecomposition(0.0, 0.0, 0.0).specific_share == 0.0
