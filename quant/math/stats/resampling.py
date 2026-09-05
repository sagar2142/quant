"""Monte Carlo trade shuffling — MASTER_PLAN §5.4 test 11.

Shuffling the *order* of trades preserves the return distribution exactly while
destroying the sequence, which isolates how much of a drawdown was sequence
luck. A strategy whose 5th-percentile shuffled drawdown is twice its realised
one got lucky in the ordering.

**The seed is explicit** (§14.1.1). An unseeded resampling test gives a
different answer each run, which means it can be re-rolled until it agrees with
you — the exact failure mode the gauntlet exists to prevent.

This module once also carried an IID bootstrap, a block bootstrap and a
permutation test. All three were written, exported, never called and never
tested — 40% of the file was unreachable. They are the sort of thing that reads
as capability on a shelf and behaves as untested code the first time someone
reaches for it, so they are gone. `quant.math.metrics.overfitting` holds the
resampling the gauntlet actually uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from quant.math.metrics.performance import max_drawdown

__all__ = ["ShuffleResult", "monte_carlo_drawdown"]

FloatArray = npt.NDArray[np.float64]

#: Below this many trades a shuffled drawdown is noise, not evidence.
MIN_TRADES_SHUFFLE = 2


@dataclass(frozen=True)
class ShuffleResult:
    realised_drawdown: float
    median_drawdown: float
    percentile_5: float
    n_shuffles: int

    @property
    def sequence_luck_ratio(self) -> float:
        """How much worse the unlucky case is than what actually happened.

        Above ~2.0 means the realised drawdown owes a lot to favourable
        ordering, and the risk limits should be set against the shuffled tail
        rather than against history.
        """
        if self.realised_drawdown == 0.0:
            return 0.0
        return self.percentile_5 / self.realised_drawdown

    def format(self) -> str:
        return (
            f"  realised DD {self.realised_drawdown:.2%}, "
            f"median {self.median_drawdown:.2%}, "
            f"5th pct {self.percentile_5:.2%} "
            f"(luck ratio {self.sequence_luck_ratio:.2f})"
        )


def monte_carlo_drawdown(
    trade_returns: npt.ArrayLike,
    seed: int,
    n_shuffles: int = 2000,
) -> ShuffleResult:
    """Drawdown distribution under random trade ordering (§5.4 test 11).

    Preserves the exact set of trade outcomes and destroys only their sequence.
    Whatever drawdown you actually experienced was one draw from this
    distribution; the 5th percentile is a far better basis for a risk limit
    than the single path history happened to take.
    """
    trades = np.asarray(trade_returns, dtype=np.float64).ravel()
    trades = trades[np.isfinite(trades)]
    if trades.size < MIN_TRADES_SHUFFLE:
        return ShuffleResult(0.0, 0.0, 0.0, 0)

    rng = np.random.default_rng(seed)
    drawdowns = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        drawdowns[i] = max_drawdown(rng.permutation(trades))

    return ShuffleResult(
        realised_drawdown=max_drawdown(trades),
        median_drawdown=float(np.median(drawdowns)),
        percentile_5=float(np.quantile(drawdowns, 0.05)),
        n_shuffles=n_shuffles,
    )
