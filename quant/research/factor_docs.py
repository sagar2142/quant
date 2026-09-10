"""What each factor is, and where it comes from — MASTER_PLAN §6.

Separated from the enum because the prose is longer than the code it annotates.
Every entry names the published effect rather than describing an arithmetic
operation: a factor whose only justification is that it backtested well is the
thing the register exists to reject.
"""

from __future__ import annotations

from quant.research.factors import Factor

__all__ = ["DESCRIPTIONS"]


DESCRIPTIONS: dict[Factor, str] = {
    Factor.MOMENTUM_12_1: (
        "Return from 252 to 21 sessions ago. The classic 12-1 window: "
        "the most recent month is skipped because short-horizon "
        "reversal runs against momentum there and including it "
        "measurably degrades the signal."
    ),
    Factor.MOMENTUM_6_1: (
        "Return from 126 to 21 sessions ago. The same construction as "
        "12-1 over half the lookback, so the two together show whether "
        "the effect is a long-horizon one or a recent-trend one."
    ),
    Factor.MOMENTUM_1M: (
        "Trailing one-month return, unskipped. Included precisely so "
        "the skip in 12-1 can be shown to matter."
    ),
    Factor.REVERSAL_1D: "Negated prior-session return. Fades one-day moves.",
    Factor.REVERSAL_5D: "Negated trailing week. Fades short-horizon moves.",
    Factor.VOLATILITY_60: (
        "Negated 60-session realised volatility, so a high score means "
        "low volatility — the direction the low-volatility anomaly pays."
    ),
    Factor.HIGH_52W_PROXIMITY: (
        "Close over the 252-session high. Near 1.0 means at highs; the "
        "anchoring effect says those keep running."
    ),
    Factor.VOLUME_SHOCK: ("Session volume over its 21-session average. Attention proxy."),
    Factor.ILLIQUIDITY: (
        "Amihud: |return| per rupee traded, negated so a high score is "
        "liquid. Illiquid names pay a premium that a retail book "
        "cannot actually collect, which is why the sign is worth seeing."
    ),
    Factor.EARNINGS_YIELD: (
        "Trailing quarterly EPS over price -- the classic value "
        "factor, E/P rather than P/E so a loss-making name scores "
        "negative instead of exploding. Uses the newest filing the "
        "market had actually seen at that session, never the quarter "
        "it describes."
    ),
    Factor.NET_MARGIN: (
        "Net profit over revenue from the newest observable filing. "
        "Quality rather than value: it asks how much of what a company "
        "sells it keeps, and does not need shares outstanding, which "
        "no free Indian source publishes cleanly."
    ),
    Factor.RESIDUAL_MOMENTUM: (
        "Momentum of the return that market beta does not explain "
        "(Blitz, Huij and Martens). Cleaner than raw momentum, which "
        "pays partly for having held high-beta names in a rising market."
    ),
    Factor.IDIOSYNCRATIC_VOL: (
        "Negated volatility of the residual, so a high score is a name "
        "that is quiet for its own reasons (Ang, Hodrick, Xing and "
        "Zhang). This is the low-volatility anomaly as documented; raw "
        "volatility instead rewards anything that barely moves."
    ),
    Factor.BETA: (
        "Negated trailing beta: betting against beta (Frazzini and "
        "Pedersen). Low-beta names have historically outperformed on a "
        "risk-adjusted basis, which the CAPM says should not happen."
    ),
    Factor.DOWNSIDE_BETA: (
        "Negated beta measured only on sessions when the market fell "
        "(Ang, Chen and Xing). A name can carry ordinary beta and far "
        "worse downside beta, and only the second one hurts."
    ),
    Factor.MOMENTUM_12_7: (
        "Return from 252 to 126 sessions ago — the intermediate horizon "
        "only (Novy-Marx). The claim is that momentum lives in the "
        "older half of the window, not the recent half."
    ),
    Factor.MAX_RETURN: (
        "Negated largest single-session return of the past month: "
        "lottery demand (Bali, Cakici and Whitelaw). Investors overpay "
        "for names that recently spiked, and those subsequently "
        "underperform."
    ),
    Factor.SEASONALITY: (
        "Average return in this calendar month across prior years "
        "(Heston and Sadka). Same-month returns persist far more than "
        "a random walk allows."
    ),
    Factor.OVERNIGHT_MOMENTUM: (
        "Mean close-to-open return over 21 sessions. Overnight moves "
        "price information arriving while the exchange is shut, set by "
        "whoever is willing to carry risk to the open."
    ),
    Factor.INTRADAY_MOMENTUM: (
        "Mean open-to-close return over 21 sessions. The other half of "
        "the day, largely liquidity provision to the flow the overnight "
        "gap creates."
    ),
    Factor.AVG_TRADE_SIZE: (
        "Traded value per transaction. Large tickets are a patient "
        "institution working an order; small ones are retail churn. The "
        "only factor here that reads the panel's `trades` column."
    ),
    Factor.TRADE_COUNT_SHOCK: (
        "Transactions against their own 21-session norm, negated. Many "
        "small orders is attention rather than repositioning, and "
        "attention-driven buying reverts."
    ),
    Factor.RANGE_VOL_RATIO: (
        "Parkinson high-low volatility over close-to-close volatility. "
        "High means the session churned intraday and closed where it "
        "started — uninformed two-sided flow."
    ),
    Factor.MOMENTUM_CONSISTENCY: (
        "Fraction of the last 252 sessions that closed up. The same "
        "annual return earned smoothly and earned in one jump are "
        "different signals; the smooth one draws less arbitrage."
    ),
    Factor.MOMENTUM_ACCELERATION: (
        "This month's return minus last month's. The second derivative "
        "of price turns before the level does, so it sees the "
        "inflection in flow first."
    ),
    Factor.MA200_DISTANCE: (
        "Distance from the 200-day moving average. A different anchor "
        "from the 52-week high and watched by different participants — "
        "institutional trend filter rather than retail salience."
    ),
    Factor.VOL_OF_VOL: (
        "Volatility of trailing volatility, negated. Uncertainty about "
        "how risky a position will be is distinct from the risk itself, "
        "and a name whose vol is unstable is hard to size."
    ),
    Factor.SEMI_DEVIATION_RATIO: (
        "Downside dispersion over upside dispersion. Investors pay more "
        "to avoid losses than to gain, so names with asymmetric "
        "downside should carry a discount."
    ),
    Factor.LOW_52W_PROXIMITY: (
        "Closeness to the 52-week low, negated. The disposition effect "
        "near lows is a different bias from anchoring near highs, so "
        "this is not assumed to be the mirror of high proximity."
    ),
    Factor.RESIDUAL_REVERSAL: (
        "Five-day reversal computed on residual rather than raw "
        "returns. Strips market-wide reversal, leaving the part that is "
        "payment for absorbing an imbalance in that specific name."
    ),
}
