# NEUTRON — Real Quant Trading System
## Master Plan v2 (solo operator, multi-market)

> v1 was an engineering roadmap. v2 is the actual plan: engineering + mathematics + research protocol + economics, fused. Grounded in the PDF blueprint and in your pipeline model (`Data → Math/Stats → Strategy → Backtest → Risk → Order → Broker → P&L → Feedback`), which is correct.

---

## PART 0 — What Changed From v1, And Why

Three corrections after thinking harder, plus one scope decision made later.

### 0.0 SCOPE: equities only (decided after v2)
**This system trades stock exchanges. No crypto, no FX, no commodities.**

Everything below that mentioned crypto has been revised accordingly. The
consequences are worth stating rather than quietly absorbing:

**What was lost.** Crypto was going to be the *research and paper* environment
— free data, free testnet, 24/7 markets meaning you could iterate at any hour.
That convenience is gone.

**What replaces it.**

| Need | Crypto was giving | Now |
|---|---|---|
| Free deep history | Binance klines | NSE bhavcopy archive (free, back to the 1990s, includes delisted names) |
| Free paper venue | Binance testnet | **Alpaca paper** — free account, free IEX data, real US equities |
| Fast iteration | 24/7 markets | NSE 6.25h/day + US 6.5h/day. Slower, and adequate — a 6-week paper run needs 6 weeks either way |
| Second market for cross-validation | crypto vs equity | **US equities** — genuinely better, since a strategy that works on both NSE and US is far more likely real |

**What got simpler, and it is not a small list.** One asset class means: one cost
model family, no funding rates, no perpetual mechanics, no 24/7 calendar edge
cases, no VDA tax regime, and no instrument that trades on a weekend. The
`ContinuousCalendar`, `FundingRate` event and the entire crypto cost module are
deleted rather than left dormant — §14.3 is explicit that unused flexibility is
a permanent maintenance tax.

**Cross-market validation is the real gain.** Momentum that works on NSE *and*
on US large caps is evidence of a risk premium. Momentum that works on NSE and
on Bitcoin is mostly evidence that both went up.

### 0.1 Why live capital goes to NSE equity, not elsewhere
The original reasoning here was about crypto's tax treatment, and it is retained
because it explains the decision and because the arithmetic generalises.

India's VDA regime (Section 115BBH) taxes gains at 30% flat with **no loss
set-off at all** — not against other income, not even against other crypto
gains — plus 1% TDS per transfer:

```
100 trades: 60 winners avg +₹1,000  = +₹60,000
            40 losers  avg   -₹800  = -₹32,000
            ─────────────────────────────────
            Net P&L                 = +₹28,000
            Taxable (gains only)    =  ₹60,000
            Tax @30%                =  ₹18,000
            ─────────────────────────────────
            Effective tax rate ≈ 64% of net P&L
```

A genuine 1.2 Sharpe strategy becomes uninvestable. **NSE equity** instead:
STCG 20%, LTCG 12.5%, losses carry forward eight years and can be set off.
That is the difference between a tax system that taxes profit and one that taxes
gross winnings. **Verify with a CA — rules change.**

US equities held by an Indian resident are a third regime again (LRS limits,
Schedule FA reporting, foreign tax credit). Alpaca paper trading involves no
real money and therefore no tax question at all, which is exactly why it is the
paper venue rather than a live one.

### 0.2 Mathematics is a system layer, not a footnote
v1 buried math inside "features." Wrong. Your dump correctly puts Math/Stats as a stage in the pipeline. It gets its own module (`quant/math/`), its own test suite, and its own build phase. You will implement estimators yourself and validate against `scipy`/`statsmodels` — that is how learning and building fuse into one activity instead of competing for hours.

### 0.3 The anti-overfitting protocol is the actual product
v1 had "validation" as one phase among nine. That understates it. Your dump nailed the MA=137 problem. The honest framing: **your enemy is not the market, it is yourself.** A retail quant with a decent backtester will generate a hundred beautiful-looking equity curves in a month, and 95+ of them are noise. The system's most valuable component is the one that kills your ideas efficiently. It gets Part 5 of this document, in full.

---

## PART 1 — The Frame

Everything in this system exists to serve one equation:

```
Net Edge = Gross Edge − Commission − Spread − Slippage − Market Impact − Funding − Taxes

Survival = Net Edge × Opportunities × Capital − P(Ruin)
```

Every module maps to exactly one of three jobs:

| Job | Modules | Failure mode |
|---|---|---|
| **Find gross edge** | data, math, features, strategy, backtest | Overfitting — finding edge that isn't there |
| **Preserve net edge** | cost model, execution, portfolio, reconciliation | Cost blindness — real edge destroyed in translation |
| **Prevent ruin** | risk engine, kill switch, monitoring, position sizing | One bad day erases two good years |

If a proposed feature doesn't serve one of these three, it doesn't get built. This is the complexity budget, and as a solo operator you are always over it.

**The order of importance is the reverse of the order of glamour:** ruin prevention > net edge preservation > gross edge discovery. Beginners spend 90% of their time on the third.

---

## PART 2 — Ground Truth (expectation calibration)

Read this section twice. Miscalibrated expectations are how people quit at month 8.

### 2.1 What a good result actually looks like
| Metric | Fantasy | Reality for good solo systematic strategy |
|---|---|---|
| Net Sharpe (daily freq, after all costs) | 3.0+ | **0.7 – 1.5** |
| Max drawdown | "5%" | **15 – 30%**, and you will live through it |
| Win rate (trend) | 70% | **35 – 45%** (positive expectancy from tail winners) |
| Win rate (mean reversion) | 70% | **55 – 65%** (with a fat left tail that will scare you) |
| Ideas tested → survive OOS | 1 in 2 | **~1 in 20** |
| Ideas surviving OOS → survive live 6mo | most | **~1 in 3** |
| Time to first live capital | 3 months | **9 – 12 months** at 20 hrs/week |

**Smell test**: any daily-frequency backtest showing Sharpe > 2.5 after realistic costs is a bug or a leak until proven otherwise. Go find the look-ahead. It's there. This rule will save you months.

### 2.2 What is actually accessible to you
Sorted by realistic edge availability for a solo operator with retail infrastructure:

| Strategy space | Accessible? | Why |
|---|---|---|
| Cross-sectional equity momentum/value (daily, NSE) | ✅ Yes | Capacity is huge, edge is documented, costs manageable at low turnover |
| Volatility targeting / risk-parity overlays | ✅ Yes | Improves almost any strategy; pure math, no edge discovery needed |
| US large-cap cross-sectional momentum (Alpaca paper) | ✅ Yes | Deepest, cleanest market; free paper venue; validates whether an NSE result is a risk premium or a local artefact |
| Pairs / stat-arb on liquid NSE names | ⚠️ Marginal | Edge is thin, decayed, cost-sensitive — good for *learning*, bad for capital |
| Options vol selling | ⚠️ Dangerous | Sells pennies in front of a bulldozer without serious risk machinery |
| Intraday / minute-bar | ❌ Not yet | Costs and STT dominate; needs microstructure work |
| Market making / HFT | ❌ Never (solo) | Colocation, latency, capital, team. Not a solo game. Your dump already says this — correct. |

**Where you should hunt**: low-turnover, cross-sectional, multi-instrument, risk-managed. Not intraday prediction.

### 2.3 The solo-operator constraint
You are simultaneously: Quant Researcher, Quant Developer, Data Engineer, Infra Engineer, Risk Manager, Trader, and Compliance. Seven roles from your own list, one person.

Three consequences, non-negotiable:
1. **Anything not automated will not happen.** Not "will happen late" — will not happen.
2. **Alerting is mandatory from paper-trading day one.** You will be asleep, at work, or on a train when it breaks.
3. **Every feature is maintained by you, alone, forever, at 2am when it breaks during a market event.** Budget accordingly.

---

## PART 3 — System Architecture (9 layers)

Modular monolith. One repo, strict internal boundaries enforced by CI. Not microservices — you don't have an ops team.

```
┌──────────────────────────────────────────────────────────────┐
│ L9  OPS      monitoring · alerting · runbooks · incident      │
├──────────────────────────────────────────────────────────────┤
│ L8  EXECUTION   order state machine · broker adapters ·       │
│                 idempotency · reconciliation                  │
├──────────────────────────────────────────────────────────────┤
│ L7  RISK        pre-trade · intraday · strategic · killswitch │  ← independent process
├──────────────────────────────────────────────────────────────┤
│ L6  PORTFOLIO   sizing · allocation · vol targeting ·         │
│                 correlation · turnover control                │
├──────────────────────────────────────────────────────────────┤
│ L5  VALIDATION  walk-forward · DSR · PBO · sensitivity ·      │
│                 placebo · regime · the gauntlet               │
├──────────────────────────────────────────────────────────────┤
│ L4  BACKTEST    event loop · fill models · cost models ·      │
│                 accounting · metrics · experiment registry    │
├──────────────────────────────────────────────────────────────┤
│ L3  STRATEGY    StrategySpec · signal generation · families   │
├──────────────────────────────────────────────────────────────┤
│ L2  MATH        stats · timeseries · linalg · optimization ·  │  ← NEW in v2, first-class
│                 estimators · metrics                          │
├──────────────────────────────────────────────────────────────┤
│ L1  DATA        feeds · normalization · quality · versioning ·│
│                 point-in-time universe · corporate actions    │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 Repo layout
```
neutron/
├── core/
│   ├── instruments/     # instrument master, symbol mapping over time
│   ├── calendars/       # NSE sessions, US sessions, holidays
│   ├── events/          # MarketDataEvent, OrderEvent, FillEvent
│   └── clock/           # event_time vs receive_time vs decision_time
├── data/
│   ├── feeds/           # nse.py · kite.py · alpaca.py
│   ├── store/           # Parquet writer · DuckDB reader · Postgres metadata
│   ├── quality/         # gaps · dupes · outliers · staleness · CA consistency
│   ├── corpactions/     # splits · bonus · dividends · adjustment engine
│   └── universe/        # point-in-time universe builder
├── quant/
│   ├── math/
│   │   ├── stats/       # moments · robust estimators · hypothesis tests
│   │   ├── timeseries/  # stationarity · cointegration · volatility (GARCH/EWMA)
│   │   ├── linalg/      # PCA · covariance shrinkage · conditioning
│   │   ├── optim/       # mean-variance · risk parity · HRP
│   │   └── metrics/     # Sharpe · Sortino · DSR · PBO · drawdown analytics
│   ├── features/        # versioned feature definitions
│   └── strategies/      # StrategySpec implementations by family
├── engine/
│   ├── backtest/        # event loop · fill models · accounting
│   ├── costs/           # per-asset-class cost + tax models
│   ├── validation/      # the gauntlet
│   └── experiments/     # registry · pre-registration · trial counter
├── trading/
│   ├── risk/            # INDEPENDENT — separate process
│   ├── paper/           # simulated fills on live data
│   ├── execution/       # order state machine · broker routing
│   └── reconcile/       # broker vs internal truth
├── ai/                  # Phase 7+ ONLY
├── ops/                 # monitor · alert · runbooks
├── apps/
│   ├── api/             # FastAPI gateway
│   ├── report/          # static HTML research reports (artifacts, not sessions)
│   └── web/             # React ops console (M8+, designed — see Part 12)
│       ├── tokens/      # design tokens = single source of truth for color/size
│       ├── components/  # Button, Table, Metric, VitalsBar, ...
│       └── screens/
└── tests/
```

### 3.2 Enforced boundaries (import-linter in CI, build fails on violation)
```
quant/*        ⇏ trading/*          # research never imports execution
ai/*           ⇏ trading/*          # AI never touches the trading plane
ai/*           ⇏ secrets            # AI never sees credentials
trading/risk   ⇏ quant/strategies   # risk is independent of the thing it polices
data/*         ⇏ trading/*          # data layer knows nothing downstream
```

That fourth rule is the one everyone breaks and the one that causes look-ahead bias.

### 3.3 The three clocks (source of most look-ahead bias)
```
event_time     — when it happened at the exchange
receive_time   — when your system got it
decision_time  — when your strategy is permitted to act on it
```
**Invariant, enforced in the backtester core:** a strategy at `decision_time = T` may only see data where `receive_time ≤ T`. Not `event_time ≤ T`. Fundamentals released "for Q2" are not usable on the last day of Q2 — they're usable on the filing date. Index constituents as of 2019 are not today's constituents. Enforce this in code, not in discipline.

---

## PART 4 — The Math Layer (build-to-learn)

Your Month 1–6 learning path from the dump gets fused into the build. Rule: **you implement the estimator yourself, then validate against the reference library, then use the reference library in production.** Implementing teaches; the library is battle-tested.

| Module | You implement | Validate against | Strategy it unlocks |
|---|---|---|---|
| `stats/moments` | rolling mean/var/skew/kurt, Welford's algorithm | `numpy`, `scipy.stats` | everything |
| `stats/robust` | median, MAD, Huber, winsorization | `statsmodels.robust` | outlier-resistant signals |
| `stats/tests` | t-test, bootstrap CI, permutation test | `scipy.stats` | significance of any result |
| `timeseries/stationarity` | ADF, KPSS, Hurst exponent | `statsmodels.tsa` | mean reversion validity |
| `timeseries/cointegration` | Engle-Granger, Johansen | `statsmodels` | pairs / stat-arb |
| `timeseries/ou` | Ornstein-Uhlenbeck fit, half-life | — | mean-reversion holding period |
| `timeseries/garch` | EWMA vol, GARCH(1,1) MLE | `arch` | volatility targeting |
| `timeseries/kalman` | Kalman filter, dynamic hedge ratio | `pykalman` | adaptive pairs |
| `timeseries/hmm` | 2–3 state Gaussian HMM | `hmmlearn` | regime detection |
| `linalg/pca` | PCA via SVD, eigendecomposition | `numpy.linalg` | factor extraction, risk model |
| `linalg/covariance` | sample cov, Ledoit-Wolf shrinkage | `sklearn.covariance` | portfolio optimization that doesn't explode |
| `optim/meanvar` | Markowitz, efficient frontier | `cvxpy` | portfolio construction |
| `optim/riskparity` | equal risk contribution solver | — | robust allocation |
| `optim/hrp` | Hierarchical Risk Parity | — | allocation without matrix inversion |
| `metrics/perf` | CAGR, Sharpe, Sortino, Calmar, DD, turnover | `empyrical` | evaluation |
| `metrics/dsr` | **Deflated Sharpe Ratio** | paper: Bailey & López de Prado | overfitting defense |
| `metrics/pbo` | **Prob. of Backtest Overfitting (CSCV)** | paper: Bailey et al. | overfitting defense |

**Why covariance shrinkage matters and nobody tells you**: with 50 instruments and 250 days of data, the sample covariance matrix is nearly singular. Mean-variance optimization on it produces insane leveraged positions in the smallest-eigenvalue directions. Ledoit-Wolf shrinkage or HRP fixes this. This one item is the difference between a portfolio optimizer that works and one that blows up.

**Reading spine** (not a course list — these four, deeply):
1. *Advances in Financial Machine Learning* — López de Prado (Ch. 7, 11–13 on CV, overfitting, backtest statistics — the important chapters)
2. *Quantitative Trading* / *Algorithmic Trading* — Ernie Chan (practical, honest, Kalman + pairs done right)
3. *Active Portfolio Management* — Grinold & Kahn (the Fundamental Law: `IR ≈ IC × √Breadth` — why breadth beats brilliance)
4. *Trading and Exchanges* — Larry Harris (market microstructure — where your costs actually come from)

The Fundamental Law is worth internalizing now: **information ratio ≈ skill × √(number of independent bets)**. A mediocre signal applied across 200 stocks beats a great signal on one stock. This directly implies: build cross-sectional, multi-instrument strategies, not single-name predictors.

---

## PART 5 — The Research Protocol (the actual product)

This is the part that separates a real quant system from a backtest toy. Your MA=137 example is exactly right; here is the machinery that prevents it.

### 5.1 Pre-registration (mandatory, enforced by code)
Before any backtest runs, a row is written to `hypotheses`:

```python
Hypothesis {
    id, created_at
    statement          # "Cross-sectional 12-1 momentum in NSE top-100 is positive after costs"
    economic_mechanism # WHY should this exist? Who is on the other side? What's the risk premium
                       # or behavioral bias being harvested? If you can't answer, don't test.
    prediction         # "Long-short decile portfolio, net Sharpe > 0.5, monthly rebalance"
    success_criteria   # numeric, pre-committed, BEFORE seeing results
    kill_criteria      # what result would make you abandon this
    data_period_dev    # 2015-2020
    data_period_val    # 2021-2022
    data_period_test   # 2023-2025  ← LOCKED
    status             # OPEN | CONFIRMED | REJECTED | ABANDONED
}
```

**The economic mechanism field is the highest-value line in the whole system.** "The z-score reverts" is not a mechanism. "Index funds mechanically buy at rebalance dates, creating temporary price pressure that reverts within 5 days" is a mechanism. Ideas without mechanisms are data mining with extra steps.

### 5.2 The trial counter (the thing nobody implements)
Every backtest run increments `hypothesis.n_trials`. This number feeds the Deflated Sharpe Ratio.

The reason: if you test 100 variants at p<0.05, ~5 pass by pure chance. The Sharpe ratio you observe is the *maximum* of 100 draws, not a single draw. DSR corrects for exactly this — it asks "given that I tried N times, and given the skew and kurtosis of these returns, what's the probability the true Sharpe is above zero?"

```
DSR accounts for:  N trials · return skewness · return kurtosis · sample length
Reject if:         DSR < 0.95
```

Without a trial counter, DSR is uncomputable and you are flying blind. This is why it goes in the schema on day one.

### 5.3 Data partition (enforced, not disciplined)
```
DEVELOPMENT   2015 – 2020   unlimited access, iterate freely
VALIDATION    2021 – 2022   walk-forward + robustness, limited touches (logged)
LOCKED TEST   2023 – now    ONE access per strategy, EVER
```

Implement `test_set_access` table: strategy_id → accessed_at. Data loader **raises an exception** on second access to the locked period for the same strategy_id. Not a warning. An exception. You will try to override it; that's what it's for.

If a strategy fails the locked test — it's dead. You may not tweak and re-run. You may form a *new* hypothesis (new id, new mechanism) but the test period is now partially burned and DSR should reflect the accumulated trials across the family.

### 5.4 The Gauntlet — all must pass
| # | Test | Pass criterion | Kills |
|---|---|---|---|
| 1 | Data integrity | zero quality flags in period | garbage-in |
| 2 | Look-ahead audit | shuffle-future test: corrupt all future data → results unchanged | leakage |
| 3 | Deflated Sharpe Ratio | DSR > 0.95 given N trials | multiple testing |
| 4 | PBO (CSCV) | < 0.5 | selection bias |
| 5 | Walk-forward efficiency | OOS return / IS return > 0.5 | fragility |
| 6 | Parameter surface | plateau not spike — neighbors within ±20% keep >60% of Sharpe | curve fitting |
| 7 | Cost sensitivity | still positive Sharpe at **3×** modeled costs | cost blindness |
| 8 | Universe dropout | drop random 20% instruments ×100 → 5th pct Sharpe > 0 | single-name dependence |
| 9 | Regime slice | positive in ≥2 of 3 regimes, no regime worse than −2× target DD | regime luck |
| 10 | Placebo | random-entry, same holding period & exposure, ×1000 → real strategy above 95th pct | "any trading works here" |
| 11 | Monte Carlo trade shuffle | 5th pct max DD within pre-committed limit | sequence luck |
| 12 | Locked OOS | success criteria met on untouched period | everything |

**Test #2 (shuffle-future) is the cheapest highest-value test in the system.** Replace all data after each decision point with noise; if backtest results barely change, your strategy is reading the future. It catches leaks that eyeballing never will. Run it automatically on every backtest.

**Test #6, the parameter surface**, deserves a picture:
```
GOOD (mesa)                    BAD (needle) — this is your MA=137
Sharpe                         Sharpe
 1.2 │   ▁▄██████▄▁             2.4 │        █
 0.8 │  ▄████████▄              1.0 │        █
 0.4 │ ▄██████████▄             0.2 │▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁
     └────────────────              └────────────────
      20  50  80 110 140             20  50  80 110 140
```
Always plot it. A needle is a fitted artifact of one noise realization.

### 5.5 The rejection log (a real asset)
Every killed strategy writes: hypothesis, mechanism, what killed it, at which gauntlet stage. After a year you'll have 100+ entries. This prevents re-testing dead ideas, reveals *patterns* in failure ("all my mean-reversion ideas die on cost sensitivity → my cost assumptions or my holding periods are wrong"), and is the closest thing a solo quant has to institutional memory.

**Healthy rejection rate is 90%+.** If most of your ideas pass the gauntlet, the gauntlet is broken, not your ideas.

---

## PART 6 — Strategy Families, In Build Order

Each teaches specific math and specific failure modes. Build in this order; don't skip.

| # | Family | Math it teaches | Failure mode it teaches | Live-worthy? |
|---|---|---|---|---|
| 1 | SMA/EMA crossover, single asset | rolling stats, signal lag | whipsaw, turnover drag, why costs matter | ❌ (baseline only) |
| 2 | Time-series momentum (12-1) | return aggregation, vol scaling | regime dependence | ⚠️ |
| 3 | **Cross-sectional momentum**, NSE top-100 | ranking, PIT universe, neutralization | survivorship bias — hard | ✅ **strongest first candidate** |
| 4 | Mean reversion, single asset z-score | stationarity, ADF, Hurst | non-stationarity, fat left tail | ⚠️ **gated**: `ZScoreReversion(require_stationarity=True)` by default |
| 5 | Pairs trading | cointegration, Kalman hedge ratio, OU half-life | spurious correlation, regime break | ⚠️ (learn, don't fund) **gated**: `PairsTrading(require_cointegration=True)` by default |
| 6 | **Volatility targeting overlay** | GARCH/EWMA, realized vol | — | ✅ **apply to everything** |
| 7 | Multi-factor cross-sectional | cross-sectional regression, PCA, shrinkage | factor crowding, correlated bets | ✅ |
| 8 | Cross-market momentum (NSE + US) | cross-sectional ranking across venues | currency and session misalignment | ✅ strong validation tool |
| 9 | Options / vol | Greeks, BS, IV surface | tail risk, margin spirals | ❌ not year 1 |

**Two things to internalize:**

**#6 is nearly free alpha.** Volatility targeting — scale position size inversely to recent realized vol, targeting constant portfolio vol — improves the risk-adjusted return of *most* strategies without requiring any new edge. It's pure math applied to sizing. Build it early, apply it universally.

**#3 is your best first real candidate.** Cross-sectional momentum on liquid NSE names: documented across decades and markets, monthly rebalance (low turnover → survives Indian costs), high breadth (Fundamental Law working for you), capacity far above your capital. It is not exciting. It is plausible. Take plausible.

---

## PART 7 — The Cost & Tax Engine

This module decides which strategies are viable. Build it *before* the strategies, not after. Most retail backtests are fiction because this layer is a single `commission=0.001` constant.

### 7.1 India equity (delivery / CNC)
```
Brokerage         ₹0 (Zerodha delivery) or flat ₹20/order
STT               0.1% on BUY + 0.1% on SELL     ← dominant cost
Exchange txn      ~0.00297% (NSE)
SEBI              ₹10 per crore
Stamp duty        0.015% on buy
GST               18% on (brokerage + txn + SEBI)
DP charges        ~₹13-16 per scrip per sell day  ← brutal for small positions
─────────────────────────────────────────────────
Round trip ≈ 0.22% + DP charges
```
**Implication that changes strategy design**: at ~0.22% round trip, a strategy rebalancing weekly needs ~11% annual gross edge just to break even. Monthly rebalance needs ~2.6%. **This alone rules out most higher-frequency ideas and pushes you toward monthly cross-sectional rebalancing.** The cost model isn't a detail, it's a strategy-space constraint.

DP charges are per-scrip-per-day flat — with ₹5L capital across 20 positions (₹25k each), a ₹15 DP charge is 0.06% on top. Small accounts pay proportionally more. Model it.

### 7.2 India F&O
```
Brokerage    ₹20/order flat
STT          Futures: 0.02% sell · Options: 0.1% on PREMIUM sell,
             and 0.125% on INTRINSIC value if exercised  ← the classic account-killer
Exchange txn Futures ~0.0019% · Options ~0.05% on premium
Stamp, GST, SEBI as above
```
The exercised-ITM-option STT trap has genuinely wiped out retail accounts. If you ever build options strategies, model expiry mechanics explicitly. Not year 1.

### 7.3 US equity (Alpaca / IBKR)
```
Commission       $0 at Alpaca and most US retail brokers
SEC fee          0.0000278 x sell proceeds   (sell side only, rate revised annually)
FINRA TAF        $0.000166 per share sold, capped at $8.30   (sell side only)
Spread           the dominant cost; ~1-2bp on large caps, far wider on small
Slippage         square-root impact, see 7.5
─────────────────────────────────────────────────────────────
Round trip on a liquid large cap ≈ 2-4bp
```

**This is roughly 50x cheaper than NSE delivery**, and that difference is not a
detail — it changes which strategies are viable. A weekly rebalance needs ~11%
gross annual to break even on NSE and well under 1% on US large caps. It is why
Alpaca is the right paper venue for testing higher-turnover ideas, and equally
why a strategy validated only on US costs must be re-checked against the Indian
cost model before it means anything for live capital here.

**Tax note (India resident holding foreign assets).** A separate regime again:
LRS remittance limits, Schedule FA reporting, foreign tax credit. Alpaca *paper*
trading involves no real money and therefore no tax question, which is precisely
why it is the paper venue rather than a live one. Consult a CA before any real
US position.

### 7.5 Slippage model (use this, not a constant)
Square-root market impact law — the empirically-supported form:
```python
slippage_bps = k * half_spread_bps + λ * σ_daily * sqrt(order_qty / ADV)

# k ≈ 1.0 for marketable orders (you cross the spread)
# λ ≈ 0.5 – 1.0, calibrate from your own live fills once you have them
# ADV = 20-day average daily volume
```
The `sqrt` matters: impact is concave in size, not linear. At your capital level the second term is small — but build it correctly now so it stays correct when it isn't.

### 7.6 Fill assumptions (where look-ahead hides)
| Timeframe | Signal at | Legitimate fill | **Never** |
|---|---|---|---|
| Daily | close of day T | open of T+1, or last-30min VWAP of T | close of T ← this is look-ahead |
| Intraday bar | close of bar T | open of bar T+1 | close of bar T |
| Live | signal time S | S + measured latency | S |

The "close of the same bar the signal used" fill is the single most common source of fake backtest profit. The backtester should make it *structurally impossible* — the fill simulator only receives bars strictly after the decision bar.

---

## PART 8 — Risk Architecture

Three independent layers. Risk runs as a **separate process** with its own DB connection. A strategy cannot import it, configure it, or disable it.

### Layer 1 — Pre-trade (per order, synchronous)
```
max order notional · max position per instrument (% NAV) · price band sanity
(reject if >5% from last traded — fat finger) · lot size / tick size validity
· margin availability · order rate limit (max N orders/min — runaway loop guard)
```

### Layer 2 — Portfolio (continuous)
```
gross exposure ≤ X% NAV · net exposure within [−Y, +Y]
· per-sector cap · per-correlation-cluster cap (positions correlated >0.7 count as one bet)
· daily loss limit → halt · liquidity: position ≤ 5% of ADV
```
The correlation-cluster limit is the one people miss: 10 positions in correlated PSU banks is 1 bet, not 10. It defeats the Fundamental Law silently.

### Layer 3 — Strategic (drawdown ladder, pre-committed, automated)
```
DD  −5%  →  scale all positions to 50%
DD  −8%  →  scale to 25%
DD −10%  →  flat all · halt · manual restart only after written review
```
Pre-commit these numbers *in code, in writing, before going live*. The purpose is to remove you from the decision at the exact moment you're least capable of making it. Deciding "should I cut?" during a −9% drawdown is the worst possible time to think.

### Kill switch
One command, one Telegram message. Effect: halt new orders → cancel all open → alert → preserve state → require manual restart. **Test it monthly.** An untested kill switch is not a kill switch.

### Position sizing (before any of the above)
Volatility targeting as the default sizing rule:
```python
target_vol = 0.15  # 15% annualized portfolio vol
position_weight = (target_vol / realized_vol_asset) * signal_strength / n_positions
weight = clip(weight, 0, max_position_pct)
```
Never fixed-rupee sizing. Never Kelly at full fraction — quarter-Kelly at most, and only with a well-estimated edge, which you won't have.

---

## PART 9 — Execution & Reconciliation

### Order state machine (from PDF §19, keep it verbatim)
```
CREATED → RISK_CHECKED → SUBMITTED → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED
                                   ↘ REJECTED
                                   ↘ CANCEL_REQUESTED → CANCELLED
                                   ↘ UNKNOWN → reconciliation required
```
`UNKNOWN` is the important state. Networks fail mid-submit. You must be able to answer "did that order reach the broker?" without guessing. Every order carries a client-generated idempotency key so a retry can never double-fill.

### Reconciliation (runs every day, non-negotiable)
```
broker positions  vs  internal positions   → must match exactly
broker fills      vs  internal fills       → must match exactly
broker cash       vs  internal cash        → must match to the paisa
```
Any mismatch → alert + halt new orders until explained. **An unexplained reconciliation break is a system-down event**, not a rounding issue. This is how you find out your order logic double-counted before it costs real money.

### Execution quality tracking (feeds back into the cost model)
Log for every fill: intended price, actual fill price, decision timestamp, submit timestamp, fill timestamp. Weekly, compute realized slippage and compare to modeled slippage. **Recalibrate `λ` from your own fills.** This closes the loop between your backtest's fiction and reality — and it's how you learn whether your backtests can be trusted at all.

---

## PART 10 — Roadmap (12 months, ~20 hrs/week)

Each phase has a gate. No gate skip. The PDF's 90-day plan is team-scale; this is solo-scale and honest.

### M1 — Foundation & Data Spine
- Monorepo + the **full enforcement stack from §14.8 on day one** — ruff, mypy strict, pytest, import-linter boundaries, pre-commit, gitleaks, and the 5 custom AST lints. Retrofitting these onto 20k lines is miserable; installing them into an empty repo costs one afternoon.
- Postgres schema v1: instruments, hypotheses, experiments, backtests, orders, fills, positions, audit_events, test_set_access
- Instrument master + symbol-history mapping + calendars (NSE, US)
- NSE bhavcopy loader → Parquet (free, deep, includes delisted names)
- **Gate**: CI green; 3+ years of BTC/ETH daily+hourly bars stored, quality report clean, calendar answers `is_open()` for any timestamp in both markets.

### M2 — Point-in-Time Data (the unglamorous month that decides everything)
- NSE loader: Kite historical / bhavcopy archives → normalized schema
- Corporate actions engine: splits, bonuses, dividends → adjusted series with an audit trail of every adjustment
- **Point-in-time universe builder**: "NSE top-100 by ADV as of date D" using only data available at D
- Data quality: gaps, duplicates, zero/negative prices, staleness, CA consistency, volume spikes
- Dataset versioning: content hash + manifest row
- **Gate**: universe query for 2019-01-01 returns genuinely different constituents than 2024-01-01, and includes companies that have since been delisted. If delisted names are missing, you have survivorship bias and the whole system is built on sand. This gate is the most important one in the plan.

### M3–M4 — Backtester (spend the time here, it's the scientific instrument)
- Event loop exactly as PDF §14 / your section 25
- Fill models: next-open, VWAP-slice, spread-crossing — decision/fill separation structurally enforced
- Cost engine: India equity and US equity models from Part 7
- Portfolio accounting: cash, positions, corporate actions, margin
- Metrics module
- Experiment registry: params + data version + git commit + seed + cost model + results, all recorded
- Backtest regression suite: golden runs with pinned numbers, CI fails on drift
- Baselines: buy-and-hold, SMA crossover
- **Gate**: (a) same experiment reruns to identical numbers; (b) buy-and-hold backtest matches hand calculation to the rupee including dividends and splits; (c) **shuffle-future test passes** on baselines; (d) 3× costs degrades performance sensibly.

### M5 — Math Library + The Gauntlet
- Build `quant/math/` per Part 4 — implement, validate against reference libs
- Build the 12-test gauntlet per Part 5.4
- Hypothesis pre-registration + trial counter + locked-test enforcement
- **Gate**: gauntlet correctly **rejects** a deliberately overfit strategy (50 parameters fit in-sample), correctly **passes** buy-and-hold, and DSR computed with N=200 trials on a random strategy returns < 0.95.

### M6–M7 — Real Research (first honest hypotheses)
- Cross-sectional momentum on PIT NSE top-100
- Volatility targeting overlay
- Mean reversion + pairs (learning)
- Every idea pre-registered with an economic mechanism, run through the full gauntlet
- **Gate**: at least 15 hypotheses tested, ≥90% rejected, rejection log populated, ≥1 strategy through the locked OOS test. **If nothing survives, that is a successful outcome** — the gauntlet worked. Go generate more hypotheses.

### M8 — Portfolio + Risk Engine
- Portfolio construction: vol targeting, correlation clustering, turnover control, HRP allocation
- Risk engine as independent process, three layers, drawdown ladder
- Kill switch + test harness
- **Gate**: risk engine blocks an oversized order in test; kill switch flattens in test; drawdown ladder triggers correctly on a simulated −8% path.

### M9–M10 — Paper Trading
- **NSE paper = one scheduled cycle per session** (`apps/cli/paper`), run after the
  bhavcopy ingest. Daily-frequency trading needs a scheduled job, not a daemon:
  a process that exits cannot leak state, wedge overnight, or need a supervisor —
  and it runs free on GitHub Actions cron (§0.4). BUILT: `trading/paper/session.py`
  (plan → risk → submit → fills-from-broker → reconcile), `trading/paper/state.py`
  (atomic JSON state + NDJSON equity log), halt persisted across runs, `--clear-halt`
  is a human act.
- Alpaca websocket → US paper fills (free, real venue) — later, intraday only
- Same code path as live, broker submission swapped for simulation
- Daily reconciliation report + Telegram alerting (fills, PnL, risk utilization, data staleness)
- **Paper-vs-backtest drift tracking** — the truth metric, read from `paper_equity.ndjson`
- Alerting is WIRED: `ops/routing.py` builds the channel set from config, and `apps/cli/paper` raises a CRITICAL page with a runbook link on any reconciliation break, a warning on blocked orders, and an INFO summary otherwise. Console is always a sink so a misconfigured token degrades to a printed alert rather than silence. Set `NEUTRON_TELEGRAM_BOT_TOKEN` and `NEUTRON_TELEGRAM_CHAT_ID` to make it wake you.
- **Gate**: 6+ weeks continuous paper on ≥1 strategy; drift explained and within tolerance; zero unexplained reconciliation breaks; you have been woken by an alert at least once and the runbook worked.

### M11 — Live Readiness
- Kite live adapter, smallest possible real order end-to-end
- Secrets management, credential separation (research creds ≠ trading creds)
- Runbooks written: data outage, broker disconnect, reconciliation break, risk breach, fat finger
- Independent review: re-read your own gauntlet results as a skeptic; try to break your own strategy
- **Gate**: full checklist from PDF §40 signed off by you, in writing, with dates.

### M12+ — Small Live
- **One** strategy, NSE equity, monthly rebalance
- Capital: tuition-sized. An amount whose total loss changes nothing in your life.
- Hard daily loss stop, drawdown ladder armed, daily reconciliation
- Weekly ritual: 30 min — drift, costs vs modeled, incidents
- **Gate for scaling**: 3+ months live, zero unexplained breaks, realized costs within 1.5× modeled, live Sharpe within 1 std-dev of backtest expectation. Only then increase capital, and only on a pre-committed schedule.

### M13+ — AI Layer (Phase 7)
Deliberately last. AI accelerating a broken research loop produces garbage faster.
- Research agent: question → hypothesis + **economic mechanism** → StrategySpec draft
- Deterministic tools only: data catalog, feature catalog, experiment launcher, results reader. **The model never produces a number** — it reads numbers that the engine computed.
- Codegen sandbox: Docker, no network, no secrets, read-only data mount
- Skeptic agent: reads experiment results, hunts overfitting signatures, writes the case *against* the strategy
- **Hard rule**: AI never imports `trading/`, never sees credentials, never approves anything. Human approval before paper, always.
- **Gate**: end-to-end AI-proposed → gauntlet-evaluated → human-reviewed cycle, with every number in the report traceable to an experiment row in Postgres.

---

## PART 11 — Operating Discipline

### Daily (10 min, mostly automated)
Reconciliation ✓ · data quality ✓ · PnL · risk utilization · open incidents — all pushed to Telegram. You *read*, you don't *check*.

### Weekly (1 hr)
Paper/live vs backtest drift · realized vs modeled costs · experiment queue triage · one gauntlet test on a new hypothesis.

### Monthly (2 hrs)
Strategy health: keep / downsize / kill · incident retro · **kill switch test** · rejection-log pattern review.

### Quarterly
Re-examine: are my cost models still accurate? Has any strategy's edge decayed (rolling 3-month Sharpe vs backtest expectation)? Is anything running that I no longer understand?

### Budget
See **Part 13.5** — full license-audited breakdown. Summary: all software ₹0, M1 costs ₹0 total, M2–M8 ~₹2,000/mo, M9+ ~₹3,500–5,000/mo. Trading capital lives in a separate account and is never confused with the ops budget.

---

## PART 12 — Interface Design System

The dashboard is an **instrument panel, not a website**. Every design decision follows from one question: *can the operator read the system's true state in under 3 seconds, at 2am, under stress?*

Reference points: Bloomberg Terminal, IB Trader Workstation, TradingView, internal desk dashboards. Anti-reference: every consumer SaaS dashboard with 48px rounded buttons, gradient cards, and animated counters.

### 12.1 Governing principles

1. **Data-ink ratio** (Tufte) — every pixel either carries information or is removed. No decorative cards, no gradients, no shadows beyond 1px elevation, no icons that duplicate a label.
2. **Color carries meaning or it is not there.** Neutral gray is the default state of everything. Color = semantic signal only.
3. **Numbers are the interface.** Tabular alignment is non-negotiable. Misaligned digits are a bug.
4. **Density over comfort.** This is desktop software operated by one expert, not a landing page.
5. **Motion is information.** Animation permitted only when it communicates state change. Never decorative.
6. **The vitals never scroll away.** Connection, staleness, P&L, drawdown, risk utilization, kill-switch state — always visible on every screen.
7. **Keyboard-first.** Mouse is optional. Real operators don't reach for it.

### 12.2 Color tokens

Restricted palette. Two themes. Dark is the working default (long sessions), light exists for daylight/printing.

**Neutrals (structure — ~85% of every screen)**
| Token | Dark | Light | Use |
|---|---|---|---|
| `bg-base` | `#0B0D10` | `#FFFFFF` | app background |
| `bg-surface` | `#13161A` | `#F7F8FA` | panels, table body |
| `bg-elevated` | `#1A1E24` | `#FFFFFF` | modals, dropdowns, hover rows |
| `bg-inset` | `#080A0C` | `#EDEFF2` | inputs, code, wells |
| `border` | `#252A31` | `#E2E5E9` | dividers, table lines |
| `border-strong` | `#363C45` | `#C8CDD4` | focused inputs, active panel |
| `text-primary` | `#E6E9ED` | `#14171A` | numbers, headings |
| `text-secondary` | `#98A1AC` | `#5A636D` | labels, units |
| `text-tertiary` | `#5E6772` | `#8A929B` | placeholders, disabled |

**Semantics (the only saturated colors permitted)**
| Token | Dark | Light | Meaning |
|---|---|---|---|
| `profit` | `#26A69A` | `#0B7A6E` | positive P&L, long position, pass |
| `loss` | `#EF5350` | `#C0392B` | negative P&L, short position, fail |
| `warn` | `#F0A93B` | `#B87413` | limit approaching, degraded feed |
| `critical` | `#FF4D4F` | `#D32029` | breach, halt, kill state |
| `accent` | `#4A9EFF` | `#0B6BCB` | focus ring, selection, active tab |
| `neutral-flat` | `#98A1AC` | `#5A636D` | flat/zero position, no signal |

**Why these greens/reds and not pure `#00FF00`/`#FF0000`**: saturated primaries vibrate against dark backgrounds, cause afterimages, and fatigue the eye within an hour. `#26A69A`/`#EF5350` are TradingView's tested pair — desaturated enough to stare at all day, distinct enough to read instantly.

**Colorblind mode (ship it, don't defer it)**: ~8% of men have red-green deficiency. Provide a toggle swapping `profit → #4A9EFF` (blue), `loss → #FF9F43` (orange). **Independently of that toggle**, every P&L number always carries an explicit `+`/`−` sign and every direction always carries a `▲`/`▼` glyph — so color is never the sole encoding of anything. That redundancy is the actual fix; the toggle is the convenience.

### 12.3 Typography

| Role | Font | Size | Weight | Notes |
|---|---|---|---|---|
| Numbers (all) | **JetBrains Mono** | 12–13px | 400/500 | `font-variant-numeric: tabular-nums` always |
| UI labels | **Inter** | 12px | 500 | `letter-spacing: 0.01em` |
| Body / descriptions | Inter | 13px | 400 | |
| Table headers | Inter | 11px | 600 | uppercase, `letter-spacing: 0.04em` |
| Section titles | Inter | 14px | 600 | |
| Page title | Inter | 16px | 600 | that's the maximum — no 32px hero text |

Both fonts are SIL Open Font License, free, self-hosted (no CDN — must work offline and without leaking your usage to a third party).

**Number formatting rules (enforced by one shared formatter, not per-component):**
```
P&L              always signed         +₹12,340.50  /  −₹8,220.00
Percent change   always signed, 2dp    +1.24%  /  −0.87%
Ratios           2dp, unsigned         1.42
Price            instrument tick size  ₹1,245.30  /  $184.22  /  67,412.5
Large INR        Indian grouping       ₹12,34,567  (toggle: ₹1,234,567)
Null / NA        em dash               —          (never blank, never 0)
Zero             explicit              0.00       (never blank)
Alignment        numbers right, text left, header matches its column
```

### 12.4 Sizing scale

Desktop-only. Design target 1440×900 minimum. **Mobile is read-only monitoring — never control.** No order entry, no kill switch, no approvals from a phone. That constraint prevents a category of fat-finger disaster.

**Spacing** — 4px base: `4 · 8 · 12 · 16 · 24 · 32 · 48`. Nothing else.

**Controls**
| Variant | Height | Pad-X | Font | Use |
|---|---|---|---|---|
| `xs` | 24px | 8px | 11px | inline row actions |
| `sm` | 28px | 12px | 12px | toolbars, filters, segmented controls |
| `md` (default) | 32px | 16px | 13px | forms, standard actions |
| `lg` | 36px | 20px | 14px | page primary action (Run Backtest) |
| `danger` | 40px | 24px | 14px | KILL SWITCH / FLATTEN — deliberately oversized + confirm dialog |

Inputs and selects match button heights exactly (28/32/36) so toolbars align on a single baseline.

**Table rows**: 24px dense · 28px default · 32px comfortable. User-togglable, persisted.

**Border radius**: `2px` inputs/buttons, `4px` panels/modals. Nothing rounder. Quant software is not friendly, it is precise.

**Elevation**: `1px solid border` for panels. Modals get `0 4px 16px rgba(0,0,0,0.4)` and nothing else. No layered shadow systems.

### 12.5 Motion policy

**Permitted:**
- State transitions (hover, focus, panel open/close): `120ms ease-out`. Nothing longer.
- **Value flash on update**: 150ms background tint — `profit`/`loss` at 12% opacity — then fade. This is genuinely useful; it's how you see a fill land without watching the number.
- Indeterminate loading: static spinner or skeleton. No shimmer sweep.

**Banned:**
- Page/route transitions, slide-ins, parallax, spring physics, bounce, scroll-triggered anything
- Animated number counting (`0 → 12,340` rolling) — actively harmful, you cannot read a number mid-animation
- Chart draw-in animations on load — the data is the point, show it immediately
- Any animation over 200ms, anywhere

`prefers-reduced-motion: reduce` disables even the value flash (replace with a 1px left border marker).

### 12.6 Screen inventory

Two surfaces, deliberately different investment levels:

**Analytics terminal (`apps.cli.terminal`, M1+)** — the dense screen. One name fully decomposed (returns at six horizons, vol, drawdown, VaR/CVaR, skew/kurtosis, stationarity verdict, Hurst, autocorrelation, vol regime) or a cross-section (correlation, clusters, effective bets, condition number, HRP/ERC weights). Read by scanning columns, not by clicking.

**Prices are back-adjusted here and only here.** The panel stores raw closes because the backtester applies corporate actions to *positions* (§9). Analytics is the opposite case: a 1:1 bonus read from raw closes is a -50% day, which on RELIANCE produced a -49% three-year return, skew -6.9 and kurtosis 175 — every one an artefact. `back_adjust` carries future information and must never reach a backtest; describing what a security did is exactly what it is for.

**Research surface (static HTML reports, M1–M7)** — zero design investment beyond reusing the tokens. Built as `apps/report` + `apps.cli.report`.
- backtest report · **parameter surface plots** (the mesa-vs-needle check from §5.4) · gauntlet report · data quality report

**Revised from Streamlit (2026-08-18).** The original plan said Streamlit here, for good reasons that no longer apply. Two things changed. First, the React console already exists, so the "don't build React until you know your screens" sequencing this section depended on is spent. Second, and more important: §5 requires every number to stay traceable to an experiment row a year later, and **a Streamlit session cannot be kept**. A file can be committed beside the numbers it explains, attached to a CI run, or opened on a machine with nothing installed. The charts are inline SVG generated in ~250 lines, which costs less than the 200MB of plotting dependencies and the server process it replaces.

**The verdict is rendered above every chart, deliberately.** This whole section is a hazard: a report that opens with a rising equity curve invites "that looks good" before the reader reaches the statistics, and that is precisely the bias §5.4 exists to kill. The charts explain a verdict already reached; they are never a decision surface, and the report says so in its own footer.

**Operations surface (React, M8+)** — full design system. This is what you stare at while capital is live.

| Screen | Core content |
|---|---|
| **Vitals bar** (persistent) | feed status · broker status · staleness (sec since last tick) · today P&L · DD vs ladder · risk utilization · KILL |
| **Overview** | equity curve · open positions summary · today's fills · active alerts |
| **Positions** | instrument · qty · avg · last · unrealized · % NAV · sector · correlation cluster |
| **Blotter** | orders + fills, order state machine column, filterable, virtualized |
| **Risk** | limit utilization bars (used/limit) · drawdown ladder state · exposure gross/net · cluster concentration |
| **Strategies** | version · lifecycle state (RESEARCH→VALIDATED→PAPER→LIVE) · live-vs-backtest drift · approve/pause/kill |
| **Reconciliation** | broker vs internal diff table — any non-empty diff is a red banner |
| **Data health** | coverage heatmap · quality flags · last update per feed |
| **Incidents** | open/closed, runbook links |

### 12.7 The vitals bar (the single most important component)

40px, fixed top, present on every screen, never scrolls:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ● KITE  ● ALPACA    stale 0.4s │  P&L +₹12,340 ▲1.24% │ DD −2.1% ▓░░░░ │ ⏻ │
└────────────────────────────────────────────────────────────────────────────┘
   ↑ green/amber/red dot        ↑ amber >2s      ↑ signed+glyph   ↑ ladder  ↑ kill
```
Rules: dot = `profit`/`warn`/`critical` only. Staleness turns `warn` above 2s, `critical` above 10s **and simultaneously fires a Telegram alert** — the UI is never the only alarm, because you will not be looking at it. The drawdown meter shows position on the −5/−8/−10 ladder from §8. Kill button is `danger` size, always reachable, always requires a typed confirmation.

### 12.8 Accessibility & keyboard

- Contrast: WCAG AA (4.5:1 text, 3:1 UI) — a practical requirement here, not a checkbox, given hours of screen time
- Focus rings always visible (`2px accent`, never `outline: none`)
- `/` search · `g` + key to navigate · `Esc` closes · arrows/`j`/`k` move table rows · `?` shows shortcuts
- Every destructive action: typed confirmation, never a bare button

### 12.9 App shell — fixed viewport, zero page scroll

**The single most important layout rule: the shell is exactly `100vh` and never scrolls. Panels scroll internally.** A trading console that scrolls as a page is a website pretending to be an instrument.

At 1440×900:
```
┌──────────────────────────────────────────────────────────────────────┐
│  VITALS BAR                                                     40px │  fixed
├────┬─────────────────────────────────────────────────────────────────┤
│    │                                                                 │
│ N  │                                                                 │
│ A  │                      WORKSPACE                                  │
│ V  │                      1392 × 836                                 │
│    │                  (panel grid lives here)                        │
│48px│                                                                 │
│    │                                                                 │
├────┴─────────────────────────────────────────────────────────────────┤
│  STATUS BAR   env · git sha · latency · clock                   24px │  fixed
└──────────────────────────────────────────────────────────────────────┘
   900 − 40 − 24 = 836px of usable height. Every pixel accounted for.
```

- **Nav rail**: 48px icon-only (default) / 200px expanded, toggle persisted. Icons 16px, 32px hit box, active state = 2px `accent` left border + `bg-elevated`.
- **Status bar**: environment badge (`PAPER` amber / `LIVE` red / `DEV` gray), git SHA of running build, round-trip latency, market clock. 24px, `text-tertiary`, 11px.
- Environment badge is not cosmetic — it is the thing that stops you firing a test order into production.

### 12.10 Panel system

**No gutters. Panels are flush, separated by exactly 1px.** Achieved with one CSS trick that avoids double borders and alignment drift entirely:

```css
.workspace {
  display: grid;
  gap: 1px;
  background: var(--border);   /* the gap IS the divider */
  height: 100%;
  min-height: 0;               /* mandatory — see 12.12 */
}
.panel { background: var(--bg-surface); min-height: 0; overflow: hidden; }
```

**Panel anatomy:**
```
┌──────────────────────────────────────────┐
│ POSITIONS                   [⚙] [⋯] 28px │  header
├──────────────────────────────────────────┤
│                                          │
│  body: flex:1 · min-height:0             │
│        overflow:auto                     │
│        padding 12px  (0 for tables)      │
│                                          │
├──────────────────────────────────────────┤
│ 14 positions · net +₹12,340         24px │  footer (optional, aggregates)
└──────────────────────────────────────────┘
```

| Element | Spec |
|---|---|
| Header height | 28px · title 11px/600 uppercase `text-secondary` · 12px pad-x |
| Header actions | 20px icon buttons, right-aligned, `text-tertiary` → `text-primary` on hover |
| Body padding | **12px** for forms/text · **0 for tables and charts** — they bleed to panel edges |
| Footer | 24px, aggregates only (counts, sums), `text-secondary` 11px |
| Resize handle | 1px visual, **6px hit area**, `accent` on hover |

**The 0-padding-for-tables rule matters**: a table inside 12px panel padding wastes 24px horizontally and produces a floating slab that doesn't look like an instrument. Tables own their cell padding (8px x / 0 y) and touch the panel edge. Same for charts.

Panel sizes persist to `localStorage` per screen. Library: `react-resizable-panels` (MIT).

### 12.11 Layout archetypes

Four layouts cover every screen. Don't invent a fifth.

```
A · MASTER–DETAIL              B · GRID DASHBOARD
  Strategies, Experiments        Overview
┌────────┬──────────────────┐  ┌─────────────┬─────────────┐
│  list  │      detail      │  │  equity     │  positions  │
│  320px │      1fr         │  │   2fr       │    1fr      │
│        │                  │  ├─────────────┼─────────────┤
│        │                  │  │  risk util  │  fills      │
└────────┴──────────────────┘  └─────────────┴─────────────┘

C · SINGLE SURFACE             D · CHART OVER TABLE
  Blotter, Positions             Strategy detail
┌──────────────────────────┐   ┌──────────────────────────┐
│  toolbar          32px   │   │  chart          60%      │
├──────────────────────────┤   ├──────────────────────────┤
│  virtualized table 1fr   │   │  trades table   40%      │
├──────────────────────────┤   └──────────────────────────┘
│  aggregates       24px   │
└──────────────────────────┘
```

**Anti-patterns, explicitly banned:**
- `max-width: 1200px; margin: 0 auto` — a content-site pattern that throws away 240px on a 1440 screen
- Fixed-height charts — always `ResizeObserver`-driven, fill the panel
- Fixed-px column widths — use `fr`/percentage so tables consume the full width
- Empty states with illustrations and 200px of padding — one line of `text-tertiary`, centered, done
- Nested cards (a card inside a card inside a panel) — one level of containment, ever

### 12.12 Overflow, breakpoints, z-index

**The `min-height: 0` rule.** Every flex/grid child in the shell chain needs `min-height: 0`. Without it a child refuses to shrink below its content size, the panel grows, and the shell scrolls — breaking §12.9. This is the number-one cause of broken panel layouts and it is invisible until content gets long. Put it in the base `.panel` class and forget it.

**Breakpoints (desktop only):**
| Width | Behavior |
|---|---|
| < 1280 | nav rail → icons only; side panels stack; minimum supported |
| 1280–1599 | default layout |
| 1600–1919 | master–detail lists widen to 380px |
| ≥ 1920 | **add columns, never widen text** — split a panel into two, show more instruments, more metrics |

**At 2560px you show more data, not bigger data.** Font sizes never scale with viewport. A 13px label is 13px on every monitor.

**Z-index scale — fixed, no ad-hoc values:**
```
  0   content
 10   sticky table headers
 20   shell chrome (vitals, nav, status)
100   popover · dropdown · tooltip
200   modal + backdrop
300   toast · alert banner
999   kill-switch confirmation — must be reachable above everything
```

---

## PART 13 — Full Tech Stack (all free / open source)

License-audited. Zero recurring software cost. Only paid line item in the entire system is the Kite API (₹2,000/mo, and only from M2).

### 13.1 Frontend
| Concern | Choice | License | Why this over alternatives |
|---|---|---|---|
| Build | **Vite** | MIT | 200ms cold start. No SSR needed for a local single-user dashboard — Next.js adds routing, server components, and build complexity you'd carry alone forever |
| Framework | **React 18 + TypeScript** | MIT / Apache-2.0 | TS is mandatory: a `number \| undefined` slipping into a P&L calculation is a real bug class |
| Styling | **Tailwind CSS**, `theme` **replaced** not extended | MIT | Replacing the theme deletes Tailwind's default palette, so `bg-blue-500` becomes a compile error. You *cannot* accidentally use an off-system color. This is the enforcement mechanism for §12.2 |
| Primitives | **Radix UI** | MIT | Unstyled + accessible + keyboard nav for free. Never hand-roll a dropdown or dialog |
| Components | **shadcn/ui** (copy-in, then resize) | MIT | You own the code. Default sizes are consumer-scale — retune to §12.4 once, centrally |
| Price charts | **lightweight-charts** (TradingView) | Apache-2.0 | 45kb, purpose-built for OHLC, handles 100k+ bars at 60fps. Nothing else is close |
| Series charts | **uPlot** | MIT | 40kb, millions of points. Equity curves, drawdown, rolling metrics, latency |
| Custom viz | **D3** (selective imports) | ISC | Correlation heatmap, exposure treemap. Import modules, never the bundle |
| Tables | **TanStack Table + Virtual** | MIT | Headless, you style it. Virtualization is mandatory — a 50k-row blotter without it freezes the tab |
| Server state | **TanStack Query** | MIT | Polling, caching, stale-while-revalidate |
| Client state | **Zustand** | MIT | ~1kb, no boilerplate. Redux is overkill for one user |
| Live data | native **WebSocket** | — | No socket.io. You control both ends |
| Icons | **Lucide** | ISC | Clean at 16px, tree-shakeable |
| Fonts | **Inter** + **JetBrains Mono** | OFL | Self-hosted, no CDN |

**Explicitly rejected:** Recharts/Chart.js (too slow at financial data density) · AG Grid (community tier is heavy, the useful features are paid) · Material UI / Ant Design (opinionated consumer sizing fights §12.4 at every step) · Framer Motion (a motion library is an attractive nuisance given §12.5).

### 13.2 Backend & data
| Concern | Choice | License | Cost |
|---|---|---|---|
| Language | Python 3.12+ | PSF | ₹0 |
| API | FastAPI + Pydantic v2 | MIT | ₹0 |
| Dataframes | Polars (Pandas at edges) | MIT | ₹0 |
| Analytical queries | DuckDB (queries Parquet directly) | MIT | ₹0 |
| Columnar store | Parquet + local disk | Apache-2.0 | ₹0 |
| Operational DB | PostgreSQL | PostgreSQL | ₹0 |
| Cache / pubsub | Redis (only from M9) | RSALv2 (free self-host) or Valkey (BSD) | ₹0 |
| Stats | NumPy · SciPy · statsmodels · arch · scikit-learn | BSD | ₹0 |
| Optimization | cvxpy | Apache-2.0 | ₹0 |
| Fast screening | vectorbt (community) | Apache-2.0 | ₹0 |
| Research UI | static HTML + inline SVG (no dependency) | — | ₹0 |

### 13.3 Infra & ops
| Concern | Choice | License | Cost |
|---|---|---|---|
| Containers | Docker + Compose | Apache-2.0 | ₹0 |
| CI | GitHub Actions | — | ₹0 (free tier ample) |
| Boundary enforcement | import-linter | BSD | ₹0 |
| Lint / format | ruff | MIT | ₹0 |
| Types | mypy | MIT | ₹0 |
| Tests | pytest + hypothesis (property-based) | MIT / MPL | ₹0 |
| System metrics | Prometheus + Grafana | Apache-2.0 / AGPL | ₹0 self-host |
| Tracing | OpenTelemetry | Apache-2.0 | ₹0 |
| Alerting | **Telegram Bot API** | — | ₹0 |
| Secrets | SOPS + age (local), env vars | MPL / MIT | ₹0 |
| Error tracking | Sentry self-hosted, or plain structured logs | BSL / — | ₹0 |

**Division of labor between Grafana and your React UI** — worth getting right, it saves weeks:
- **Grafana** = system metrics. CPU, memory, feed latency, event throughput, error rates, uptime. Do **not** rebuild this; Grafana is excellent at it and free.
- **React UI** = trading state. Positions, risk limits, blotter, reconciliation, approvals, kill switch. Grafana is genuinely bad at these — it's a metrics viewer, not an operations console.

### 13.4 Free data sources (M1–M7 needs zero rupees)
| Market | Source | Cost | Notes |
|---|---|---|---|
| **NSE historical** | Bhavcopy archives (daily EOD, back to the 1990s) | ₹0 | Free but messy — you build the normaliser. The **only free path to delisted names**, which is exactly what the M2 survivorship gate requires |
| NSE corp actions | NSE/BSE announcements | ₹0 | Manual-ish; automate the parser |
| **US equity + paper venue** | **Alpaca** | ₹0 | Free paper account, free IEX data, REST + websocket. This is the paper environment |
| BSE historical | BSE bhavcopy | ₹0 | Same shape as NSE |
| US fundamentals | SEC EDGAR full-text + XBRL | ₹0 | Point-in-time by filing date — correct by construction |
| Macro | FRED API | ₹0 | Rates, inflation, everything US macro |
| NSE realtime + clean history | Kite Connect | ₹2,000/mo | **Only from M2.** Buys clean adjusted history + a websocket |

**Consequence: M1 still costs ₹0.** NSE bhavcopy and Alpaca between them cover
everything M1–M7 needs — deep free history with delisted names, and a real paper
venue with live data. The Kite subscription starts only when you want clean
adjusted NSE history and a live feed, and the bhavcopy path defers even that.

**What the equities-only scope costs, honestly:** no 24/7 market means iteration
during development is bounded by NSE (03:45–10:00 UTC) and US (14:30–21:00 UTC)
hours. Between them that is roughly 12 hours of live market a day, which is
ample — and a six-week paper run takes six weeks regardless of how many hours a
day the market is open.

### 13.5 Revised budget
| Item | Monthly |
|---|---|
| All software (every row above) | **₹0** |
| VPS — only from M9, paper trading needs uptime | ₹1,500–3,000 |
| Kite Connect API — from M2 | ₹2,000 |
| Claude API — from M13 | ₹2,000–8,000 |
| **M1 total** | **₹0** |
| **M2–M8 total** | **~₹2,000** |
| **M9+ total** | **~₹3,500–5,000** |

Trading capital stays in a separate account and is never confused with the ops budget.

### 13.6 Delivery model — headless engine + thin console

**Not a website. Not a desktop app. A headless engine with a local console attached to it.**

The decisive fact: **the engine cannot run on your laptop.** Laptops sleep, drop wifi, get closed, reboot for updates. NSE opens at 09:15 IST and Alpaca at 09:30 ET whether you are at the machine or not. So the trading engine is a daemon on an always-on VPS, and the UI is a *client* to it — thin, stateless, disposable, replaceable without touching a single line of trading logic.

```
  VPS  (always on, ₹1,500–3,000/mo, starts M9)      Clients
┌────────────────────────────────────────┐       ┌──────────────────────┐
│  ENGINE (headless daemon)              │       │ Tauri desktop app    │  ← primary, M11+
│    market data · strategy · portfolio  │◄──────┤ browser (any machine)│  ← M8+, fallback
│    risk · execution · reconciliation   │  API  │ Telegram (phone)     │  ← read-only + kill
│                                        │  +WS  └──────────────────────┘
│  FastAPI — serves REST + WS + React    │
│  Postgres · Parquet · Redis            │
└────────────────────────────────────────┘
```

Everything the engine does, it does with zero clients connected. Closing the UI must never affect trading — if it can, the architecture is wrong.

**Client choice, staged. Each stage reuses the previous build unchanged:**

| Stage | Client | Why |
|---|---|---|
| M1–M7 | **HTML reports on disk**, all local, no VPS | Research surface. Artifacts, keepable, zero runtime |
| M8–M10 | **React served by FastAPI**, opened in browser | Build and validate screens against real paper data. No packaging problem to solve yet |
| M11+ | **Tauri wrapping the same React build** | Live trading. Earns its place at exactly this point |

**Why Tauri (MIT/Apache-2.0, free) once capital is live** — every item is a real limitation of a browser tab for this specific job:
- Browser chrome eats ~80px of vertical space; §12.9 budgets every pixel of an 836px workspace
- `Ctrl+W` closes your trading console mid-session, with no confirmation
- Browser shortcuts (`Ctrl+T`, `Ctrl+F`, `/`) collide with the keyboard-first scheme in §12.8
- **Global hotkey for the kill switch** — works even when the window isn't focused
- Native OS notifications, tray icon, confirm-on-quit
- ~5MB binary using the system webview (Electron ships 100MB+ of Chromium — no reason to choose it for a new project)

**Why this ordering costs nothing**: Tauri wraps the *identical* React build. You are not choosing between web and desktop — you are building web, and adding a native shell later, when live capital makes those six limitations matter. Exactly the "complexity must earn its place" rule.

**Explicitly rejected: a native desktop UI (Qt/PySide/C++).** It would discard the entire Part 12 design system and Part 13 frontend stack, and financial charting on Qt is far weaker than `lightweight-charts`. Months of work for a worse console.

**Mobile**: no app, ever. Telegram bot for alerts and kill switch, plus the browser for read-only status. Consistent with §12.4 — mobile monitors, mobile never controls.

### 13.7 Access & network security

**Never expose the console to the public internet.** It holds order entry, approvals, and a kill switch. An internet-reachable trading UI is a catastrophic, unrecoverable hole — one credential leak and someone else is trading your account.

| Layer | Choice | Cost |
|---|---|---|
| Network access | **Tailscale** — private WireGuard mesh, VPS + laptop + phone only | ₹0 (free personal tier) |
| Fallback | SSH tunnel: `ssh -L 8000:localhost:8000 vps` | ₹0 |
| Firewall | VPS accepts **only** SSH (key-only) + Tailscale. Port 8000 bound to `127.0.0.1`, never `0.0.0.0` | ₹0 |
| Auth | Session auth on the API even inside the tunnel — defence in depth | ₹0 |
| Broker credentials | On the VPS only, in SOPS-encrypted files. Never on the laptop, never in the repo, never readable by research or AI processes (§3.2) | ₹0 |

M1–M8 run entirely on your laptop — no VPS, no exposure, nothing to secure yet. Tailscale and the VPS arrive together at M9 when paper trading needs uptime.

### 13.8 Where UI work lands in the roadmap
| Phase | UI investment |
|---|---|
| M1–M7 | HTML reports only. **Do not build React yet** — you don't know your screens until you've operated the research loop |
| M8 | Design tokens + component library + vitals bar. One focused week |
| M9–M10 | Ops screens built against real paper-trading data — the requirements are now known, not guessed |
| M11+ | Refinement driven by actual operator pain, not speculation |

Building the polished UI before M8 is the most common way solo quant projects die: three months of beautiful dashboards rendering data from a backtester that has a look-ahead bug.

---

## PART 14 — Engineering Standards

Rules that aren't mechanically enforced are wishes. Every rule below names its enforcer. You are solo — the tooling *is* your code reviewer.

### 14.1 Correctness non-negotiables (quant-specific)

These six exist because violating them silently produces a wrong number, and a wrong number in this system moves money.

**1 · Determinism.** Every stochastic operation takes an explicit seed. No unseeded `random`/`np.random`. No reliance on `set` iteration order or dict insertion order in anything affecting results. Reproducibility is a roadmap gate (M3); this is how it's achieved.
```python
# BAN
weights = np.random.dirichlet(np.ones(n))
# REQUIRE
rng = np.random.default_rng(seed)  # seed recorded in experiment row
weights = rng.dirichlet(np.ones(n))
```

**2 · Float for statistics, Decimal for money.**
> **Heuristic: if a broker could disagree with you about the number, it is `Decimal`.**

`float64` — returns, features, signals, vol, correlations, optimizer inputs, research metrics.
`Decimal` (or integer paise/cents) — cash balance, realized P&L, order notional, fees, taxes, anything reconciled against a broker statement.
Accumulating float error across 100k trades produces a P&L that fails reconciliation for reasons you will spend a weekend not finding.

**3 · All timestamps timezone-aware UTC.** `datetime.now()` and naive datetimes are banned by custom lint. Convert to market-local at the display layer only.
```python
# BAN
ts = datetime.now()
# REQUIRE
ts = datetime.now(timezone.utc)
```

**4 · Look-ahead impossible by construction, not by discipline.** The data API physically cannot return the future.
```python
# BAN — nothing stops a slice bug
bars = store.load(symbol)  # returns everything
signal = compute(bars[:i])  # off-by-one → look-ahead

# REQUIRE — future is unreachable
view = store.view(symbol, as_of=decision_time)  # receive_time <= as_of, enforced
signal = compute(view)
```
Eliminating a bug class beats testing for it. The three clocks (§3.3) are distinct `NewType`s so `event_time` cannot be passed where `decision_time` is expected.

**5 · Fail loud. `return 0` on error is the most dangerous pattern in trading software.** A missing price defaulting to 0 becomes a −100% return, becomes an enormous signal, becomes an enormous position.
```python
# BAN
try:
    px = feed.price(sym)
except:
    px = 0.0  # or None, or last known, silently

# REQUIRE
px = feed.price(sym)  # raises MissingPriceError; caller decides explicitly
```
No bare `except:`. No `except Exception: pass`. No silent fallback defaults for market data. A crash is recoverable; a silently wrong number is not.

**6 · Pure functions for anything producing a number.** Signal generation, feature computation, metrics, cost models: no I/O, no globals, no mutation of inputs. Same inputs → same outputs, always. Testing and reproducibility both fall out for free.

### 14.2 Structure limits

| Limit | Value | Enforcer |
|---|---|---|
| Function length | ≤ 50 lines | ruff `PLR0915` |
| File length | ≤ 400 lines | custom lint |
| Function parameters | ≤ 5 (beyond that, pass a dataclass) | ruff `PLR0913` |
| Nesting depth | ≤ 3 (guard clauses / early return) | ruff `PLR0912` |
| Cyclomatic complexity | ≤ 10 | ruff `C901` |
| React component | ≤ 150 lines | eslint |

Over a limit means the unit is doing more than one thing. Split by responsibility, not by line count.

### 14.3 Banned patterns

| Banned | Why | Enforcer |
|---|---|---|
| Commented-out code | git remembers; dead code misleads | ruff `ERA001` |
| `utils.py` / `helpers.py` / `misc.py` | junk drawers grow forever and never shrink; name modules by domain | custom lint |
| `print()` | unstructured, unsearchable, invisible in prod | ruff `T201` |
| Bare `except:` / `except Exception: pass` | see §14.1.5 | ruff `E722`, `S110` |
| Mutable default args | classic Python foot-gun | ruff `B006` |
| Wildcard imports | breaks static analysis and boundaries | ruff `F403` |
| Magic numbers | `0.001` is unreadable; `STT_DELIVERY_RATE` is auditable | review + `PLR2004` |
| `**kwargs` passthrough chains | destroys type safety, hides the real signature | mypy strict |
| Inheritance for code reuse | compose; deep strategy hierarchies become unmaintainable | review |
| Premature abstraction | **rule of three** — don't extract before the third use | review |
| "Just in case" flexibility | YAGNI. Solo: every abstraction is a permanent tax | review |
| Unused args / imports / vars | ruff `ARG`, `F401`, `F841` | CI blocks |
| `TODO` without a linked issue | becomes permanent | ruff `TD003` |

### 14.4 Naming — units are part of the name

Every quantity carries its unit or frequency in its identifier. This kills the bps-vs-percent 100× error class, which is the quant equivalent of the Mars Climate Orbiter.

```python
# BAN
def cost(price, size, slippage): ...


vol = compute_vol(returns)
limit = 500000


# REQUIRE
def cost_inr(price_inr: Decimal, size_shares: int, slippage_bps: float) -> Decimal: ...


vol_annualized_pct = compute_vol(returns_daily)
max_notional_inr = Decimal("500000")
```

Conventions: `_inr` `_usd` `_bps` `_pct` `_ms` `_shares` `_daily` `_annualized` `_utc`.
Booleans read as assertions: `is_open`, `has_position`, `should_halt`.
Never abbreviate domain terms: `volatility` not `vol` in public APIs (locals are fine).

### 14.5 Testing

| Target | Requirement |
|---|---|
| Every math function | validated against reference impl (`scipy`/`statsmodels`/`arch`) to ~1e-10 |
| Every cost model | asserted against a **real broker contract note**, hand-computed |
| Every strategy | zero signal on constant-price input (sanity floor) |
| Portfolio math | property-based (`hypothesis`): weights sum to 1, no negative cash without margin, exposure ≤ limit |
| Backtests | golden/regression runs with pinned numbers; CI fails on any drift |
| Order state machine | every transition + every illegal transition rejected |
| Risk engine | every limit tested at boundary, boundary−1, boundary+1 |

**Coverage: don't chase 100% globally.** Require **100%** on `engine/costs`, `trading/risk`, `trading/execution` state machine, and the accounting itself — `engine/accounting.py` (the ledger) and `engine/backtest/sizing.py` (weights to share counts). The plan originally said `trading/portfolio` here; that package never held code, the accounting landed in `engine/`, and a gate pointed at an empty directory reports success over nothing. Corrected 2026-08-18. Those are the modules where a wrong number costs money. Everything else: whatever coverage the meaningful tests produce.

Naming: `test_<unit>_<condition>_<expected>` — e.g. `test_stt_delivery_charged_on_both_legs`.

### 14.6 Types & boundaries

- `mypy --strict` on `core/`, `quant/`, `engine/`, `trading/`. No `Any` in any signature.
- **Pydantic v2 models at every boundary**: HTTP API, DB rows, external feed payloads, config files. Parse once at the edge, pass typed objects inward. Never let a raw `dict` from a broker API travel deeper than its adapter.
- `NewType` for every identifier so they cannot be crossed:
```python
InstrumentId = NewType("InstrumentId", str)
OrderId = NewType("OrderId", str)
StrategyId = NewType("StrategyId", str)
# passing an OrderId where InstrumentId is expected is now a type error
```
- Frontend types are **generated** from the FastAPI OpenAPI schema (`openapi-typescript`, MIT). Backend and frontend cannot drift, because drift becomes a TypeScript compile error.

### 14.7 Frontend rules

- **No arbitrary Tailwind values.** `w-[137px]`, `text-[#3a3a3a]`, `p-[7px]` are banned by eslint. Tokens only — this is what makes Part 12 real rather than aspirational.
- No inline `style={{}}` except genuinely dynamic values (chart dimensions from `ResizeObserver`).
- **Every number renders through the shared formatter** (`formatPnL`, `formatPct`, `formatPrice`). No component formats numbers itself — that's how `−0.5%` and `(0.50)%` end up on the same screen.
- No `useEffect` for derived state. Compute during render; `useMemo` only when profiled.
- Data fetching only via TanStack Query hooks. No `fetch()` inside components.
- Every list over 100 rows is virtualized. Not negotiable — the blotter will exceed it.
- Components are presentational; business logic lives in hooks or the backend. A component that computes P&L is a bug.

### 14.8 Enforcement stack (all free)

```
pre-commit  →  ruff format · ruff check · mypy · eslint · secret scan (gitleaks)
CI (GitHub Actions, free tier):
    ruff check --no-fix        (lint, complexity, dead code)
    mypy --strict              (types)
    import-linter              (architecture boundaries, §3.2)
    pytest --cov               (per-module thresholds)
    pytest tests/regression    (golden backtest numbers)
    custom lints               (project-specific, below)
    tsc --noEmit + eslint      (frontend)
```

**Custom AST lints worth the ~100 lines each** — nothing off-the-shelf catches these, and each guards a §14.1 non-negotiable:
1. `datetime.now()` without `tz=` → error
2. `float` in accounting modules (`trading/portfolio`, `engine/costs`) → error
3. Unseeded `np.random` / `random` outside tests → error
4. `store.load()` (unbounded) outside the data layer → error, must use `store.view(as_of=)`
5. Files named `utils|helpers|misc|common` → error

### 14.9 Definition of Done

A module is not done until **all** of these hold:
- [ ] `ruff` clean, `mypy --strict` clean
- [ ] `import-linter` boundaries pass
- [ ] Tests written and passing; money-touching modules at 100% coverage
- [ ] Public functions have docstrings stating **units and edge cases** (not restating the code)
- [ ] Zero `TODO` without a linked issue
- [ ] Boundary code has a Pydantic model
- [ ] If it computes money: hand-verified against a real example
- [ ] If it's a strategy: file header docstring carries **hypothesis, economic mechanism, reference, expected behavior** (§5.1)

### 14.10 Git

- Conventional Commits: `feat(backtest): add VWAP fill model`
- One logical change per commit. A commit that touches the cost model *and* the UI is two commits.
- **Never committed**: secrets, `.env`, data files, Parquet, notebooks with output (strip via `nbstripout`), broker credentials. `gitleaks` in pre-commit.
- Every backtest result records the git SHA (§M3) — a result you cannot reproduce from a commit is not a result.
- Tag every strategy that reaches `PAPER_APPROVED` / `LIVE_APPROVED`. You must be able to check out the exact code that traded a given day.

### 14.11 Comments & docs

- Comments explain **why**, never **what**. If the *what* isn't obvious, rename things.
- Every non-obvious financial constant cites its source:
```python
STT_DELIVERY_RATE = Decimal("0.001")  # 0.1% both legs, Finance Act 2004 as amended;
# verified against Zerodha contract note 2025-11-14
```
- Runbooks live in `ops/runbooks/*.md`, in the repo, versioned with the code they describe.
- No README that describes intent — describe how to run it. Intent lives in this document.

---

## PART 15 — Kill Criteria (decided now, while calm)

Pre-commit these. Written in advance, they're wisdom. Decided in the moment, they're panic.

**Kill a strategy when:**
- Live drawdown exceeds 1.5× the worst backtest drawdown
- Rolling 6-month live Sharpe is more than 2 std-dev below backtest expectation
- The economic mechanism you wrote down demonstrably stops being true (e.g. the structural inefficiency gets arbitraged, a rule change removes it)
- You can no longer explain, in one sentence, why it makes money

**Pause the whole system when:**
- Any unexplained reconciliation break
- Data quality failure on the instruments you trade
- You haven't looked at it in 2 weeks (a system you're not watching is a system you're not operating)

**Reconsider the whole project when:**
- 18 months in with zero strategies through the locked OOS test → your data or gauntlet has a bug, or your hypothesis generation lacks economic grounding. Diagnose which; don't just grind.

---

## The One-Paragraph Version

Build the instrument before the experiment. Months 1–5 produce no trading and no profit — they produce a point-in-time-correct data layer, a cost-honest backtester, a validated math library, and a gauntlet that kills ideas efficiently. Months 6–7 use it to test real hypotheses, and roughly nine in ten will die; that is the system working. Months 8–11 add independent risk, paper trading, reconciliation, and only then the operations console — dense, monochrome, keyboard-driven, with color reserved for P&L and alarms. Month 12 puts a small amount of real money behind one boring, plausible, low-turnover cross-sectional strategy on NSE equity, with a pre-committed drawdown ladder and a tested kill switch. AI comes after all of that, as a hypothesis generator that never produces a number and never touches an order. The edge is not in any single strategy — it is in owning a loop that can evaluate a hundred strategies honestly while your competition fools itself.

---

*Backtested performance is evidence, not proof. Every shortcut around research → validation → paper → controlled live is how accounts die.*
