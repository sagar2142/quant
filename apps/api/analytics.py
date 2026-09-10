"""Analytics endpoints — MASTER_PLAN §12.6.

Serves what `apps.cli.terminal` prints, as JSON, so the console and the
terminal read the same numbers from the same code. A second implementation of
"what is this security" would eventually disagree with the first, and the one
you were looking at would be the wrong one.

**Prices are back-adjusted here**, exactly as in the terminal and for the same
reason: the panel stores raw closes because the backtester applies corporate
actions to *positions*, and a 1:1 bonus read from raw closes is a -50% day
(§9). The panel itself is never mutated.

**The panel is loaded once and cached.** It is 3.3M rows and immutable between
ingests; re-reading it per request would make every screen slow enough to stop
being a terminal.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import polars as pl
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from apps.api.auth import ReadAccess
from apps.api.schemas import (
    CrossSectionResponse,
    HorizonReturn,
    NameRow,
    ScreenResponse,
    ScreenRowResponse,
    SecurityResponse,
)
from apps.cli.terminal import aligned_returns, load_actions, series_for
from core.clock import as_decision_time, utc_now
from core.config import settings
from data.feeds.nse_indices import BENCHMARK
from data.store.bars import NoDataError
from data.store.indices import IndexStore
from data.store.panel import PanelStore
from data.universe.pit import UniverseBuilder, UniverseSpec
from quant.analytics.crosssection import analyse_cross_section
from quant.analytics.screener import ScreenCriteria, SortKey, screen_universe
from quant.analytics.security import profile_security

__all__ = [
    "benchmark",
    "benchmark_names",
    "build_analytics_router",
    "latest_quote",
    "sector_breakdown",
]

#: Cap on symbols per cross-section request. The correlation work is O(n^2) and
#: a browser cannot read a 200-name matrix anyway.
MAX_SYMBOLS = 40

#: Rows returned by the symbol search. Enough to pick from, few enough to scan.
SEARCH_LIMIT = 60

#: Candidates named in a 404. Enough to spot the one you meant, few enough to
#: read in a single line.
SUGGESTIONS = 8

#: A cross-section is a comparison; one name is a security screen.
MIN_CROSS_SECTION = 2

#: Symbols accepted by the pairs screen.
#:
#: Lower than `MAX_SYMBOLS` because every unordered pair is tested and the count
#: is quadratic: twenty names is 190 regressions, forty would be 780. It is also
#: a research limit rather than a performance one — searching a larger set makes
#: a spurious cointegration near-certain, and the trial count is the thing that
#: has to be declared.
MAX_PAIR_SYMBOLS = 20

#: Sessions a name needs before it can be tested for cointegration.
#:
#: An ADF on a short series has almost no power, so a shorter name is excluded
#: rather than tested and reported as "not cointegrated" — that verdict would
#: be a statement about the sample rather than the pair.
MIN_PAIR_OBSERVATIONS = 120

#: Trailing window used to rank the symbol search by liquidity.
LIQUIDITY_WINDOW = 60


#: Exchanges this console can serve.
#:
#: **Kept apart rather than merged into one tape.** A name listed on both
#: exchanges has an NSE row and a BSE row with the same ISIN and different
#: prices, and which venue you trade is a decision, not a detail. Merging them
#: here would make that decision silently, for every screen at once (§13.4).
VENUES = ("NSE", "BSE")
DEFAULT_VENUE = "NSE"


def _venue(raw: str | None) -> str:
    """Validate a venue, or say which ones exist."""
    if not raw:
        return DEFAULT_VENUE
    name = raw.upper()
    if name not in VENUES:
        raise HTTPException(status_code=422, detail=f"unknown venue {raw!r}; expected {VENUES}")
    return name


#: One cached panel per venue. Two, not one: the BSE panel is 2.7M rows beside
#: NSE's 3.4M, and evicting one to read the other would make every switch
#: between them a full re-read.
@lru_cache(maxsize=len(VENUES))
def _read_panel(fingerprint: tuple[int, str], venue: str) -> pl.DataFrame:
    """Read one venue's whole panel. Keyed on the lake's own state.

    The fingerprint is not used for anything except cache identity: a different
    fingerprint is a different lake, so `lru_cache` evicts and re-reads.
    """
    del fingerprint
    return PanelStore(settings.lake, venue=venue).view(as_of=as_decision_time(utc_now()))


def _lake_fingerprint(venue: str = DEFAULT_VENUE) -> tuple[int, str]:
    """How many sessions this venue holds and the newest one, as a cache key.

    `sessions()` is a filename glob and touches no file contents, so this costs
    a directory walk rather than a parquet read. Cheap enough to check on every
    request, which is what lets an ingest be picked up without a restart.
    """
    try:
        sessions = PanelStore(settings.lake, venue=venue).sessions()
    except OSError:
        return (0, "")
    return (len(sessions), sessions[-1].isoformat() if sessions else "")


@lru_cache(maxsize=2)
def _read_benchmark(fingerprint: tuple[int, str], name: str) -> pl.DataFrame:
    """One index's whole history. Keyed on the index lake's own state."""
    del fingerprint  # cache identity only
    try:
        return IndexStore(settings.lake).series(name, as_of=as_decision_time(utc_now()))
    except NoDataError:
        return pl.DataFrame()


def _index_fingerprint() -> tuple[int, str]:
    """Sessions of index data held and the newest, as a cache key."""
    try:
        sessions = IndexStore(settings.lake).sessions()
    except OSError:
        return (0, "")
    return (len(sessions), sessions[-1].isoformat() if sessions else "")


def benchmark_names() -> list[str]:
    """Every index the lake holds, from the newest observable session.

    The feed stores all 165 indices NSE publishes because the file contains
    them all; without this only the default benchmark was reachable and the
    rest were ingested decoration.
    """
    try:
        return IndexStore(settings.lake).names(as_of=as_decision_time(utc_now()))
    except (NoDataError, OSError):
        return []


def benchmark(name: str = BENCHMARK) -> pl.DataFrame:
    """The benchmark index series, empty if it has not been ingested.

    Empty rather than raising: the console must still render a risk model with
    the equal-weight market proxy on a lake that has no index in it yet. What
    it must not do is claim the proxy is the index, which is why the model
    carries `market_source`.
    """
    return _read_benchmark(_index_fingerprint(), name)


class SectorRow(BaseModel):
    industry: str
    names: int
    #: Share of the universe in this industry, by name count.
    share: float


class SectorResponse(BaseModel):
    """What the tradable universe is actually made of."""

    observed_at: str | None
    #: Share of the requested universe NSE classifies. A name delisted before
    #: the classification was observed appears in no constituent list, so
    #: coverage over history is missing exactly the names that failed — the
    #: direction that flatters, hence reported rather than assumed.
    coverage: float
    universe: int
    rows: list[SectorRow]
    note: str = ""


def sector_breakdown(top: int = 100, venue: str = DEFAULT_VENUE) -> SectorResponse:
    """The universe's industry composition.

    Nothing in this system knew what a company does until the classification
    was ingested; a book eleven-thirtieths in one industry looked, from every
    screen, like thirty independent positions.
    """
    from collections import Counter  # noqa: PLC0415

    from data.store.sectors import SectorStore  # noqa: PLC0415

    # Present-tense by design: this screen describes the universe as it stands
    # now, so the newest classification is the right one. A dated read here
    # would be answering a question nobody asked.
    view = SectorStore(settings.lake).view()  # lint: allow-unbounded-read
    if not view.industries:
        return SectorResponse(
            observed_at=None,
            coverage=0.0,
            universe=0,
            rows=[],
            note="No classification ingested. Run: python -m apps.cli.ingest_sectors",
        )

    store = PanelStore(settings.lake, venue=venue)
    universe = UniverseBuilder(store).build(
        as_decision_time(utc_now()),
        UniverseSpec(top_n=top, min_sessions=20, lookback_days=60),
    )
    members = [str(m) for m in universe.members]
    if not members:
        return SectorResponse(
            observed_at=view.observed_at.isoformat() if view.observed_at else None,
            coverage=0.0,
            universe=0,
            rows=[],
            note="The universe is empty — ingest more sessions.",
        )

    counts = Counter(view.industry_of(m) or "unclassified" for m in members)
    return SectorResponse(
        observed_at=view.observed_at.isoformat() if view.observed_at else None,
        coverage=view.coverage(members),
        universe=len(members),
        rows=[
            SectorRow(industry=name, names=n, share=n / len(members))
            for name, n in counts.most_common()
        ],
    )


def _panel(venue: str = DEFAULT_VENUE) -> pl.DataFrame:
    """The whole panel, re-read only when the lake has actually changed.

    Cached because it is large enough that a per-request read would be felt on
    every screen, and keyed on the lake's contents rather than on nothing. The
    previous version cached unconditionally and documented a restart as the way
    to pick up new sessions. That was honest — the staleness the console showed
    was the true age of the panel being served, red light and all — but it made
    a daily-ingest system need a restart after every daily ingest to see the
    day it had just fetched.
    """
    return _read_panel(_lake_fingerprint(venue), venue)


def _not_found(history: pl.DataFrame, name: str) -> str:
    """A 404 that names the near misses.

    "TATA is not in the panel" is true and useless when TATASTEEL, TATAPOWER
    and eight others are. An error that leaves the reader guessing the ticker
    is a dead end; one that lists candidates is a next step.
    """
    matches = (
        history.filter(pl.col("symbol").str.starts_with(name))["symbol"].unique().sort().to_list()
    )
    if not matches:
        # Fall back to a substring search: the typo may be a prefix, not a stem.
        matches = (
            history.filter(pl.col("symbol").str.contains(name, literal=True))["symbol"]
            .unique()
            .sort()
            .to_list()
        )
    if matches:
        listed = ", ".join(matches[:SUGGESTIONS])
        more = f" (+{len(matches) - SUGGESTIONS} more)" if len(matches) > SUGGESTIONS else ""
        return f"{name} is not a ticker. Did you mean: {listed}{more}"
    return f"{name} is not in the panel"


def _windowed(history: pl.DataFrame, sessions: int) -> pl.DataFrame:
    if sessions <= 0:
        return history
    recent = history["event_time"].unique().sort().tail(sessions)
    return history.filter(pl.col("event_time").is_in(recent.implode()))


def latest_quote(symbol: str, venue: str = DEFAULT_VENUE) -> dict[str, object] | None:
    """Identity, last close and ADV for one ticker, or None if unknown.

    **The instrument id is the point.** A ticket types "RELIANCE", but an order
    and a position are keyed on ISIN (§1.1) — a symbol can wear more than one
    ISIN over its life, and 344 names in this panel do. Sending a ticker to a
    broker as though it were an identity is how an order ends up against the
    wrong security after a face-value change.

    The last close and ADV are returned with it because the risk engine needs
    both: without a last price the fat-finger band cannot be evaluated, and it
    blocks rather than passes when it cannot — correctly, since an order whose
    sanity cannot be established has not been established as sane.
    """
    name = symbol.upper()
    recent = _windowed(_panel(venue), LIQUIDITY_WINDOW).filter(pl.col("symbol") == name)
    if recent.is_empty():
        return None
    last = recent.sort("event_time").tail(1)
    turnover = recent.select((pl.col("close") * pl.col("volume")).median().alias("adv")).item()
    return {
        "symbol": name,
        "venue": venue,
        "instrument_id": str(last["instrument_id"][0]),
        "last_close": float(last["close"][0]),
        "adv": float(turnover or 0.0),
        "as_of": last["event_time"][0].date().isoformat(),
    }


def _register_venues(router: APIRouter) -> None:
    @router.get("/venues", dependencies=[ReadAccess])
    def venues() -> list[dict[str, object]]:
        """Which exchanges have data, and how much.

        Reported rather than assumed, because they do not cover the same
        period: NSE runs from 2019, while BSE's UDiFF archive only begins in
        2024 — ask for an earlier BSE session and the exchange returns its
        homepage with a 200. A console that offered both as though they were
        interchangeable would show an empty chart and no reason for it.
        """
        rows: list[dict[str, object]] = []
        for name in VENUES:
            try:
                sessions = PanelStore(settings.lake, venue=name).sessions()
            except OSError:
                sessions = []
            rows.append(
                {
                    "venue": name,
                    "sessions": len(sessions),
                    "first": sessions[0].isoformat() if sessions else None,
                    "last": sessions[-1].isoformat() if sessions else None,
                }
            )
        return rows


def _register_search(router: APIRouter) -> None:
    @router.get("/symbols", dependencies=[ReadAccess])
    def symbols(
        q: str = Query("", max_length=32), venue: str = Query(DEFAULT_VENUE)
    ) -> list[dict[str, object]]:
        """Symbol search, ranked by liquidity.

        Ranked rather than alphabetical: the name you want is almost always one
        you can actually trade, and an alphabetical list buries it under
        illiquid tickers sharing a prefix.
        """
        try:
            history = _panel(_venue(venue))
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        recent = _windowed(history, LIQUIDITY_WINDOW)
        ranked = (
            recent.with_columns((pl.col("close") * pl.col("volume")).alias("traded"))
            .group_by("symbol")
            .agg(pl.col("traded").median().alias("adv"), pl.len().alias("bars"))
            .sort("adv", descending=True)
        )
        if q:
            ranked = ranked.filter(pl.col("symbol").str.starts_with(q.upper()))
        return [
            {"symbol": r["symbol"], "adv": float(r["adv"] or 0.0), "bars": int(r["bars"])}
            for r in ranked.head(SEARCH_LIMIT).to_dicts()
        ]


def _register_security(router: APIRouter) -> None:
    @router.get("/security/{symbol}", response_model=SecurityResponse, dependencies=[ReadAccess])
    def security(
        symbol: str, sessions: int = Query(0, ge=0), venue: str = Query(DEFAULT_VENUE)
    ) -> SecurityResponse:
        try:
            history = _windowed(_panel(_venue(venue)), sessions)
        except NoDataError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        name = symbol.upper()
        actions = load_actions(history, [name])
        rows = series_for(history, name, actions)
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=_not_found(history, name))

        try:
            p = profile_security(name, rows["close"].to_list(), rows["volume"].to_list())
        except ValueError as exc:
            # Too little history is a 422, not a 500: the request was
            # well-formed and the answer is "not enough data", which the
            # console should show rather than swallow.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SecurityResponse(
            symbol=p.symbol,
            observations=p.observations,
            last_close=p.last_close,
            horizons=[HorizonReturn(label=k, value=v) for k, v in p.horizon_returns.items()],
            cagr=p.cagr,
            high_52w=p.high_52w,
            low_52w=p.low_52w,
            off_high=p.off_high,
            annual_volatility=p.annual_volatility,
            max_drawdown=p.max_drawdown,
            current_drawdown=p.current_drawdown,
            adv_value=p.adv_value,
            sharpe=p.sharpe,
            sortino=p.sortino,
            calmar=p.calmar,
            hit_rate=p.hit_rate,
            skewness=p.skewness,
            kurtosis=p.kurtosis,
            var_5=p.var_5,
            cvar_5=p.cvar_5,
            tail_ratio=p.tail_ratio,
            verdict=p.stationarity.verdict.value,
            adf_pvalue=p.stationarity.adf_pvalue,
            kpss_pvalue=p.stationarity.kpss_pvalue,
            hurst=p.stationarity.hurst,
            tradable_as_mean_reversion=p.stationarity.tradable_as_mean_reversion,
            autocorrelation={str(k): v for k, v in p.autocorrelation.items()},
            realised_vol=p.realised_vol.annualised,
            ewma_vol=p.ewma_vol.annualised,
            vol_regime=p.vol_regime,
            is_implausible=p.is_implausible,
            fat_left_tail=p.fat_left_tail,
        )

    @router.get("/security/{symbol}/series", dependencies=[ReadAccess])
    def series(
        symbol: str, sessions: int = Query(0, ge=0), venue: str = Query(DEFAULT_VENUE)
    ) -> dict[str, list[object]]:
        """Back-adjusted close series, for charting."""
        history = _windowed(_panel(_venue(venue)), sessions)
        name = symbol.upper()
        rows = series_for(history, name, load_actions(history, [name]))
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=_not_found(history, name))
        return {
            "dates": [d.date().isoformat() for d in rows["event_time"].to_list()],
            "closes": [float(c) for c in rows["close"].to_list()],
        }

    @router.get("/security/{symbol}/ohlc", dependencies=[ReadAccess])
    def ohlc(
        symbol: str, sessions: int = Query(0, ge=0), venue: str = Query(DEFAULT_VENUE)
    ) -> dict[str, object]:
        """Back-adjusted OHLC and volume — what a candle chart needs.

        `series_for` has always returned all five columns; the close-only
        endpoint above simply discarded four of them, which is why the console
        could draw a sparkline and not a chart.

        Back-adjusted for the same reason every other analytics screen is: a
        1:1 bonus read from raw prices is a -50% candle that never happened
        (§9). The panel itself still stores raw prices, because the backtester
        applies actions to positions instead.
        """
        history = _windowed(_panel(_venue(venue)), sessions)
        name = symbol.upper()
        rows = series_for(history, name, load_actions(history, [name]))
        if rows.is_empty():
            raise HTTPException(status_code=404, detail=_not_found(history, name))
        return {
            "symbol": name,
            "dates": [d.date().isoformat() for d in rows["event_time"].to_list()],
            "open": [float(v) for v in rows["open"].to_list()],
            "high": [float(v) for v in rows["high"].to_list()],
            "low": [float(v) for v in rows["low"].to_list()],
            "close": [float(v) for v in rows["close"].to_list()],
            "volume": [float(v) for v in rows["volume"].to_list()],
        }


def _register_quote(router: APIRouter) -> None:
    @router.get("/quote/{symbol}", dependencies=[ReadAccess])
    def quote(symbol: str, venue: str = Query(DEFAULT_VENUE)) -> dict[str, object]:
        """Identity, last close and ADV for one ticker.

        What a ticket needs before it can price anything: an order priced off a
        blank field is not an order, and a market order still needs a reference
        for the fat-finger band to be measured against.
        """
        found = latest_quote(symbol, _venue(venue))
        if found is None:
            raise HTTPException(status_code=404, detail=f"{symbol.upper()} is not in the panel")
        return found


def _register_watchlist(router: APIRouter) -> None:
    @router.get("/watchlist", dependencies=[ReadAccess])
    def watchlist(
        symbols: str = Query("", max_length=1024), venue: str = Query(DEFAULT_VENUE)
    ) -> list[dict[str, object]]:
        """Last close and session change for a handful of names.

        From the panel, not from a vendor. `/quotes` fetches live delayed
        prices and is the right thing mid-session; this is the close-to-close
        move, which is what the charts on the same screen are drawing. Mixing
        the two in one row would put a live price next to a change computed
        from closes and invite the reader to subtract them.

        Unknown symbols are omitted rather than returned as zeros: a watchlist
        row reading 0.00 is indistinguishable from a stock that fell to nothing.
        """
        wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()][:MAX_SYMBOLS]
        if not wanted:
            return []

        recent = _windowed(_panel(_venue(venue)), 2).filter(pl.col("symbol").is_in(wanted))
        rows: list[dict[str, object]] = []
        for symbol in wanted:
            series = recent.filter(pl.col("symbol") == symbol).sort("event_time")
            if series.is_empty():
                continue
            closes = [float(c) for c in series["close"].to_list()]
            last = closes[-1]
            previous = closes[-2] if len(closes) > 1 else None
            rows.append(
                {
                    "symbol": symbol,
                    "last": last,
                    "change_pct": None if previous in (None, 0) else (last / previous - 1) * 100,
                    "volume": float(series["volume"].to_list()[-1]),
                    "as_of": series["event_time"].to_list()[-1].date().isoformat(),
                }
            )
        return rows


class PairRow(BaseModel):
    """One pair, tested for a stationary spread."""

    a: str
    b: str
    cointegrated: bool
    #: Cointegrated *and* reverting fast enough to pay for the round trip.
    tradable: bool
    hedge_ratio: float
    intercept: float
    adf_pvalue: float
    half_life_bars: float | None
    correlation: float
    observations: int
    #: Why `cointegrated` is what it is, and the field to read when the two
    #: look inconsistent. The verdict needs ADF *and* KPSS to agree, so a
    #: spread can reject the unit root at 5% and still land INCONCLUSIVE —
    #: which is not a pair. One test alone would call it one.
    spread_verdict: str
    #: Where the spread sits now, in standard deviations of its own history.
    #: The entry signal: a pair can be cointegrated and have nothing to do.
    spread_z: float | None


class PairsResponse(BaseModel):
    sessions: int
    tested: int
    rows: list[PairRow]
    note: str


def _register_pairs(router: APIRouter) -> None:
    @router.get("/pairs", response_model=PairsResponse, dependencies=[ReadAccess])
    def pairs(
        symbols: str = Query(..., description="Comma-separated symbols, every pair tested"),
        sessions: int = Query(750, ge=60),
        venue: str = Query(DEFAULT_VENUE),
    ) -> PairsResponse:
        """Which of these names share a stationary spread.

        **The one strategy family the console could not reach.** Engle-Granger,
        the hedge ratio and the spread have been built and tested since the
        maths went in, and nothing outside a test had ever called them — a
        whole class of trade was library-only.

        Every unordered pair is tested rather than a chosen few, because which
        pair cointegrates is exactly what is not known in advance; the
        combinatorics are why the symbol list is capped.

        Ordered by p-value, but read `tradable` rather than `cointegrated`: a
        spread whose half-life exceeds a quarter of the sample cannot be
        verified inside it, and one that takes a year to close is a directional
        position wearing a pairs-trade label.

        A low ADF p-value beside `cointegrated: false` is not a contradiction.
        The verdict needs ADF and KPSS to agree, and a spread that rejects the
        unit root while KPSS also rejects stationarity is INCONCLUSIVE — which
        is not a pair. `spread_verdict` carries that reasoning.
        """
        from itertools import combinations  # noqa: PLC0415

        from quant.math.timeseries.cointegration import (  # noqa: PLC0415
            engle_granger,
            spread_series,
        )

        wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()][:MAX_PAIR_SYMBOLS]
        if len(wanted) < MIN_CROSS_SECTION:
            raise HTTPException(status_code=422, detail="a pair needs 2+ symbols")

        history = _windowed(_panel(_venue(venue)), sessions)
        actions = load_actions(history, wanted)
        closes: dict[str, npt.NDArray[np.float64]] = {}
        for symbol in wanted:
            frame = series_for(history, symbol, actions)
            if frame.height >= MIN_PAIR_OBSERVATIONS:
                closes[symbol] = frame["close"].to_numpy()

        if len(closes) < MIN_CROSS_SECTION:
            raise HTTPException(
                status_code=404,
                detail=f"fewer than two of those have {MIN_PAIR_OBSERVATIONS}+ sessions",
            )

        rows: list[PairRow] = []
        for first, second in combinations(sorted(closes), 2):
            left, right = closes[first], closes[second]
            width = min(left.size, right.size)
            left, right = left[-width:], right[-width:]
            try:
                report = engle_granger(left, right)
            except (ValueError, ZeroDivisionError):
                continue

            spread = spread_series(left, right, report.hedge_ratio, report.intercept)
            deviation = float(np.std(spread, ddof=1)) if spread.size > 1 else 0.0
            # None, not zero: a spread with no dispersion has no z-score, and
            # zero would read as "sitting exactly on its mean".
            z = float((spread[-1] - np.mean(spread)) / deviation) if deviation > 0 else None
            half_life = report.half_life_bars if np.isfinite(report.half_life_bars) else None
            rows.append(
                PairRow(
                    a=first,
                    b=second,
                    cointegrated=report.cointegrated,
                    tradable=report.tradable,
                    hedge_ratio=report.hedge_ratio,
                    intercept=report.intercept,
                    adf_pvalue=report.adf_pvalue,
                    half_life_bars=half_life,
                    correlation=report.correlation,
                    observations=report.observations,
                    spread_verdict=report.spread_verdict.value,
                    spread_z=z,
                )
            )

        rows.sort(key=lambda r: (not r.tradable, r.adf_pvalue))
        tradable = sum(1 for r in rows if r.tradable)
        return PairsResponse(
            sessions=sessions,
            tested=len(rows),
            rows=rows,
            note=(
                f"{len(rows)} pair(s) tested, {tradable} tradable. Cointegration found by "
                "searching every pair is the textbook case of a result that needs its "
                "trial count declared before it is believed."
            ),
        )


def _register_cross_section(router: APIRouter) -> None:
    @router.get("/crosssection", response_model=CrossSectionResponse, dependencies=[ReadAccess])
    def crosssection(
        symbols: str = Query(..., description="Comma-separated symbols"),
        sessions: int = Query(750, ge=0),
        venue: str = Query(DEFAULT_VENUE),
    ) -> CrossSectionResponse:
        wanted = [s.strip().upper() for s in symbols.split(",") if s.strip()][:MAX_SYMBOLS]
        if len(wanted) < MIN_CROSS_SECTION:
            raise HTTPException(status_code=422, detail="a cross-section needs 2+ symbols")

        history = _windowed(_panel(_venue(venue)), sessions)
        actions = load_actions(history, wanted)
        kept, matrix = aligned_returns(history, wanted, actions)
        if not matrix.size:
            raise HTTPException(status_code=404, detail="none of those symbols are in the panel")

        try:
            section = analyse_cross_section(kept, matrix)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return CrossSectionResponse(
            names=[
                NameRow(
                    symbol=n.symbol,
                    total_return=n.total_return,
                    annual_volatility=n.annual_volatility,
                    sharpe=n.sharpe,
                    beta=n.beta,
                    correlation_to_market=n.correlation_to_market,
                    weight_hrp=n.weight_hrp,
                    weight_erc=n.weight_erc,
                    cluster=n.cluster,
                )
                for n in section.ranked_by("total_return")
            ],
            sessions=section.sessions,
            mean_correlation=section.mean_correlation,
            clusters=len(section.clusters),
            effective_bets=section.effective_bets,
            diversification_ratio=section.diversification_ratio,
            condition_number=section.condition_number,
            shrinkage=section.shrinkage,
            market_return=section.market_return,
            market_volatility=section.market_volatility,
            is_ill_conditioned=section.is_ill_conditioned,
            concentration_warning=section.concentration_warning,
            correlation_labels=kept,
            correlation=[[float(v) for v in row] for row in section.correlation],
        )


def _register_screen(router: APIRouter) -> None:
    @router.get("/screen", response_model=ScreenResponse, dependencies=[ReadAccess])
    def screen(  # noqa: PLR0913, PLR0917 - a screen is its criteria; grouping them hides the API
        sort: str = Query("liquidity"),
        limit: int = Query(25, ge=1, le=200),
        window: int = Query(250, ge=60),
        min_adv: float = Query(1e7, ge=0),
        stationary_only: bool = Query(default=False),
        venue: str = Query(DEFAULT_VENUE),
    ) -> ScreenResponse:
        """Which names, rather than what is this name.

        Two-stage by construction: the vectorised filter runs over every symbol
        in a fraction of a second, and only the shortlist pays for ADF, KPSS
        and Hurst.
        """
        try:
            key = SortKey(sort)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"unknown sort {sort!r}") from exc

        result = screen_universe(
            _panel(_venue(venue)),
            ScreenCriteria(
                window=window,
                min_adv=min_adv,
                sort_by=key,
                limit=limit,
                stationary_only=stationary_only,
            ),
        )
        rows = []
        for row in result.rows:
            profile = row.profile
            if profile is None:
                continue
            rows.append(
                ScreenRowResponse(
                    symbol=row.symbol,
                    adv=row.adv,
                    bars=row.bars,
                    last_close=row.last_close,
                    window_return=row.window_return,
                    annual_volatility=profile.annual_volatility,
                    sharpe=profile.sharpe,
                    max_drawdown=profile.max_drawdown,
                    hurst=profile.stationarity.hurst,
                    verdict=row.verdict,
                    fadeable=row.fadeable,
                    is_implausible=profile.is_implausible,
                    fat_left_tail=profile.fat_left_tail,
                )
            )
        return ScreenResponse(
            rows=rows,
            considered=result.considered,
            passed_filters=result.passed_filters,
            profiled=result.profiled,
            suspected_actions=result.suspected_actions,
            sort_by=key.value,
        )


def build_analytics_router() -> APIRouter:
    """Every analytics endpoint, registered onto one router.

    Split into three registrars because they share nothing but the router: one
    function holding all of them grows past the complexity limit as endpoints
    are added, and the limit is there to catch exactly that.
    """
    router = APIRouter(tags=["analytics"])
    _register_venues(router)
    _register_search(router)
    _register_security(router)

    @router.get("/sectors", response_model=SectorResponse, dependencies=[ReadAccess])
    def sectors(
        top: int = Query(100, ge=2, le=500),
        venue: str | None = None,
    ) -> SectorResponse:
        """What the tradable universe is made of, by industry."""
        return sector_breakdown(top, _venue(venue))

    _register_quote(router)
    _register_watchlist(router)
    _register_cross_section(router)
    _register_pairs(router)
    _register_screen(router)
    return router
