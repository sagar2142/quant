/**
 * One security, decomposed — MASTER_PLAN §12.6.
 *
 * A single window of blocks rather than a sequence of screens. Deciding
 * whether a name is worth trading means holding its price action, its
 * distribution, its drawdown and its stationarity in view at once; paging
 * between them turns a comparison into an act of memory.
 *
 * **Every number comes from the server, and every chart from the same series.**
 * `profile_security` is what `apps.cli.terminal` prints, so the console and the
 * terminal cannot disagree about what a security is. The drawdown, histogram
 * and rolling-volatility panes are derived here from the same back-adjusted
 * OHLC the candles use — one download, one truth, no second opinion.
 *
 * **The verdict block is not a recommendation.** ADF, KPSS and Hurst describe
 * whether a series has been mean-reverting, which is a property of the past.
 * It is shown next to the tail statistics precisely so that "tradable as mean
 * reversion" is read beside "this name has a fat left tail".
 */

import { useCallback, useEffect, useState } from "react";
import { Block } from "./Block";
import { Chart } from "./Chart";
import { MiniChart, drawdownSeries, histogram, returnSeries, rollingVol } from "./MiniChart";
import { OptionChain } from "./OptionChain";
import { SymbolSearch } from "./SymbolSearch";

const VOL_WINDOW = 21;

interface HorizonReturn {
  label: string;
  value: number;
}

interface Security {
  symbol: string;
  observations: number;
  last_close: number;
  horizons: HorizonReturn[];
  cagr: number;
  high_52w: number;
  low_52w: number;
  off_high: number;
  annual_volatility: number;
  max_drawdown: number;
  current_drawdown: number;
  adv_value: number | null;
  sharpe: number;
  sortino: number;
  calmar: number;
  hit_rate: number;
  skewness: number;
  kurtosis: number;
  var_5: number;
  cvar_5: number;
  tail_ratio: number;
  verdict: string;
  adf_pvalue: number;
  kpss_pvalue: number;
  hurst: number;
  tradable_as_mean_reversion: boolean;
  autocorrelation: Record<string, number>;
  realised_vol: number;
  ewma_vol: number;
  vol_regime: string;
  is_implausible: boolean;
  fat_left_tail: boolean;
}

interface Ohlc {
  dates: string[];
  close: number[];
  volume: number[];
}

function pct(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function num(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function money(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  if (value >= 1e7) return `${(value / 1e7).toFixed(2)} Cr`;
  if (value >= 1e5) return `${(value / 1e5).toFixed(2)} L`;
  return value.toFixed(0);
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "up" | "down" }) {
  return (
    <div className="stat">
      <dt>{label}</dt>
      <dd className={tone ? `mono ${tone}` : "mono"}>{value}</dd>
    </div>
  );
}

function signOf(value: number | null | undefined): "up" | "down" | undefined {
  if (value == null || !Number.isFinite(value) || value === 0) return undefined;
  return value > 0 ? "up" : "down";
}

export interface ResearchProps {
  symbol: string;
  venue: string;
  onSymbolChange: (symbol: string) => void;
  onVenueChange: (venue: string) => void;
  onTrade?: (symbol: string) => void;
}

export function Research({ symbol, venue, onSymbolChange, onVenueChange, onTrade }: ResearchProps) {
  const [security, setSecurity] = useState<Security | null>(null);
  const [series, setSeries] = useState<Ohlc | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  //: Whether this name has listed derivatives. Asked once per symbol, because
  //: only 216 of roughly three thousand do and offering an empty chain tab for
  //: the rest would be offering a dead end.
  const [hasOptions, setHasOptions] = useState(false);
  //: Which block, if any, has taken over the window. One at a time: two
  //: expanded blocks is just the grid again, with fewer of them.
  const [expanded, setExpanded] = useState<string | null>(null);

  const toggle = useCallback(
    (id: string) => setExpanded((current) => (current === id ? null : id)),
    [],
  );

  const load = useCallback(() => {
    if (!symbol) {
      setSecurity(null);
      setSeries(null);
      setError("");
      return;
    }
    setLoading(true);
    setError("");
    const query = `venue=${encodeURIComponent(venue)}`;
    Promise.all([
      fetch(`/api/security/${encodeURIComponent(symbol)}?${query}`).then(async (r) => {
        if (!r.ok) {
          const body = (await r.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(body?.detail ?? `HTTP ${r.status}`);
        }
        return (await r.json()) as Security;
      }),
      fetch(`/api/security/${encodeURIComponent(symbol)}/ohlc?${query}`).then((r) =>
        r.ok ? (r.json() as Promise<Ohlc>) : null,
      ),
    ])
      .then(([profile, ohlc]) => {
        setSecurity(profile);
        setSeries(ohlc);
      })
      .catch((exc: Error) => {
        setSecurity(null);
        setSeries(null);
        setError(exc.message);
      })
      .finally(() => setLoading(false));
  }, [symbol, venue]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!symbol) {
      setHasOptions(false);
      return;
    }
    let cancelled = false;
    fetch(`/api/options/underlyings?q=${encodeURIComponent(symbol)}`)
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: string[]) => {
        if (!cancelled) setHasOptions(Array.isArray(rows) && rows.includes(symbol.toUpperCase()));
      })
      .catch(() => !cancelled && setHasOptions(false));
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  const closes = series?.close ?? [];
  const returns = returnSeries(closes);
  const drawdown = drawdownSeries(closes);
  const vol = rollingVol(returns, VOL_WINDOW);
  const bars = histogram(returns);

  return (
    <section className="research" aria-label="Security analysis">
      <header className="charts-bar">
        <SymbolSearch
          symbol={symbol}
          venue={venue}
          onSymbolChange={onSymbolChange}
          onVenueChange={onVenueChange}
        />
        {security && (
          <>
            <span className="research-price mono">{security.last_close.toFixed(2)}</span>
            <span className="muted">{security.observations.toLocaleString()} sessions</span>
            {security.is_implausible && (
              <span className="badge blocked" title="The series contains moves the data quality checks flagged">
                implausible data
              </span>
            )}
          </>
        )}
        {onTrade && symbol && (
          <button type="button" className="ghost" onClick={() => onTrade(symbol)}>
            trade
          </button>
        )}
        <span className="charts-hint">{loading ? "loading…" : error}</span>
      </header>

      {!symbol && <p className="empty">Search for a security to analyse it.</p>}
      {error && symbol && <p className="ticket-message">{error}</p>}

      {symbol && security && (
        <div className={expanded ? "research-grid is-expanded" : "research-grid"}>
          <Block title="Price" id="price" wide expanded={expanded === "price"} onToggle={toggle}>
            <Chart
              symbol={symbol}
              venue={venue}
              sessions={0}
              height={expanded === "price" ? Math.max(320, window.innerHeight - 190) : 300}
            />
          </Block>

          <Block title="Returns" id="returns" expanded={expanded === "returns"} onToggle={toggle}>
            <dl className="stats">
              {security.horizons.map((h) => (
                <Stat key={h.label} label={h.label} value={pct(h.value)} tone={signOf(h.value)} />
              ))}
              <Stat label="CAGR" value={pct(security.cagr)} tone={signOf(security.cagr)} />
              <Stat label="hit rate" value={pct(security.hit_rate)} />
            </dl>
          </Block>

          <Block title="Range" id="range" expanded={expanded === "range"} onToggle={toggle}>
            <dl className="stats">
              <Stat label="52w high" value={num(security.high_52w)} />
              <Stat label="52w low" value={num(security.low_52w)} />
              <Stat label="off high" value={pct(security.off_high)} tone="down" />
              <Stat label="ADV" value={money(security.adv_value)} />
            </dl>
          </Block>

          <Block
            title="Risk-adjusted"
            id="riskadj"
            expanded={expanded === "riskadj"}
            onToggle={toggle}
          >
            <dl className="stats">
              <Stat label="Sharpe" value={num(security.sharpe)} tone={signOf(security.sharpe)} />
              <Stat label="Sortino" value={num(security.sortino)} tone={signOf(security.sortino)} />
              <Stat label="Calmar" value={num(security.calmar)} tone={signOf(security.calmar)} />
              <Stat label="ann. vol" value={pct(security.annual_volatility)} />
            </dl>
          </Block>

          <Block title="Drawdown" id="drawdown" expanded={expanded === "drawdown"} onToggle={toggle}>
            <MiniChart
              values={drawdown}
              kind="area"
              colour="#ef5350"
              baseline={0}
              height={expanded === "drawdown" ? Math.max(160, window.innerHeight - 280) : 104}
            />
            <dl className="stats compact">
              <Stat label="max" value={pct(security.max_drawdown)} tone="down" />
              <Stat label="current" value={pct(security.current_drawdown)} tone="down" />
            </dl>
          </Block>

          <Block
            title="Daily return distribution"
            id="histogram"
            expanded={expanded === "histogram"}
            onToggle={toggle}
          >
            <MiniChart
              values={bars.counts}
              kind="histogram"
              colour="#4a9eff"
              height={expanded === "histogram" ? Math.max(160, window.innerHeight - 280) : 104}
              first={pct(bars.low)}
              last={pct(bars.high)}
            />
            <dl className="stats compact">
              <Stat label="skew" value={num(security.skewness)} tone={signOf(security.skewness)} />
              <Stat label="kurtosis" value={num(security.kurtosis)} />
            </dl>
          </Block>

          <Block title="Tails" id="tails" expanded={expanded === "tails"} onToggle={toggle}>
            <dl className="stats">
              <Stat label="VaR 5%" value={pct(security.var_5)} tone="down" />
              <Stat label="CVaR 5%" value={pct(security.cvar_5)} tone="down" />
              <Stat label="tail ratio" value={num(security.tail_ratio)} />
              <Stat
                label="fat left tail"
                value={security.fat_left_tail ? "yes" : "no"}
                tone={security.fat_left_tail ? "down" : undefined}
              />
            </dl>
          </Block>

          <Block
            title={`Volatility · ${VOL_WINDOW}d rolling`}
            id="vol"
            expanded={expanded === "vol"}
            onToggle={toggle}
          >
            <MiniChart
              values={vol}
              kind="line"
              colour="#f0a93b"
              height={expanded === "vol" ? Math.max(160, window.innerHeight - 280) : 104}
            />
            <dl className="stats compact">
              <Stat label="realised" value={pct(security.realised_vol)} />
              <Stat label="EWMA" value={pct(security.ewma_vol)} />
              <Stat label="regime" value={security.vol_regime} />
            </dl>
          </Block>

          <Block
            title="Stationarity"
            id="stationarity"
            expanded={expanded === "stationarity"}
            onToggle={toggle}
          >
            <dl className="stats">
              <Stat label="ADF p" value={num(security.adf_pvalue, 3)} />
              <Stat label="KPSS p" value={num(security.kpss_pvalue, 3)} />
              <Stat label="Hurst" value={num(security.hurst, 3)} />
              <Stat
                label="mean-reverting"
                value={security.tradable_as_mean_reversion ? "yes" : "no"}
              />
            </dl>
            <p className="block-note">
              A property of the past, not a recommendation. Read it beside the tail statistics.
            </p>
          </Block>

          <Block
            title="Autocorrelation"
            id="autocorr"
            expanded={expanded === "autocorr"}
            onToggle={toggle}
          >
            <dl className="stats">
              {Object.entries(security.autocorrelation).map(([lag, value]) => (
                <Stat key={lag} label={`lag ${lag}`} value={num(value, 3)} tone={signOf(value)} />
              ))}
            </dl>
          </Block>

          {hasOptions && (
            <Block
              title="Option chain"
              id="chain"
              wide
              expanded={expanded === "chain"}
              onToggle={toggle}
            >
              <OptionChain underlying={symbol} />
            </Block>
          )}

          <Block title="Verdict" id="verdict" wide expanded={expanded === "verdict"} onToggle={toggle}>
            <p className="verdict">{security.verdict}</p>
          </Block>
        </div>
      )}
    </section>
  );
}
