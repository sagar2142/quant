# Neutron

Personal multi-market quantitative research and trading system.

Design, rationale and roadmap live in **[MASTER_PLAN.md](MASTER_PLAN.md)**. This file is only how to run it.

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d postgres
```

Postgres listens on **127.0.0.1:5433** (not 5432 — that port is commonly taken).
Use `127.0.0.1`, never `localhost`: on Windows `localhost` resolves to `::1`
first while Docker publishes IPv4 only, which makes connections hang rather
than fail.

The schema in `db/migrations/` is applied automatically on first container
start. To reapply from scratch: `docker compose down -v; docker compose up -d postgres`.

---

## Daily commands

```powershell
# Ingest NSE sessions. Sessions already in the panel are skipped, so this is
# safe to re-run and safe to interrupt — a multi-year backfill resumes for free.
.\.venv\Scripts\python.exe -m apps.cli.ingest_nse --start 2019-01-01

# Backtest the ingested panel (corporate actions on by default)
.\.venv\Scripts\python.exe -m apps.cli.backtest --strategy momentum --top 30

# The validation gauntlet. Expect rejection — that is the system working.
.\.venv\Scripts\python.exe -m apps.cli.validate --top 30

# One paper trading cycle. Run it once per session, after the ingest.
.\.venv\Scripts\python.exe -m apps.cli.paper --top 30

# Backtest + gauntlet as one self-contained HTML file you can keep
.\.venv\Scripts\python.exe -m apps.cli.report --top 30 --out reports\momentum.html

# New here? The research loop, explained end to end
.\.venv\Scripts\python.exe -m apps.cli.tutorial
.\.venv\Scripts\python.exe -m apps.cli.tutorial --principles
# ...or the Tutorial screen in the console (key 't'), shown on first visit

# Factor lab — score a signal against forward returns in about a second
.\.venv\Scripts\python.exe -m apps.cli.factor momentum_12_1
.\.venv\Scripts\python.exe -m apps.cli.factor --all

# Combine signals into one composite, and check they are not duplicates
.\.venv\Scripts\python.exe -m apps.cli.factor --overlap
.\.venv\Scripts\python.exe -m apps.cli.factor --combine momentum_12_1,volatility_60

# Screen the whole universe — which names, not what is this name
.\.venv\Scripts\python.exe -m apps.cli.screen --sort momentum --limit 20
.\.venv\Scripts\python.exe -m apps.cli.screen --stationary-only

# Analytics terminal — one name, fully decomposed
.\.venv\Scripts\python.exe -m apps.cli.terminal RELIANCE

# ...or a cross-section: correlation, clusters, effective bets, HRP/ERC weights
.\.venv\Scripts\python.exe -m apps.cli.terminal RELIANCE TCS INFY HDFCBANK --sessions 750

# Data quality report; exits non-zero on any CRITICAL finding
.\.venv\Scripts\python.exe -m apps.cli.quality

# The full enforcement stack (MASTER_PLAN §14.8)
.\.venv\Scripts\python.exe -m ruff check .        # lint, complexity, dead code
.\.venv\Scripts\python.exe -m ruff format .       # format
.\.venv\Scripts\python.exe -m mypy core data quant engine trading ops   # types
.\.venv\Scripts\lint-imports.exe                  # architecture boundaries (§3.2)
.\.venv\Scripts\python.exe -m tools.lints         # project AST lints (§14.8)
.\.venv\Scripts\python.exe -m pytest -q           # tests

# Or all of it at once
.\check.ps1
```

---

## Layout

| Path | Role |
|---|---|
| `core/` | Shared kernel: clocks, instruments, calendars, events. Depends on nothing. |
| `data/` | Feeds, Parquet lake, quality checks, point-in-time universes. |
| `quant/` | Mathematics, analytics, strategies. **Never imports `trading/`.** |
| `engine/` | Backtester, cost models, validation gauntlet, experiment registry. |
| `trading/` | Portfolio, risk, paper, execution, reconciliation. |
| `ai/` | Research agents. No import path to `trading/` or secrets. M13+. |
| `apps/` | `cli/` commands, `report/` static HTML research reports, `api/` FastAPI, `web/` React console. |
| `db/migrations/` | Schema. The research protocol is enforced here, not in app code. |
| `tools/lints/` | The five custom AST lints. |

Boundaries between these are enforced by `import-linter` and fail CI. They are
not conventions — see `.importlinter` for why each one exists.

---

## Three things that will bite you if you forget them

**1. There is no way to read "all bars".** The only read path is
`store.view(instrument, timeframe, as_of=decision_time)`, which filters on
`receive_time <= as_of`. This makes look-ahead bias structurally impossible
rather than something you test for. A lint rejects `.view()` without `as_of`.

**2. Float for statistics, Decimal for money.** If a broker could disagree with
you about a number, it is `Decimal`. A lint rejects `float` anywhere under
`trading/portfolio`, `trading/execution`, `trading/reconcile` or `engine/costs`.
The same modules, plus `engine/accounting.py` and `engine/backtest/sizing.py`,
carry a 100% coverage gate in CI.

**3. `receive_time` is never a copy of `event_time`.** Every loader stamps a
real publication lag. If they were equal, a strategy could act on a bar's close
at the instant of that close, which no live system can do.

**4. A held name that stops trading keeps its last traded mark.** Suspensions,
renames and delistings happen mid-backtest. Valuing such a position at cost
basis instead silently erases its unrealised P&L from the equity curve — an
error that produces a plausible curve rather than an exception.

---

## Status

| Milestone | Machinery | Gate |
|---|---|---|
| M1 — foundation, enforcement, core domain, NSE data | ✅ | ✅ met |
| M2 — point-in-time data, corporate actions, universes | ✅ | ✅ met (1,871 continuous NSE sessions, 2019→2026) |
| M3–M4 — backtester, cost engine, accounting | ✅ | ✅ met (all four) |
| M5 — math library, DSR/PBO, 12-check gauntlet | ✅ | ✅ met |
| M6–M7 — strategy families, hypothesis registry | ✅ | ⏳ needs your research |
| M8 — risk engine, ladder, kill switch, allocation | ✅ | ✅ met |
| M9–M10 — paper trading, reconciliation, ops console | ✅ | ⏳ needs 6 weeks of calendar |
| M11 — live readiness, Kite adapter, runbooks | ✅ | ⏳ needs attestation |
| M11+ — Tauri desktop shell | ✅ | — |

**Machinery** means the code exists, is tested and passes every check.
**Gate** is the plan's own acceptance criterion, and three of them cannot be met
by writing code:

- **M6–M7** wants 15 hypotheses tested with ≥90% rejected. That is research,
  and it needs economic mechanisms you actually believe.
- **M9** wants 6+ weeks of continuous paper trading. Calendar time. The loop
  itself is built and runs free: see `ops/deploy/github_actions_paper.yml` for
  the scheduled job, one cycle per session.
- **M11** wants human attestations — legal advice, tax position, tested kill
  switch. Run `python -m apps.cli.readiness` to see the list.

### Console API authentication

The API is unauthenticated on loopback for *reads* and refuses *mutations*
until you configure a token:

```powershell
# generate one, put it in .env, restart both servers
NEUTRON_API_TOKEN=<64 hex chars>

# the console picks it up from the same variable
$env:NEUTRON_API_TOKEN="..."; $env:NEUTRON_API_PORT=8010; npm run dev
```

**Unset**: analytics and screens work; `/kill` and `/kill/release` return 503.
An install that was never configured cannot release a halt, which is the
fail-safe direction — engaging a halt is recoverable, releasing one is not.

**Set**: required on every endpoint, reads included. The dev-server proxy
injects it so the token never reaches browser JavaScript.

**Required before this API binds to anything but 127.0.0.1.**

### Nothing here trades

`NEUTRON_LIVE_ENABLED` defaults to `false`. Four independent guards stand between
this code and a real order (see `trading/execution/kite.py`), and
`core.secrets.load_broker_credentials` refuses to load a broker key unless
**both** `env=live` and `live_enabled=true`.

```powershell
python -m apps.cli.readiness   # the PDF §40 checklist, executable
```
