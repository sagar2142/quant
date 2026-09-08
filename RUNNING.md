# Running neutron

Every command that starts something, in the order you need them.

---

## First time

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
docker compose up -d postgres
```

Postgres listens on **127.0.0.1:5433**, not 5432 — that port is commonly taken.
Use `127.0.0.1`, never `localhost`: on Windows `localhost` resolves to `::1`
first while Docker publishes IPv4 only, so a wrong host makes connections *hang*
rather than fail, which is much harder to diagnose.

The schema in `db/migrations/` applies automatically on first container start.

Postgres is optional for research. Without it, experiment recording is skipped
and says so; backtests, the gauntlet and the console all work.

---

## Run the console

```powershell
.\dev.ps1
```

| | |
|---|---|
| API | http://127.0.0.1:8000 |
| Console | http://127.0.0.1:5173 |

Open the console. Both halves reload on save — uvicorn restarts the API when a
source package changes, Vite hot-reloads the browser.

```powershell
.\dev.ps1 -ApiOnly      # API alone, for endpoint or CLI work
.\dev.ps1 -Port 8010    # when something else holds 8000
```

`dev.ps1` runs `npm install` itself if `apps/web/node_modules` is missing.

**Only `dev.ps1` should start the API.** It resolves the port once and hands the
same number to both processes. Starting them separately with mismatched ports
gives you a console that loads perfectly and shows nothing at all — which reads
as a data problem, so the configuration error is the last thing anyone checks.

---

## Feed the data

```powershell
python -m apps.cli.daily                    # every feed, whatever each is missing
python -m apps.cli.daily --events --alerts  # + calendar, + evaluate alert rules
python -m apps.cli.daily --dry-run          # say what it would fetch, fetch nothing
```

One command covers NSE, BSE, F&O and the index series, plus the cadenced feeds —
industry classification, quarterly results, the announced board-meeting calendar.
It works out what each is missing and refuses to ask for files that are not
published yet, so it is safe to run repeatedly and safe to run at the wrong time
of day.

Individual feeds, when you want one specifically:

```powershell
python -m apps.cli.ingest_nse --start 2019-01-01     # equity panel (2.5h cold)
python -m apps.cli.ingest_results --start 2019       # quarterly results + XBRL
python -m apps.cli.ingest_events                     # announced calendar
python -m apps.cli.ingest_sectors                    # industry classification
python -m apps.cli.quality                           # data-quality suite
```

---

## Research

```powershell
python -m apps.cli.screen --sort liquidity --limit 20
python -m apps.cli.screen --sort reversal --stationary-only

python -m apps.cli.factor momentum_12_1 --horizon 5   # IC, quantiles, turnover
python -m apps.cli.factor --all --sessions 1000

python -m apps.cli.backtest --strategy momentum --top 30
python -m apps.cli.backtest --strategy momentum --cost-multiple 3

python -m apps.cli.validate --top 30                  # the twelve-check gauntlet
python -m apps.cli.terminal RELIANCE TCS INFY         # analytics without a browser
```

The register, which is what keeps the trial count honest:

```powershell
python -m apps.cli.preregister --list
python -m apps.cli.preregister --register
python -m apps.cli.resolve --dry-run
python -m apps.cli.resolve --commit
```

Run `--dry-run` before `--commit`. Committing writes verdicts against the
criteria registered beforehand, and that is the whole point of registering them.

---

## Paper trading

```powershell
python -m apps.cli.paper --top 30
```

One invocation is one cycle: read the latest session, produce target weights,
size, risk-check, submit, apply fills, reconcile, persist, exit. It is a
scheduled job, not a daemon.

In normal operation you do not run this by hand — `.github/workflows/paper.yml`
fires it at 13:30 UTC (19:00 IST) on weekdays. **Scheduled workflows run only
from the default branch**, so a fix on a feature branch changes nothing until it
reaches `main`.

Progress shows in the console's System panel: cycles against the thirty the M9
gate asks for, days since the last one, halt state.

**Exit codes matter here.** `0` ran · `1` could not run (no data, no universe) ·
`2` HALTED. A scheduler must treat `2` as "page a human", never as "retry" — it
means reconciliation found a break, and the halt persists across runs until
someone clears it deliberately:

```powershell
python -m apps.cli.paper --clear-halt   # a human decision, not a retry
```

---

## Checks

```powershell
.\check.ps1
```

ruff, ruff-format, mypy --strict, import boundaries, the custom AST lints, the
test suite, and the console typecheck. The same checks CI runs, so green here
means a green pipeline.

---

## When it does not work

**The console loads but every screen is empty.**
The proxy is pointing at a port with no API behind it. Check the API is up:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health -UseBasicParsing
```

**Your edits appear to do nothing.**
A stale server from an earlier session is still holding the port, serving old
code. This is common and it does not announce itself. Find it:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  ForEach-Object { Get-Process -Id $_.OwningProcess }
```

`dev.ps1` refuses to start when the port is held and names the process, which is
the reason to use it rather than starting uvicorn by hand.

**`apps.cli.paper` refuses to run.**

```
the paper trading cycle must not run with live trading enabled.
```

`.env` has `NEUTRON_ENV=live`. Paper and live are separate planes (§21) and the
guard is working. Set `NEUTRON_ENV=paper` (or `dev`) to run the cycle locally.
The GitHub runner is unaffected — `.env` is gitignored, and the default is `dev`.

**Mutating endpoints return 503.**
`NEUTRON_API_TOKEN` is unset. Reads work without it; mutations do not. An
install that was never configured cannot release a halt, which is the fail-safe
direction — engaging a halt is recoverable, releasing one is not.

**Nothing here trades.** `NEUTRON_LIVE_ENABLED` defaults to `false`, and four
independent guards stand between this code and a real order. Run
`python -m apps.cli.readiness` for the live checklist.
