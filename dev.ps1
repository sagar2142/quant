# Both dev servers, each reloading on its own changes (MASTER_PLAN §13.6, §13.7).
#
#     .\dev.ps1              # API + console
#     .\dev.ps1 -ApiOnly     # API alone, for CLI or endpoint work
#     .\dev.ps1 -Port 8010   # when something else holds 8000
#
# WHY A SCRIPT. Starting this by hand is two terminals, two working directories
# and two environment variables that must agree — and when they disagree the
# console loads normally and shows nothing, because the proxy is pointing at a
# port with no API behind it. That failure looks like a data problem and is not.
#
# WHAT RELOADS. Uvicorn watches the Python packages and restarts the process on
# a change; Vite already hot-reloads the console and needs no help. The watch
# list is explicit rather than the whole tree: `lake/` holds hundreds of Parquet
# files that a running ingest rewrites, and watching it would restart the API
# mid-request every evening.
#
# BINDING. 127.0.0.1 for both, never 0.0.0.0 (§13.7). This surface carries order
# entry and a kill switch; remote access is Tailscale's job.
#
# The API runs in a second window rather than the background, because a reload
# loop that fails to start prints its traceback once and then sits silent — and
# a hidden process cannot be read.

param(
    [int]$Port = 0,
    [switch]$ApiOnly,
    [switch]$WebOnly
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No virtualenv at .venv — run: python -m venv .venv; .\.venv\Scripts\pip install -e '.[dev]'" -ForegroundColor Red
    exit 1
}

# The port is one value shared by two processes. Resolved once here so the
# console's proxy and the API can never be told different numbers.
if ($Port -eq 0) {
    $Port = if ($env:NEUTRON_API_PORT) { [int]$env:NEUTRON_API_PORT } else { 8000 }
}
$env:NEUTRON_API_PORT = "$Port"

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy -and -not $WebOnly) {
    # Named rather than "address in use": a stale uvicorn from an earlier
    # session serving old code is the common case, and it presents as edits
    # that appear to do nothing.
    $owner = (Get-Process -Id $busy[0].OwningProcess -ErrorAction SilentlyContinue).ProcessName
    Write-Host "Port $Port is already held by $owner (pid $($busy[0].OwningProcess))." -ForegroundColor Red
    Write-Host "Stop it, or choose another:  .\dev.ps1 -Port 8010" -ForegroundColor DarkGray
    exit 1
}

Write-Host ""
Write-Host "  API      http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "  Console  http://127.0.0.1:5173" -ForegroundColor Cyan
if (-not $env:NEUTRON_API_TOKEN) {
    Write-Host "  NEUTRON_API_TOKEN unset — reads work, mutations return 503" -ForegroundColor DarkGray
}
Write-Host ""

# Only the source packages. `lake/`, `paper/` and `.venv/` are excluded by
# omission: a reload triggered by an ingest writing Parquet would restart the
# API in the middle of serving the screen you are looking at.
$watch = @("core", "data", "quant", "engine", "trading", "ops", "apps/api", "apps/cli")
$reloadArgs = $watch | ForEach-Object { "--reload-dir", $_ }

if (-not $WebOnly) {
    $api = {
        param($py, $port, $reloadArgs)
        & $py -m uvicorn apps.api.main:app --host 127.0.0.1 --port $port --reload @reloadArgs
    }
    if ($ApiOnly) {
        & $api.Invoke($py, $Port, $reloadArgs)
        exit $LASTEXITCODE
    }
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "& '$py' -m uvicorn apps.api.main:app --host 127.0.0.1 --port $Port --reload $($reloadArgs -join ' ')"
    )
    Write-Host "API started in a second window. Close it to stop the API." -ForegroundColor DarkGray
}

if (-not $ApiOnly) {
    Push-Location .\apps\web
    try {
        if (-not (Test-Path node_modules)) {
            Write-Host "Installing console dependencies..." -ForegroundColor DarkGray
            npm install --no-fund --no-audit
        }
        npm run dev
    } finally {
        Pop-Location
    }
}
