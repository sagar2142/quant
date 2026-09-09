# The nightly ingest, as Windows Task Scheduler runs it -- MASTER_PLAN 13.4.
#
# Install with `ops\deploy\install_daily_task.ps1`. Run it by hand any time --
# it is the same command the scheduler uses, so testing it tests the real thing.
#
# WHY A SCRIPT RATHER THAN A SCHTASKS ONE-LINER. A scheduled task that fails
# silently is the failure this system keeps finding: the GitHub workflow was
# scheduled for fifteen days and produced nothing, and nothing anywhere said so.
# The one-liner has no log, so the only evidence of a failed run is a lake that
# quietly stops advancing. This appends every run to a file with its exit code,
# which makes "did it run last night?" a question with an answer.
#
# The log is trimmed rather than rotated. It gains a handful of lines a day and
# nobody is going to configure logrotate for it, so it keeps the last few
# thousand and forgets the rest.

$ErrorActionPreference = "Continue"

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
$log = Join-Path $logDir "daily.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log($message) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $message
    Add-Content -Path $log -Value $line
    Write-Host $line
}

if (-not (Test-Path $python)) {
    Write-Log "FAILED  no virtualenv at $python"
    exit 1
}

Write-Log "start   ingest"
$env:PYTHONPATH = $repo
# 2>&1 because the exchanges' failures arrive on stderr and a log missing them
# would show a clean run that fetched nothing.
$output = & $python -m apps.cli.daily --events --alerts --pause 1.0 2>&1
$code = $LASTEXITCODE

foreach ($line in $output) { Add-Content -Path $log -Value ("        " + $line) }

if ($code -eq 0) {
    Write-Log "ok      ingest complete"

    # The simulation runs here and nowhere else. It was also in the GitHub
    # workflow, which committed its book back to the repo -- two machines
    # trading one account produce two different books and a merge conflict on
    # the next pull, so there is exactly one owner and this is it.
    #
    # Only after a clean ingest: a cycle on a stale panel would trade
    # yesterday's prices and record the result as today's.
    Write-Log "start   simulation cycle"
    $cycle = & $python -m apps.cli.paper --top 30 2>&1
    $cycleCode = $LASTEXITCODE
    foreach ($line in $cycle) { Add-Content -Path $log -Value ("        " + $line) }

    if ($cycleCode -eq 2) {
        # A reconciliation halt. It survives restarts and only a human clears
        # it, so every later run will keep reporting this until someone does.
        Write-Log "HALTED  reconciliation break -- investigate, then: python -m apps.cli.paper --clear-halt"
    } elseif ($cycleCode -ne 0) {
        Write-Log "FAILED  cycle exit $cycleCode"
    } else {
        Write-Log "ok      cycle complete"
    }
} else {
    # Non-zero means a feed that was due went unfetched. Named rather than
    # swallowed: this is the line someone greps for when a screen looks stale.
    Write-Log "FAILED  ingest exit $code -- a due feed went unfetched"
}

# Keep the tail. Cheap, and it runs after the write so a crash mid-ingest still
# leaves the evidence of what it was doing.
if (Test-Path $log) {
    $kept = Get-Content $log -Tail 4000
    Set-Content -Path $log -Value $kept
}

exit $code
