# Register the daily ingest with Windows Task Scheduler — MASTER_PLAN §13.4.
#
#   pwsh -File ops/deploy/daily_task.ps1              # register
#   pwsh -File ops/deploy/daily_task.ps1 -Unregister  # remove
#   pwsh -File ops/deploy/daily_task.ps1 -Time 20:15  # a different hour
#
# Why a scheduled task rather than a GitHub Action: the lake is on this
# machine. A hosted runner has nowhere to write, and shipping seven years of
# Parquet to CI to add one session to it is the wrong shape of problem.
#
# Runs at 19:45 IST by default — after the last of the four feeds publishes,
# with room for NSE being late. The planner refuses sessions that are not
# published yet, so an early run is a no-op rather than a day recorded as
# unavailable; the margin is for convenience, not correctness.
#
# The task runs whether or not you are logged in is deliberately NOT set: it
# runs as you, in your session, so it inherits your environment (including
# .env) and writes files you own. A missed evening is caught up the next
# night, because the planner works from what the lake holds rather than from
# a schedule it assumes was kept.

[CmdletBinding()]
param(
    [string]$Time = "19:45",
    [string]$TaskName = "Neutron daily ingest",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
$log = Join-Path $logDir "daily.log"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
    return
}

if (-not (Test-Path $python)) {
    throw "No interpreter at $python. Create the virtualenv first."
}
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# The IST publication times are what matter, but the task runs on this
# machine's clock. If you are not in IST, set -Time to the local equivalent of
# roughly 19:45 IST; the planner will decline anything not yet published, so
# being late costs nothing and being early costs a retry tomorrow.
$command = "& '$python' -m apps.cli.daily --check *>> '$log'"
$action = New-ScheduledTaskAction -Execute "pwsh.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$command`"" `
    -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -Daily -At $Time

# StartWhenAvailable is the point of the whole arrangement: a laptop asleep at
# 19:45 runs the job when it wakes, and the planner fills whatever gap opened.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Fetch NSE, BSE, F&O and index sessions the lake is missing." `
    -Force | Out-Null

Write-Host "Registered '$TaskName' for $Time daily."
Write-Host "  repo:  $repo"
Write-Host "  log:   $log"
Write-Host ""
Write-Host "Check it without waiting:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "  $python -m apps.cli.daily --dry-run"
