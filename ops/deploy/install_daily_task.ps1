# Register the nightly ingest with Windows Task Scheduler -- MASTER_PLAN 13.4.
#
#     powershell -ExecutionPolicy Bypass -File ops\deploy\install_daily_task.ps1
#     powershell -ExecutionPolicy Bypass -File ops\deploy\install_daily_task.ps1 -Remove
#
# TIMING. NSE closes 15:30 IST and the bhavcopy publishes about two and a half
# hours later, so 19:15 IST leaves roughly forty-five minutes of slack. IST has
# no daylight saving, so this time is stable year-round -- unlike the GitHub
# workflow, which has to be expressed in UTC because cron there has no
# timezone.
#
# WEEKDAYS ONLY, AND HOLIDAYS ARE NOT HANDLED. The exchange calendar is not
# something Task Scheduler knows, so this fires on Diwali too. That is
# deliberate: the ingest's planner already treats an unpublished session as a
# no-op rather than a failure, so a holiday run costs one request and writes
# nothing. Encoding the holiday calendar in two places is how the two
# eventually disagree.
#
# MISSED RUNS. `StartWhenAvailable` catches the ordinary case of a laptop that
# was shut at 19:15 -- the task runs when the machine next wakes. A run missed
# entirely is recoverable anyway: the planner fetches whatever is missing, so
# the next successful run catches up on its own.

param([switch]$Remove)

$ErrorActionPreference = "Stop"
$name = "neutron-daily"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$script = Join-Path $repo "ops\deploy\daily_task.ps1"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
        Write-Host "removed scheduled task '$name'" -ForegroundColor Yellow
    } else {
        Write-Host "no scheduled task named '$name'" -ForegroundColor DarkGray
    }
    exit 0
}

if (-not (Test-Path $script)) { throw "missing $script" }

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At 19:15

# No `-RunOnlyIfNetworkAvailable`: the check is for a *configured* network, not
# a reachable exchange, so it blocks on a VPN quirk while letting a genuinely
# offline run through. The ingest reports its own failure, which is the honest
# place for it.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
    -Settings $settings -Description "Neutron: fetch every feed's missing sessions" `
    -Force | Out-Null

Write-Host ""
Write-Host "  registered '$name' -- weekdays 19:15 IST" -ForegroundColor Green
Write-Host "  log:    $(Join-Path $repo 'logs\daily.log')"
Write-Host "  run it: Start-ScheduledTask -TaskName $name"
Write-Host "  remove: ops\deploy\install_daily_task.ps1 -Remove"
Write-Host ""
