# The full enforcement stack in one command (MASTER_PLAN §14.8).
# Same checks CI runs, so a green run here means a green pipeline.
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$failed = @()

function Step($name, $script) {
    Write-Host "`n── $name " -NoNewline -ForegroundColor Cyan
    Write-Host ("─" * [Math]::Max(0, 60 - $name.Length)) -ForegroundColor DarkGray
    & $script
    if ($LASTEXITCODE -ne 0) { $script:failed += $name }
}

Step "ruff (lint)"          { & $py -m ruff check . --output-format=concise }
Step "ruff (format)"        { & $py -m ruff format --check . }
Step "mypy --strict"        { & $py -m mypy core data quant engine trading ops apps tools --no-pretty }
Step "import boundaries"    { & .\.venv\Scripts\lint-imports.exe }
Step "neutron AST lints"    { & $py -m tools.lints }
Step "tests"                { & $py -m pytest -q }

# Frontend, when its dependencies are installed. Skipped rather than failed
# otherwise: the Python system is complete without the console.
if (Test-Path ".\apps\web\node_modules") {
    Step "console typecheck" { Push-Location .\apps\web; & npx tsc --noEmit; $code = $LASTEXITCODE; Pop-Location; $global:LASTEXITCODE = $code }
} else {
    Write-Host "`n── console typecheck " -NoNewline -ForegroundColor DarkGray
    Write-Host "(skipped: run npm install in apps/web)" -ForegroundColor DarkGray
}

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
}
Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
exit 1
