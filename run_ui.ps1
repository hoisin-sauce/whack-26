[CmdletBinding()]
param(
    [switch]$Reinstall,
    [switch]$Mock,
    [switch]$Offline,
    [switch]$NoLaunch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "Creating .venv with Python 3.12..."
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
        throw "Could not create .venv. Install Python 3.12 and retry."
    }
}

$needsInstall = $Reinstall
if (-not $needsInstall) {
    & $Python -c "import PyQt6" 2>$null
    $needsInstall = $LASTEXITCODE -ne 0
}
if ($needsInstall) {
    Write-Host "Installing the optional UI stack..."
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $Python -m pip install -r requirements-ui.txt
    if ($LASTEXITCODE -ne 0) { throw "UI dependency install failed." }
}

if ($Mock) {
    $env:WHACK26_LLM_PROVIDER = "mock"
    $env:WHACK26_EMBED_PROVIDER = "mock"
    Write-Host "Mock providers enabled."
}
if ($Offline) {
    $env:WHACK26_OFFLINE = "1"
    Write-Host "Offline mode enabled."
}

if (-not (Test-Path (Join-Path $Root ".env")) -and -not $Mock) {
    Write-Warning "No .env file found. The live pipeline needs DEEPSEEK_API_KEY, GEMINI_API_KEY, and OPENALEX_API_KEY. Use -Mock for a keyless demo."
}

if ($NoLaunch) {
    Write-Host "Setup complete. UI launch skipped because -NoLaunch was supplied."
    exit 0
}

Write-Host "Starting the dashboard. Drop a CSV/XLSX into a tab; the pipeline starts automatically."
& $Python UI_frontEnd.py
exit $LASTEXITCODE
