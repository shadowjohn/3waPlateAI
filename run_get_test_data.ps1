[CmdletBinding()]
param(
    [string]$Count = "30",
    [string]$Output = "test_data",
    [int]$Workers = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (Test-Path -LiteralPath $VenvPython) {
    $PythonExe = $VenvPython
} else {
    $PythonExe = 'python'
}

$ScriptPath = Join-Path $ProjectRoot 'tools\fetch_sample_test_data.py'

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 3waPlateAI - Fetch Real Taiwan License Plate Test Photos" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Target folder: $Output"
Write-Host "Sample count:  $Count"
Write-Host ""

& $PythonExe $ScriptPath --output (Join-Path $ProjectRoot $Output) --count $Count --workers $Workers

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to fetch test data with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
