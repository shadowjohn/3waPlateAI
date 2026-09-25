param(
    [string]$Python = $env:PLATEAI_SERVICE_PYTHON,
    [string]$Config = (Join-Path $PSScriptRoot 'runs\plate-service\config.json'),
    [ValidateSet('auto', 'cpu')][string]$Device = 'auto',
    [string]$BindAddress = '127.0.0.1',
    [int]$Port = 1788
)
$ErrorActionPreference = 'Stop'
if (-not $Python) { $Python = Join-Path $PSScriptRoot '.venv-service\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Service Python missing. Run setup_api_1788.ps1 or set PLATEAI_SERVICE_PYTHON.' }
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { throw 'Explicit model manifest missing. See docs/plate-service-api.md.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONPATH = (Join-Path $PSScriptRoot 'src')
& $Python -m plateai_service.cli --config $Config --device $Device --host $BindAddress --port $Port
exit $LASTEXITCODE
