param(
    [ValidateSet('cu118', 'cu128', 'cpu')][string]$Profile = 'cu118'
)
$ErrorActionPreference = 'Stop'
$serviceEnv = Join-Path $PSScriptRoot '.venv-service'
$servicePython = Join-Path $serviceEnv 'Scripts\python.exe'
if (Test-Path -LiteralPath $serviceEnv) { throw 'Existing .venv-service is preserved. Use it, or create a separately named environment manually.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first; the script will not modify the main training environment.' }
& uv venv --python 3.11 --seed $serviceEnv
if ($LASTEXITCODE -ne 0) { throw 'Service venv creation failed.' }
$torchIndex = "https://download.pytorch.org/whl/$Profile"
& uv pip install --python $servicePython torch==2.7.1 torchvision==0.22.1 --index-url $torchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed.' }
if ($Profile -eq 'cu118') {
    & uv pip install --python $servicePython paddlepaddle-gpu==3.2.2 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu118/
} else {
    # Paddle Blackwell GPU is not qualified here. Preserve working Torch GPU + Paddle CPU.
    & uv pip install --python $servicePython paddlepaddle==3.2.2
}
if ($LASTEXITCODE -ne 0) { throw 'Paddle install failed.' }
if ($Profile -eq 'cu118') {
    & uv pip install --python $servicePython -r (Join-Path $PSScriptRoot 'requirements\service-common.txt') -c (Join-Path $PSScriptRoot 'requirements\service-cu118-observed.txt')
} else {
    & uv pip install --python $servicePython -r (Join-Path $PSScriptRoot 'requirements\service-common.txt')
}
if ($LASTEXITCODE -ne 0) { throw 'Service dependency install failed.' }
& uv pip check --python $servicePython
if ($LASTEXITCODE -ne 0) { throw 'Service dependency check failed.' }
Write-Host "Environment ready: $servicePython"
Write-Host 'Configure explicitly sourced model paths and hashes before running run_api_1788.bat.'
Write-Host 'Only cu118/GTX 1080 has local inference evidence; CPU-only and RTX 50 installation require target-host verification.'
