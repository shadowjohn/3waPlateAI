param(
    [ValidateSet('cu118', 'cu128', 'cpu')][string]$Profile = 'cu118'
)
$ErrorActionPreference = 'Stop'
$poseEnv = Join-Path $PSScriptRoot '.venv-pose'
$posePython = Join-Path $poseEnv 'Scripts\python.exe'
if (Test-Path -LiteralPath $poseEnv) { throw 'Existing .venv-pose is preserved. Check it or prepare a separately named environment manually.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first. The main training and API environments will not be changed.' }
& uv venv --python 3.11 --seed $poseEnv
if ($LASTEXITCODE -ne 0) { throw 'Pose venv creation failed.' }
& uv pip install --python $posePython torch==2.7.1 torchvision==0.22.1 --index-url "https://download.pytorch.org/whl/$Profile"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed.' }
& uv pip install --python $posePython -r (Join-Path $PSScriptRoot 'requirements\pose-common.txt')
if ($LASTEXITCODE -ne 0) { throw 'Pose dependency install failed.' }
& uv pip check --python $posePython
if ($LASTEXITCODE -ne 0) { throw 'Pose dependency check failed.' }
& $posePython -c "import torch,cv2,ultralytics,pyarrow,jsonschema; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'ultralytics',ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Pose import check failed.' }
Write-Host "Pose environment ready: $posePython"
Write-Host 'Next: docs/pose-training.md. Images and initial weights require an explicit local download.'
Write-Host 'No training started, no active model changed. RTX 50 and CPU-only hosts require target-host validation.'
