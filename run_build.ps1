[CmdletBinding()]
param(
    [switch]$BootstrapOnly,
    [switch]$SkipTests,
    [switch]$SkipPackage,
    [switch]$RecreateVenv,
    [ValidateSet('auto', 'cu118', 'cu128', 'cpu', 'none')]
    [string]$Cuda = 'auto'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'
$TrainingLockPath = Join-Path $ProjectRoot 'requirements\py311.training.lock.txt'
$DistributionPath = Join-Path $ProjectRoot 'dist'

$TorchCu118Version = 'torch==2.7.1+cu118'
$TorchCu118Index = 'https://download.pytorch.org/whl/cu118'
$TorchCu128Version = 'torch==2.11.0+cu128'
$TorchCu128Index = 'https://download.pytorch.org/whl/cu128'

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Find-CommandPath {
    param([Parameter(Mandatory = $true)][string[]]$Names)

    foreach ($Name in $Names) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($null -ne $Command) {
            return $Command.Source
        }
    }

    return $null
}

function Resolve-CudaTarget {
    param([string]$Requested)

    if ($Requested -ne 'auto') {
        return $Requested
    }

    $NvidiaSmi = Find-CommandPath -Names @('nvidia-smi.exe', 'nvidia-smi')
    if ($null -eq $NvidiaSmi) {
        Write-Host 'No nvidia-smi found. Falling back to CPU PyTorch.'
        return 'cpu'
    }

    try {
        $GpuInfo = & $NvidiaSmi --query-gpu=name --format=csv,noheader 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($GpuInfo)) {
            Write-Host 'Unable to query GPU with nvidia-smi. Falling back to CPU PyTorch.'
            return 'cpu'
        }

        Write-Host "Detected GPU: $($GpuInfo.Trim())"
        # Check for RTX 50-series (5060, 5070, 5080, 5090) or Blackwell
        if ($GpuInfo -match '(RTX\s*50[0-9]{2}|Blackwell)') {
            Write-Host 'Selected CUDA 12.8 (cu128) build for NVIDIA RTX 50-series / Blackwell architecture.'
            return 'cu128'
        }
        # Check for GTX 1080 / Pascal architecture
        if ($GpuInfo -match '(GTX\s*1080|GTX\s*10[0-9]{2}|Pascal)') {
            Write-Host 'Selected CUDA 11.8 (cu118) build for NVIDIA GTX 10-series / Pascal architecture.'
            return 'cu118'
        }

        # For newer GPUs (RTX 40, 30, etc.), cu128 is compatible with modern drivers
        Write-Host 'Auto-detected NVIDIA GPU; defaulting to cu128 for modern architectures.'
        return 'cu128'
    }
    catch {
        Write-Host "Exception querying GPU: $_. Falling back to CPU PyTorch."
        return 'cpu'
    }
}

function Install-PyTorchForGpu {
    param([string]$Target)

    if ($Target -eq 'none' -or $Target -eq 'cpu') {
        Write-Host "Keeping CPU PyTorch ($Target mode)."
        return
    }

    $ExpectedVersion = if ($Target -eq 'cu118') { $TorchCu118Version } else { $TorchCu128Version }
    $ExpectedIndex = if ($Target -eq 'cu118') { $TorchCu118Index } else { $TorchCu128Index }

    $CurrentTorch = (& $PythonPath -c "import torch; print(torch.__version__)").Trim()
    $ExpectedVerOnly = $ExpectedVersion -replace '^torch==', ''

    if ($CurrentTorch -eq $ExpectedVerOnly) {
        Write-Host "PyTorch $CurrentTorch already matches target $Target. Skipping re-download."
    }
    else {
        Write-Host "Installing PyTorch for $Target ($ExpectedVersion)..."
        Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', $ExpectedVersion, '--extra-index-url', $ExpectedIndex)
    }

    # Verify installation and CUDA availability
    $CudaCheck = (& $PythonPath -c "import torch; print(f'Torch: {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')").Trim()
    Write-Host "PyTorch runtime: $CudaCheck"
}

function New-ProjectVenv {
    if ($RecreateVenv -and (Test-Path -LiteralPath $VenvPath)) {
        Write-Host "Removing the explicitly requested virtual environment: $VenvPath"
        Remove-Item -LiteralPath $VenvPath -Recurse -Force
    }

    if (-not (Test-Path -LiteralPath $PythonPath)) {
        $UvPath = Find-CommandPath -Names @('uv.exe', 'uv')
        if ($null -ne $UvPath) {
            Write-Host 'Creating .venv with uv and CPython 3.11...'
            Invoke-Checked -FilePath $UvPath -ArgumentList @('venv', '--seed', '--python', '3.11', $VenvPath)
        }
        else {
            $PyLauncherPath = Find-CommandPath -Names @('py.exe', 'py')
            if ($null -eq $PyLauncherPath) {
                throw 'CPython 3.11 is required. Install uv (recommended) or CPython 3.11 with the py launcher, then rerun build.ps1.'
            }

            Write-Host 'Creating .venv with the CPython 3.11 launcher...'
            Invoke-Checked -FilePath $PyLauncherPath -ArgumentList @('-3.11', '-m', 'venv', $VenvPath)
        }
    }

    $PythonVersion = (& $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query the virtual environment interpreter: $PythonPath"
    }
    if ($PythonVersion -ne '3.11') {
        throw "The existing .venv uses Python $PythonVersion; this project requires CPython 3.11. Rerun with -RecreateVenv to replace only .venv."
    }

    & $PythonPath -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $UvPath = Find-CommandPath -Names @('uv.exe', 'uv')
        if ($null -eq $UvPath) {
            throw 'The existing .venv has no pip. Install uv or rerun with -RecreateVenv after installing CPython 3.11.'
        }

        Write-Host 'Seeding pip into the existing virtual environment...'
        Invoke-Checked -FilePath $UvPath -ArgumentList @('pip', 'install', '--python', $PythonPath, 'pip')
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    New-ProjectVenv

    Write-Host 'Installing the locked training and test dependencies...'
    Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', '--upgrade', 'pip')

    $CudaTarget = Resolve-CudaTarget -Requested $Cuda

    # If a GPU build is targeted and currently installed, prevent pip install -r lockfile from downgrading torch to CPU
    $CurrentTorch = ""
    try {
        $CurrentTorch = (& $PythonPath -c "import torch; print(torch.__version__)" 2>$null).Trim()
    }
    catch {
        $CurrentTorch = ""
    }
    if ($CudaTarget -in @('cu118', 'cu128') -and ($CurrentTorch -like '*+cu*')) {
        Write-Host "Preserving current GPU PyTorch ($CurrentTorch)..."
        # Filter out torch from lockfile lines to prevent downgrade
        $NonTorchDeps = Get-Content -LiteralPath $TrainingLockPath | Where-Object { $_ -notmatch '^torch==' }
        $TempLock = [System.IO.Path]::GetTempFileName()
        try {
            $NonTorchDeps | Set-Content -LiteralPath $TempLock -Encoding utf8
            Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', '-r', $TempLock)
        }
        finally {
            if (Test-Path -LiteralPath $TempLock) {
                Remove-Item -LiteralPath $TempLock -Force
            }
        }
    }
    else {
        Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', '-r', $TrainingLockPath)
    }

    Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', '--no-deps', '-e', '.[test,training]')

    Install-PyTorchForGpu -Target $CudaTarget

    if ($BootstrapOnly) {
        Write-Host 'Bootstrap completed. Run .\run_build.ps1 for the full test and package gate.'
        return
    }

    if (-not $SkipTests) {
        Write-Host 'Running the full test suite...'
        Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pytest', '-q')
    }

    if (-not $SkipPackage) {
        Write-Host 'Building source and wheel distributions...'
        $BuildStartedAt = [DateTime]::UtcNow
        New-Item -ItemType Directory -Path $DistributionPath -Force | Out-Null
        Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'build', '--outdir', $DistributionPath)

        $Wheel = Get-ChildItem -LiteralPath $DistributionPath -Filter '*.whl' -File |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -eq $Wheel -or $Wheel.LastWriteTimeUtc -lt $BuildStartedAt.AddSeconds(-2)) {
            throw 'The package build did not produce a fresh wheel in dist.'
        }

        Write-Host "Installing built wheel: $($Wheel.Name)"
        Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'install', '--force-reinstall', '--no-deps', $Wheel.FullName)

        $ReaderCommand = Join-Path $VenvPath 'Scripts\plateai-read.exe'
        if (-not (Test-Path -LiteralPath $ReaderCommand)) {
            throw 'The built wheel did not install plateai-read.exe.'
        }
        Write-Host 'Checking the installed Reader command...'
        Invoke-Checked -FilePath $ReaderCommand -ArgumentList @('--help')

        $SmokeRoot = Join-Path ([System.IO.Path]::GetTempPath()) "3wa-plate-ai-build-$PID"
        try {
            Write-Host 'Running installed-package smoke generation...'
            Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'plateai_trainer.synthetic', 'generate', '--count', '3', '--seed', '42', '--output', $SmokeRoot)
        }
        finally {
            if (Test-Path -LiteralPath $SmokeRoot) {
                Remove-Item -LiteralPath $SmokeRoot -Recurse -Force
            }
        }
    }

    Write-Host 'Checking installed dependency consistency...'
    Invoke-Checked -FilePath $PythonPath -ArgumentList @('-m', 'pip', 'check')
    Write-Host 'Build gate completed successfully.'
}
finally {
    Pop-Location
}
