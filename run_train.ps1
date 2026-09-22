[CmdletBinding()]
param(
    [int]$Epochs = 10,
    [int]$BatchSize = 32,
    [double]$LearningRate = 1e-3,
    [int]$Seed = 42,
    [string]$Device = 'auto',
    [string]$TrainDir = '',
    [string]$ValDir = '',
    [string]$RunName = '',
    [string]$OutputBundle = '',
    [string]$Font = 'taiwan_plate',
    [string]$Charset = '',
    [string]$Rules = '',
    [int]$SynthTrainCount = 2000,
    [int]$SynthValCount = 500,
    [switch]$SkipExport
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$PythonPath = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "[ERROR] Python environment not found: $PythonPath" -ForegroundColor Red
    Write-Host "Please run .\run_build.bat first to build or bootstrap the virtual environment." -ForegroundColor Yellow
    exit 1
}

# Resolve configurations
if (-not $Charset) {
    $Charset = Join-Path $ProjectRoot 'configs\charsets\tw_standard_v1.txt'
}
if (-not $Rules) {
    $Rules = Join-Path $ProjectRoot 'configs\plate_rules\tw_standard_v1.json'
}

# Resolve run directory
if (-not $RunName) {
    $RunName = "run-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}
$RunDir = Join-Path $ProjectRoot ("runs\" + $RunName)

# Resolve output bundle
if (-not $OutputBundle) {
    $OutputBundle = Join-Path $ProjectRoot 'models\bundles\active-v1'
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " 3waPlateAI - PyTorch CTC Model Training & Export Workflow" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Run name:        $RunName"
Write-Host "Output run dir:  $RunDir"
Write-Host "Epochs:          $Epochs"
Write-Host "Batch size:      $BatchSize"
Write-Host "Learning rate:   $LearningRate"
Write-Host "Device:          $Device"
Write-Host "Charset:         $Charset"
Write-Host "Rules:           $Rules"

$PlateAiGenerate = Join-Path $VenvPath 'Scripts\plateai-generate.exe'
$PlateAiTrain = Join-Path $VenvPath 'Scripts\plateai-train.exe'
$PlateAiExport = Join-Path $VenvPath 'Scripts\plateai-export.exe'

# 1. Check or auto-generate training data
if (-not $TrainDir -or -not (Test-Path -LiteralPath $TrainDir)) {
    $DefaultTrain = Join-Path $ProjectRoot 'out\train-std'
    if (Test-Path -LiteralPath $DefaultTrain) {
        $TrainDir = $DefaultTrain
        Write-Host "Using existing training dataset: $TrainDir" -ForegroundColor Gray
    } else {
        $ExistingTrains = Get-ChildItem (Join-Path $ProjectRoot 'out') -Directory -Filter '*train*' -ErrorAction SilentlyContinue
        if ($ExistingTrains -and $ExistingTrains.Count -gt 0) {
            $TrainDir = $ExistingTrains[0].FullName
            Write-Host "Using existing training dataset: $TrainDir" -ForegroundColor Gray
        } else {
            $TrainDir = $DefaultTrain
            Write-Host "No training dataset found. Automatically generating $SynthTrainCount synthetic samples..." -ForegroundColor Yellow
            if (Test-Path -LiteralPath $PlateAiGenerate) {
                & $PlateAiGenerate generate `
                    --count $SynthTrainCount `
                    --seed 42 `
                    --font $Font `
                    --charset $Charset `
                    --rules $Rules `
                    --output $TrainDir
            } else {
                & $PythonPath -m plateai_trainer.synthetic.cli generate `
                    --count $SynthTrainCount `
                    --seed 42 `
                    --font $Font `
                    --charset $Charset `
                    --rules $Rules `
                    --output $TrainDir
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Dataset synthesis failed." -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
    }
}

# 2. Check or auto-generate validation data
if (-not $ValDir -or -not (Test-Path -LiteralPath $ValDir)) {
    $DefaultVal = Join-Path $ProjectRoot 'out\val-std'
    if (Test-Path -LiteralPath $DefaultVal) {
        $ValDir = $DefaultVal
        Write-Host "Using existing validation dataset: $ValDir" -ForegroundColor Gray
    } else {
        $ExistingVals = Get-ChildItem (Join-Path $ProjectRoot 'out') -Directory -Filter '*val*' -ErrorAction SilentlyContinue
        if ($ExistingVals -and $ExistingVals.Count -gt 0) {
            $ValDir = $ExistingVals[0].FullName
            Write-Host "Using existing validation dataset: $ValDir" -ForegroundColor Gray
        } else {
            $ValDir = $DefaultVal
            Write-Host "No validation dataset found. Automatically generating $SynthValCount synthetic samples..." -ForegroundColor Yellow
            if (Test-Path -LiteralPath $PlateAiGenerate) {
                & $PlateAiGenerate generate `
                    --count $SynthValCount `
                    --seed 43 `
                    --font $Font `
                    --charset $Charset `
                    --rules $Rules `
                    --output $ValDir
            } else {
                & $PythonPath -m plateai_trainer.synthetic.cli generate `
                    --count $SynthValCount `
                    --seed 43 `
                    --font $Font `
                    --charset $Charset `
                    --rules $Rules `
                    --output $ValDir
            }
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[ERROR] Validation dataset synthesis failed." -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
    }
}

Write-Host "Training dataset:   $TrainDir" -ForegroundColor Gray
Write-Host "Validation dataset: $ValDir" -ForegroundColor Gray
Write-Host ""

# 3. Execute PyTorch CTC training
Write-Host ">>> Starting PyTorch CTC Training..." -ForegroundColor Green
if (Test-Path -LiteralPath $PlateAiTrain) {
    & $PlateAiTrain `
        --train $TrainDir `
        --validation $ValDir `
        --output $RunDir `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --learning-rate $LearningRate `
        --seed $Seed `
        --device $Device `
        --charset $Charset `
        --rules $Rules
} else {
    & $PythonPath -m plateai_trainer.training.cli `
        --train $TrainDir `
        --validation $ValDir `
        --output $RunDir `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --learning-rate $LearningRate `
        --seed $Seed `
        --device $Device `
        --charset $Charset `
        --rules $Rules
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

$BestCheckpoint = Join-Path $RunDir 'best.pt'
$ReportJson = Join-Path $RunDir 'report.json'

if (-not (Test-Path -LiteralPath $BestCheckpoint)) {
    Write-Host "[ERROR] Training finished but best.pt checkpoint was not found at $BestCheckpoint" -ForegroundColor Red
    exit 1
}

Write-Host ">>> Training completed successfully!" -ForegroundColor Green
Write-Host "Best checkpoint: $BestCheckpoint" -ForegroundColor Gray
Write-Host "Training report: $ReportJson" -ForegroundColor Gray
Write-Host ""

# 4. Export to ONNX bundle
if (-not $SkipExport) {
    Write-Host ">>> Exporting ONNX Model Bundle..." -ForegroundColor Green
    if (Test-Path -LiteralPath $OutputBundle) {
        Write-Host "Replacing existing bundle at $OutputBundle..." -ForegroundColor Cyan
        Remove-Item -LiteralPath $OutputBundle -Recurse -Force
    }

    if (Test-Path -LiteralPath $PlateAiExport) {
        & $PlateAiExport `
            --checkpoint $BestCheckpoint `
            --report $ReportJson `
            --output $OutputBundle `
            --charset $Charset `
            --rules $Rules
    } else {
        & $PythonPath -m plateai_trainer.export.cli `
            --checkpoint $BestCheckpoint `
            --report $ReportJson `
            --output $OutputBundle `
            --charset $Charset `
            --rules $Rules
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Model bundle export failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    Write-Host ">>> ONNX Model Bundle exported to: $OutputBundle" -ForegroundColor Green
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " All Done! To test recognition or launch Web Studio:" -ForegroundColor Green
Write-Host "   .\run_server.bat" -ForegroundColor White
Write-Host "   .\.venv\Scripts\plateai-read --bundle models\bundles\active-v1 <image.jpg>" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
