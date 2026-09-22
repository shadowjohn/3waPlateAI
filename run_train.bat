@echo off
setlocal
rem 3waPlateAI PyTorch CTC Training & Export Runner
rem Usage:
rem   run_train.bat                       (train with default settings, auto-detect GPU/fallback CPU)
rem   run_train.bat -Epochs 20            (train for 20 epochs)
rem   run_train.bat -Device cpu           (force CPU training)
rem   run_train.bat -Device cuda          (force CUDA GPU training)
rem   run_train.bat -BatchSize 64         (batch size 64)
rem   run_train.bat -TrainDir path\train -ValDir path\val
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_train.ps1" %*
exit /b %ERRORLEVEL%
