@echo off
cd /d "%~dp0"
title 3waPlateAI Web Studio (Port 1688)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" tools\run_server.py %*
) else (
    python tools\run_server.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    pause
)
