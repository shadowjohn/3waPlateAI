@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_api_1788.ps1" %*
exit /b %ERRORLEVEL%
