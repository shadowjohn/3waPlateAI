@echo off
setlocal
rem 3waPlateAI - Fetch Real Taiwan License Plate Test Photos
rem Usage:
rem   run_get_test_data.bat               (fetch default 30 sample photos into test_data\)
rem   run_get_test_data.bat -Count 50     (fetch 50 sample photos)
rem   run_get_test_data.bat -Count all    (fetch all 259 sample photos)
rem   run_get_test_data.bat -Output mydir (save to custom directory)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_get_test_data.ps1" %*
exit /b %ERRORLEVEL%
