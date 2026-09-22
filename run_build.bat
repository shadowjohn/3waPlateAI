@echo off
setlocal
rem 3waPlateAI build wrapper
rem Usage:
rem   run_build.bat                      (default: auto-detect GPU -> cu128 for 5060/5090, cu118 for 1080, cpu otherwise)
rem   run_build.bat -Cuda cu118          (explicitly install PyTorch cu118 for GTX 1080)
rem   run_build.bat -Cuda cu128          (explicitly install PyTorch cu128 for RTX 5060/5090)
rem   run_build.bat -Cuda cpu            (keep standard CPU PyTorch)
rem   run_build.bat -BootstrapOnly       (only set up venv & dependencies without running tests/packaging)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_build.ps1" %*
exit /b %ERRORLEVEL%
