@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "VENV_DIR=%CD%\.venv-font-build"
set "BOOTSTRAP_PY="
set "BOOTSTRAP_ARGS="
set "PROJECT_PY=%CD%\.venv\Scripts\python.exe"

if exist "%PROJECT_PY%" (
    "%PROJECT_PY%" -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))" >nul 2>nul
    if not errorlevel 1 set "BOOTSTRAP_PY=%PROJECT_PY%"
)

if not defined BOOTSTRAP_PY (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.11 -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))" >nul 2>nul
        if not errorlevel 1 (
            set "BOOTSTRAP_PY=py.exe"
            set "BOOTSTRAP_ARGS=-3.11"
        )
    )
)

if not defined BOOTSTRAP_PY (
    where python >nul 2>nul
    if not errorlevel 1 (
        python -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))" >nul 2>nul
        if not errorlevel 1 set "BOOTSTRAP_PY=python.exe"
    )
)

if not defined BOOTSTRAP_PY (
    echo [ERROR] Python 3.11 is required. Install it, then run this file again.
    exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Creating isolated font-build environment...
    "%BOOTSTRAP_PY%" %BOOTSTRAP_ARGS% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

echo Installing local font-build dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_DIR%\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade "fonttools>=4.0" "opencv-python-headless>=5.0.0.93,<5.1" "Pillow>=12.3.0,<13"
if errorlevel 1 exit /b 1

echo.
echo Ready: %VENV_DIR%
echo Next: build_taiwanplate_ttf.bat
exit /b 0
