@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "FONT_PY=%CD%\.venv-font-build\Scripts\python.exe"
set "OUTPUT_FONT=%CD%\assets\local\fonts\TaiwanPlate-Regular.ttf"

if not exist "%FONT_PY%" (
    echo [ERROR] Local font-build environment is missing.
    echo Run build_prepare_ttf_env.bat first.
    exit /b 1
)

if not exist "%CD%\assets\spec_images\page_0.jpg" (
    echo [ERROR] Required local reference images are missing: assets\spec_images\page_0.jpg
    exit /b 1
)

"%FONT_PY%" tools\build_plate_font.py
if errorlevel 1 exit /b 1

if not exist "%OUTPUT_FONT%" (
    echo [ERROR] Font build completed without producing the expected local artifact.
    exit /b 1
)

echo.
echo Built local-only font: %OUTPUT_FONT%
echo This artifact is ignored by Git and excluded from release packages.
exit /b 0
