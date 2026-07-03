@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  Stars2Cells Windows build
::
::  Usage (from the repo root, any Python 3.10-3.12 on PATH):
::      packaging\build_windows.bat [version]
::
::  Produces:
::      dist\Stars2Cells\Stars2Cells.exe        (one-folder app)
::      Stars2Cells_<version>_windows_x64.zip   (ship this)
::
::  No conda required - everything installs into a throwaway venv.
:: ============================================================

set VERSION=%1
if "%VERSION%"=="" set VERSION=1.0.0
set S2C_VERSION=%VERSION%

cd /d "%~dp0\.."

echo.
echo ============================================
echo   Stars2Cells %VERSION% - Windows build
echo ============================================

:: 1) Fresh build venv
if exist build_env rmdir /s /q build_env
python -m venv build_env
if errorlevel 1 (
    echo ERROR: Could not create venv. Is Python 3.10+ on PATH?
    exit /b 1
)
call build_env\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r packaging\requirements-build.txt
if errorlevel 1 (
    echo ERROR: dependency install failed
    exit /b 1
)

:: 2) Icons from S2C_logo.png
python packaging\make_icons.py
if errorlevel 1 exit /b 1

:: 3) Build
pyinstaller --noconfirm --clean packaging\stars2cells.spec
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)

:: 4) Smoke test: exe exists
if not exist "dist\Stars2Cells\Stars2Cells.exe" (
    echo ERROR: dist\Stars2Cells\Stars2Cells.exe not produced
    exit /b 1
)

:: 4b) Ship the optional auto-restart launcher next to the exe
copy /y "packaging\launchers\Launch Stars2Cells.bat" "dist\Stars2Cells\" >nul

:: 5) Zip the one-folder app
set ZIPNAME=Stars2Cells_%VERSION%_windows_x64.zip
if exist "%ZIPNAME%" del "%ZIPNAME%"
powershell -NoProfile -Command "Compress-Archive -Path 'dist\Stars2Cells' -DestinationPath '%ZIPNAME%'"
if errorlevel 1 (
    echo ERROR: zip failed
    exit /b 1
)

echo.
echo ============================================
echo   Done.
echo   App:  dist\Stars2Cells\Stars2Cells.exe
echo   Zip:  %ZIPNAME%
echo ============================================
endlocal
