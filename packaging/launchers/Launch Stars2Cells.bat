@echo off
setlocal
:: ============================================================
::  Optional launcher for the PACKAGED Windows build.
::
::  Ships inside the Stars2Cells folder next to Stars2Cells.exe.
::  Double-clicking Stars2Cells.exe works on its own; use this
::  instead if you want the app to relaunch automatically after
::  a crash (same behavior as the old launch_s2c.bat), with the
::  exit code shown in a console window.
:: ============================================================

set MAX_RESTARTS=5
set RESTARTS=0

cd /d "%~dp0"

if not exist "Stars2Cells.exe" (
    echo ERROR: Stars2Cells.exe not found next to this launcher.
    pause
    exit /b 1
)

:LAUNCH
echo.
echo ============================================
if %RESTARTS%==0 (
    echo   Stars2Cells Launcher
) else (
    echo   Stars2Cells Restarting [attempt %RESTARTS%/%MAX_RESTARTS%]
)
echo ============================================
echo.

"Stars2Cells.exe"
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE%==0 (
    echo.
    echo   Stars2Cells exited cleanly.
    goto END
)

set /a RESTARTS+=1
if %RESTARTS% GTR %MAX_RESTARTS% (
    echo.
    echo   Max restarts [%MAX_RESTARTS%] reached. Giving up.
    echo   Last exit code: %EXIT_CODE%
    goto END
)

echo.
echo   Crashed with exit code %EXIT_CODE%. Restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto LAUNCH

:END
echo.
pause
