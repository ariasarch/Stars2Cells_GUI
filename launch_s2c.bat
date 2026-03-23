@echo off
setlocal enabledelayedexpansion

set SCRIPT=stars2cells.py
set MAX_RESTARTS=5
set RESTARTS=0

:: Locate conda
set CONDA_PATH=%USERPROFILE%\anaconda3
if not exist "%CONDA_PATH%" set CONDA_PATH=%USERPROFILE%\miniconda3
if not exist "%CONDA_PATH%" (
    echo ERROR: Could not find anaconda3 or miniconda3 in %USERPROFILE%
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

:: Reactivate s2c env
call "%CONDA_PATH%\Scripts\activate.bat" s2c
if errorlevel 1 (
    echo   WARNING: Could not activate s2c env, trying anyway...
) else (
    echo   Conda env: s2c [OK]
)

echo   Launching: python %SCRIPT%
echo.

python %SCRIPT%
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