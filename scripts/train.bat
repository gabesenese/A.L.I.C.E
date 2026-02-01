@echo off
cd /d %~dp0\..
REM Quick training commands for A.L.I.C.E

if "%1"=="" goto help
if "%1"=="scenarios" goto scenarios
if "%1"=="promote" goto promote
if "%1"=="nightly" goto nightly
if "%1"=="help" goto help

goto help

:scenarios
echo.
echo Running scenario simulations...
python -m scenarios.sim.run_scenarios --policy minimal
goto end

:promote
echo.
echo Promoting patterns from logs...
python -m ai.promote_patterns
goto end

:nightly
echo.
echo Running complete nightly training pipeline...
python scripts/nightly_training.py
goto end

:help
echo.
echo A.L.I.C.E Training Commands
echo ===========================
echo.
echo Usage: train.bat [command]
echo.
echo Commands:
echo   scenarios    Run scenario simulations
echo   promote      Promote patterns from logs
echo   nightly      Run complete nightly training pipeline
echo   help         Show this help message
echo.
echo Examples:
echo   train.bat scenarios
echo   train.bat promote
echo   train.bat nightly
echo.
goto end

:end
if "%1"=="" pause
