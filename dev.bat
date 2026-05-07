@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM A.L.I.C.E Development Environment
REM Interactive command menu with the virtual environment activated

cd /d "%~dp0"
call .venv\Scripts\activate.bat

where make >nul 2>nul
if %errorlevel%==0 (
	set "USE_MAKE=1"
	set "RUN_CMD=python app\dev.py"
	set "DEBUG_MODE_CMD=python -m app.alice --model llama3.1:8b --debug"
	set "TEST_CMD=python -m pytest tests/unit tests/integration tests/e2e tests/golden -v"
	set "LINT_CMD=make lint"
	set "FIX_CMD=ruff check . --fix --unsafe-fixes --exit-zero"
	set "FORMAT_CMD=make format"
) else (
	set "USE_MAKE=0"
	set "RUN_CMD=python app\dev.py"
	set "DEBUG_MODE_CMD=python -m app.alice --model llama3.1:8b --debug"
	set "TEST_CMD=python -m pytest tests/unit tests/integration tests/e2e tests/golden -v"
	set "LINT_CMD=ruff check ."
	set "FIX_CMD=ruff check . --fix --unsafe-fixes --exit-zero"
	set "FORMAT_CMD=ruff format ."
)

:menu
cls
echo.
echo ========================================
echo   A.L.I.C.E Development Environment
echo   Virtual Environment: ACTIVATED
echo ========================================
echo.
echo Choose a command:
echo   1  Run app
echo   2  Run tests
echo   3  Lint
echo   4  Collect tests (pytest --collect-only)
echo   5  Full check
echo   6  Format
echo   7  Open plain shell
echo   8  Debug mode
echo   9  Exit
echo.
choice /c 123456789 /n /m "Select an option: "

if errorlevel 9 goto exit
if errorlevel 8 goto debug_mode
if errorlevel 7 goto plain_shell
if errorlevel 6 goto format
if errorlevel 5 goto check
if errorlevel 4 goto collect
if errorlevel 3 goto lint
if errorlevel 2 goto test
if errorlevel 1 goto run
goto menu

:run
echo.
echo Running: %RUN_CMD%
call %RUN_CMD%
pause
goto menu

:test
echo.
echo Running: %TEST_CMD%
call %TEST_CMD%
call :wait_after_tests
goto menu

:lint
echo.
echo Running: %LINT_CMD%
call %LINT_CMD%
pause
goto menu

:collect
echo.
echo Running: python -m pytest --collect-only -q
call python -m pytest --collect-only -q
call :wait_after_tests
goto menu

:check
echo.
echo Running: %LINT_CMD%
call %LINT_CMD%
echo.
echo Running: %TEST_CMD%
call %TEST_CMD%
call :wait_after_tests
goto menu

:format
echo.
echo Running: %FORMAT_CMD%
call %FORMAT_CMD%
pause
goto menu

:plain_shell
echo.
echo Opening an interactive shell in the activated environment.
cmd /k
goto menu

:debug_mode
echo.
echo Running: %DEBUG_MODE_CMD%
call %DEBUG_MODE_CMD%
pause
goto menu

:exit
endlocal
exit /b 0

:wait_after_tests
echo.
set "CONTINUE_INPUT="
set /p "CONTINUE_INPUT=Press Enter (or type Y) to return to the menu: "
exit /b 0
