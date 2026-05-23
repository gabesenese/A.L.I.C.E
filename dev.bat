@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
call .venv\Scripts\activate.bat

if not defined ALICE_OLLAMA_MODEL set "ALICE_OLLAMA_MODEL=alice_ollama"
set "MODEL=%ALICE_OLLAMA_MODEL%"

REM Command-line shortcut: dev.bat [1-6] jumps directly to that section
if not "%1"=="" (
    if "%1"=="1" goto run
    if "%1"=="2" goto debug_menu
    if "%1"=="3" goto test_menu
    if "%1"=="4" goto quality_menu
    if "%1"=="5" goto diag_menu
    if "%1"=="6" goto exit
)

:menu
cls
echo.
echo   A.L.I.C.E  Dev Console                 model: %MODEL%
echo   =========================================================
echo.
echo     1   Run              dev mode with auto-reload
echo     2   Debug            all debug launch options
echo     3   Test             run test suites
echo     4   Quality          lint / format / fix
echo     5   Diagnostics      health checks and benchmarks
echo     6   Shell
echo     7   Exit
echo.
choice /c 1234567 /n /m "  Select: "
if errorlevel 7 goto exit
if errorlevel 6 goto shell
if errorlevel 5 goto diag_menu
if errorlevel 4 goto quality_menu
if errorlevel 3 goto test_menu
if errorlevel 2 goto debug_menu
if errorlevel 1 goto run

REM ─────────────────────────────────────────────
REM  1  RUN
REM ─────────────────────────────────────────────
:run
cls
echo.
echo   A.L.I.C.E  Dev Console                 model: %MODEL%
echo   =========================================================
echo.
echo   Starting dev mode (auto-reload on file changes)...
echo   python app\dev.py
echo.
python app\dev.py
pause
goto menu

REM ─────────────────────────────────────────────
REM  2  DEBUG SUBMENU
REM ─────────────────────────────────────────────
:debug_menu
cls
echo.
echo   A.L.I.C.E  Dev Console  ^>  Debug        model: %MODEL%
echo   =========================================================
echo.
echo     1   Standard debug          --debug
echo     2   Full logs               app.main  (raw stdout, no UI)
echo     3   Dev + debug             app.dev --debug  (auto-reload)
echo     4   Custom model + debug    prompt for model name
echo     5   Policy: minimal         --debug --llm-policy minimal
echo     6   Policy: strict          --debug --llm-policy strict
echo     7   Privacy mode            --debug --privacy-mode
echo     8   Voice + debug           --debug --voice
echo     B   Back
echo.
choice /c 12345678B /n /m "  Select: "
if errorlevel 9 goto menu
if errorlevel 8 goto debug_voice
if errorlevel 7 goto debug_privacy
if errorlevel 6 goto debug_strict
if errorlevel 5 goto debug_minimal
if errorlevel 4 goto debug_custom_model
if errorlevel 3 goto debug_dev
if errorlevel 2 goto debug_full_logs
if errorlevel 1 goto debug_standard

:debug_standard
cls
echo.
echo   python -m app.alice --model %MODEL% --debug
echo.
python -m app.alice --model %MODEL% --debug
pause
goto debug_menu

:debug_full_logs
cls
echo.
echo   python -m app.main
echo.
python -m app.main
pause
goto debug_menu

:debug_dev
cls
echo.
echo   python app\dev.py --debug
echo.
python app\dev.py --debug
pause
goto debug_menu

:debug_custom_model
cls
echo.
echo   A.L.I.C.E  Dev Console  ^>  Debug  ^>  Custom Model
echo   =========================================================
echo.
set "CUSTOM_MODEL="
set /p "CUSTOM_MODEL=  Model name (e.g. llama3.3:70b, gemma2:9b) [%MODEL%]: "
if "!CUSTOM_MODEL!"=="" set "CUSTOM_MODEL=%MODEL%"
echo.
echo   python -m app.alice --model !CUSTOM_MODEL! --debug
echo.
python -m app.alice --model "!CUSTOM_MODEL!" --debug
pause
goto debug_menu

:debug_minimal
cls
echo.
echo   python -m app.alice --model %MODEL% --debug --llm-policy minimal
echo.
python -m app.alice --model %MODEL% --debug --llm-policy minimal
pause
goto debug_menu

:debug_strict
cls
echo.
echo   python -m app.alice --model %MODEL% --debug --llm-policy strict
echo.
python -m app.alice --model %MODEL% --debug --llm-policy strict
pause
goto debug_menu

:debug_privacy
cls
echo.
echo   python -m app.alice --model %MODEL% --debug --privacy-mode
echo.
python -m app.alice --model %MODEL% --debug --privacy-mode
pause
goto debug_menu

:debug_voice
cls
echo.
echo   python -m app.alice --model %MODEL% --debug --voice
echo.
python -m app.alice --model %MODEL% --debug --voice
pause
goto debug_menu

REM ─────────────────────────────────────────────
REM  3  TEST SUBMENU
REM ─────────────────────────────────────────────
:test_menu
cls
echo.
echo   A.L.I.C.E  Dev Console  ^>  Test         model: %MODEL%
echo   =========================================================
echo.
echo     1   All suites          unit + integration + e2e + golden
echo     2   Unit only
echo     3   Integration only
echo     4   E2E only
echo     5   Golden only
echo     6   Collect             list all tests (no run)
echo     B   Back
echo.
choice /c 123456B /n /m "  Select: "
if errorlevel 7 goto menu
if errorlevel 6 goto test_collect
if errorlevel 5 goto test_golden
if errorlevel 4 goto test_e2e
if errorlevel 3 goto test_integration
if errorlevel 2 goto test_unit
if errorlevel 1 goto test_all

:test_all
cls
echo.
echo   python -m pytest tests/unit tests/integration tests/e2e tests/golden -v
echo.
python -m pytest tests/unit tests/integration tests/e2e tests/golden -v
call :pause_test
goto test_menu

:test_unit
cls
echo.
echo   python -m pytest tests/unit -v
echo.
python -m pytest tests/unit -v
call :pause_test
goto test_menu

:test_integration
cls
echo.
echo   python -m pytest tests/integration -v
echo.
python -m pytest tests/integration -v
call :pause_test
goto test_menu

:test_e2e
cls
echo.
echo   python -m pytest tests/e2e -v
echo.
python -m pytest tests/e2e -v
call :pause_test
goto test_menu

:test_golden
cls
echo.
echo   python -m pytest tests/golden -v
echo.
python -m pytest tests/golden -v
call :pause_test
goto test_menu

:test_collect
cls
echo.
echo   python -m pytest --collect-only -q
echo.
python -m pytest --collect-only -q
call :pause_test
goto test_menu

REM ─────────────────────────────────────────────
REM  4  QUALITY SUBMENU
REM ─────────────────────────────────────────────
:quality_menu
cls
echo.
echo   A.L.I.C.E  Dev Console  ^>  Quality      model: %MODEL%
echo   =========================================================
echo.
echo     1   Lint                ruff check .
echo     2   Format              ruff format .
echo     3   Fix                 ruff check --fix --unsafe-fixes
echo     4   Full check          lint then all tests
echo     B   Back
echo.
choice /c 1234B /n /m "  Select: "
if errorlevel 5 goto menu
if errorlevel 4 goto quality_fullcheck
if errorlevel 3 goto quality_fix
if errorlevel 2 goto quality_format
if errorlevel 1 goto quality_lint

:quality_lint
cls
echo.
echo   ruff check .
echo.
ruff check .
pause
goto quality_menu

:quality_format
cls
echo.
echo   ruff format .
echo.
ruff format .
pause
goto quality_menu

:quality_fix
cls
echo.
echo   ruff check . --fix --unsafe-fixes --exit-zero
echo.
ruff check . --fix --unsafe-fixes --exit-zero
pause
goto quality_menu

:quality_fullcheck
cls
echo.
echo   ruff check .
echo.
ruff check .
echo.
echo   python -m pytest tests/unit tests/integration tests/e2e tests/golden -v
echo.
python -m pytest tests/unit tests/integration tests/e2e tests/golden -v
call :pause_test
goto quality_menu

REM ─────────────────────────────────────────────
REM  5  DIAGNOSTICS SUBMENU
REM ─────────────────────────────────────────────
:diag_menu
cls
echo.
echo   A.L.I.C.E  Dev Console  ^>  Diagnostics  model: %MODEL%
echo   =========================================================
echo.
echo     1   Startup doctor      tools/auditing/startup_doctor.py
echo     2   Memory health       scripts/check_memory_health.py
echo     3   Core benchmark      tools/auditing/core_benchmark_gate.py run
echo     4   Open logs folder
echo     B   Back
echo.
choice /c 1234B /n /m "  Select: "
if errorlevel 5 goto menu
if errorlevel 4 goto diag_logs
if errorlevel 3 goto diag_benchmark
if errorlevel 2 goto diag_memory
if errorlevel 1 goto diag_startup

:diag_startup
cls
echo.
echo   python tools/auditing/startup_doctor.py
echo.
python tools/auditing/startup_doctor.py
pause
goto diag_menu

:diag_memory
cls
echo.
echo   python scripts/check_memory_health.py
echo.
python scripts/check_memory_health.py
pause
goto diag_menu

:diag_benchmark
cls
echo.
echo   python tools/auditing/core_benchmark_gate.py run
echo.
python tools/auditing/core_benchmark_gate.py run
pause
goto diag_menu

:diag_logs
cls
echo.
echo   Opening logs\ folder...
echo.
if exist logs\ (
    explorer logs\
) else (
    echo   logs\ folder not found.
)
pause
goto diag_menu

REM ─────────────────────────────────────────────
REM  6  SHELL
REM ─────────────────────────────────────────────
:shell
cls
echo.
echo   Opening interactive shell (venv active). Type "exit" to return.
echo.
cmd /k
goto menu

REM ─────────────────────────────────────────────
REM  EXIT
REM ─────────────────────────────────────────────
:exit
endlocal
exit /b 0

REM ─────────────────────────────────────────────
REM  SUBROUTINES
REM ─────────────────────────────────────────────
:pause_test
echo.
set "CONTINUE_INPUT="
set /p "CONTINUE_INPUT=  Press Enter to return to menu: "
exit /b 0
