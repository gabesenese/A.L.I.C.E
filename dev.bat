@echo off
REM A.L.I.C.E Development Environment
REM This script activates the virtual environment and keeps CMD open

cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo.
echo ========================================
echo   A.L.I.C.E Development Environment
echo   Virtual Environment: ACTIVATED
echo ========================================
echo.
echo Quick Commands:
echo   make run                    - Run A.L.I.C.E API
echo   python -m uvicorn app.main:app --reload --app-dir "%~dp0" - Run API directly
echo   python -m pytest tests/ -v  - Run all tests
echo   ruff check .                - Check code quality
echo.

where make >nul 2>nul
if %errorlevel%==0 (
	make run
) else (
	echo [WARN] make not found. Falling back to uvicorn.
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --app-dir "%~dp0"
)

cmd /k
