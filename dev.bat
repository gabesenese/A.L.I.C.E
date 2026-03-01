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
echo   python app/main.py          - Run A.L.I.C.E
echo   python app/main.py --debug  - Run with debug logging
echo   python -m pytest tests/ -v  - Run all tests
echo   python -m black ai/ --check - Check code formatting
echo.

cmd /k
