@echo off
cd /d %~dp0\..
REM A.L.I.C.E Installation Script for Windows
REM Run this script to set up A.L.I.C.E on your system

echo ================================================================================
echo    A.L.I.C.E Installation Script
echo    Advanced Linguistic Intelligence Computer Entity
echo ================================================================================
echo.

REM Check Python installation
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo [OK] Python is installed
echo.

REM Install Python dependencies
echo [2/6] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Download NLTK data
echo [3/6] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('vader_lexicon', quiet=True)"
if errorlevel 1 (
    echo [WARNING] Failed to download NLTK data
    echo You may need to download it manually
) else (
    echo [OK] NLTK data downloaded
)
echo.

REM Check for Ollama
echo [4/6] Checking for Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not installed
    echo.
    echo Please install Ollama:
    echo 1. Download from: https://ollama.ai
    echo 2. Run the installer
    echo 3. Open a new terminal and run: ollama pull llama3.3:70b
    echo.
) else (
    echo [OK] Ollama is installed
    echo.
    
    REM Check if model is available
    echo [5/6] Checking for Llama model...
    ollama list | findstr "llama3.3" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Llama 3.3 70B model not found
        echo.
        set /p DOWNLOAD="Download model now? This may take 15-30 minutes (Y/N): "
        if /i "%DOWNLOAD%"=="Y" (
            echo Downloading model...
            ollama pull llama3.3:70b
            echo [OK] Model downloaded
        ) else (
            echo [SKIP] Model download skipped
            echo You can download later with: ollama pull llama3.3:70b
        )
    ) else (
        echo [OK] Llama model found
    )
)
echo.

REM Run system test
echo [6/6] Running system test...
python test_system.py
echo.

echo ================================================================================
echo    Installation Complete!
echo ================================================================================
echo.
echo Next steps:
echo 1. Make sure Ollama is running: ollama serve
echo 2. Start A.L.I.C.E: python -m app.main --name "Your Name"
echo.
echo Optional:
echo - Enable voice: python -m app.main --voice --name "Your Name"
echo - Voice only: python -m app.main --voice-only --name "Your Name"
echo.
echo Documentation:
echo - Quick Start: QUICK_START.md
echo - Examples: EXAMPLES.md
echo - Full docs: README.md
echo.
echo ================================================================================
pause
