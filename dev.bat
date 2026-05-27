@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if not defined ALICE_OLLAMA_MODEL set "ALICE_OLLAMA_MODEL=alice_ollama"
python dev_console.py %*
