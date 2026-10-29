@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "tools\visualiza_vetores_ahrs.py" --port COM4 --texto --duracao 30
pause
