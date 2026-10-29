@echo off
cd /d "%~dp0"
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0tools\visualiza_vetores_ahrs.py" --port COM4
