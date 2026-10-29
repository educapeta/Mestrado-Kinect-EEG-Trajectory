@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "tools\visualiza_vetores_ahrs.py" --port COM4 --opencv --log auto
echo.
echo O CSV esta' em ahrs_logs\ahrs_*.csv
pause
