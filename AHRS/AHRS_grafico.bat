@echo off
setlocal
set "AHRS=%~dp0"
for %%I in ("%~dp0..") do set "RAIZ=%%~fI"
cd /d "%RAIZ%"
"%RAIZ%\.venv\Scripts\python.exe" "%AHRS%tools\visualiza_vetores_ahrs.py" --port COM4 --opencv
pause
endlocal
