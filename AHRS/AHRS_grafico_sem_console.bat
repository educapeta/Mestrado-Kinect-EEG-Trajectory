@echo off
setlocal
set "AHRS=%~dp0"
for %%I in ("%~dp0..") do set "RAIZ=%%~fI"
cd /d "%RAIZ%"
start "" "%RAIZ%\.venv\Scripts\pythonw.exe" "%AHRS%tools\visualiza_vetores_ahrs.py" --port COM4 --opencv
endlocal
