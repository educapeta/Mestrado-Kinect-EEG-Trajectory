@echo off
setlocal
set "AHRS=%~dp0"
for %%I in ("%~dp0..") do set "RAIZ=%%~fI"
cd /d "%RAIZ%"
set "PY=%RAIZ%\.venv\Scripts\python.exe"
set "PYW=%RAIZ%\.venv\Scripts\pythonw.exe"
set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
set "VIS=%AHRS%tools\visualiza_vetores_ahrs.py"
set "FW_TESTE=%AHRS%firmware\esp32_teste_ahrs"
set "FW_LUVAS=%RAIZ%\firmware\esp32_duas_imus"
:menu
cls
echo ============================================================
echo  TESTE AHRS 2x MPU6050 -- vetores 3D (pasta AHRS)
echo ============================================================
echo  1 - Ver vetores no CONSOLE (modo texto, sem janela)
echo  2 - Abrir JANELA 3D (OpenCV nativo)
echo  3 - Monitor serial do firmware de teste
echo  4 - Gravar FIRMWARE DE TESTE (precisa BOOT)
echo  5 - Voltar ao FIRMWARE DAS LUVAS
echo  6 - Abrir JANELA 3D SEM CONSOLE
echo  7 - JANELA 3D GRAVANDO LOG para analise
echo  0 - Sair
echo.
set "OPCAO="
set /p "OPCAO=Escolha uma opcao: "
if "%OPCAO%"=="1" goto texto
if "%OPCAO%"=="2" goto grafico
if "%OPCAO%"=="3" goto monitor
if "%OPCAO%"=="4" goto gravar_teste
if "%OPCAO%"=="5" goto luvas
if "%OPCAO%"=="6" goto grafico_sem_console
if "%OPCAO%"=="7" goto grafico_log
if "%OPCAO%"=="0" goto fim
goto menu

:texto
cls
"%PY%" "%VIS%" --port COM4 --texto --duracao 30
pause
goto menu

:grafico
cls
"%PY%" "%VIS%" --port COM4 --opencv
pause
goto menu

:monitor
cls
echo Feche o visualizador antes. Para sair do monitor: Ctrl+]
"%PIO%" device monitor -d "%FW_TESTE%" --baud 115200
pause
goto menu

:gravar_teste
cls
echo Coloque a placa em DOWNLOAD MODE:
echo   segure BOOT, aperte/solte EN, solte BOOT.
pause
"%PIO%" run -d "%FW_TESTE%" -t upload
pause
goto menu

:luvas
cls
"%PIO%" run -d "%FW_LUVAS%" -t upload
pause
goto menu

:grafico_sem_console
cls
start "" "%PYW%" "%VIS%" --port COM4 --opencv
echo Janela 3D aberta SEM console (se nao aparecer, veja atras do PowerShell).
timeout /t 3 >nul
goto menu

:grafico_log
cls
echo A janela 3D vai gravar em "%AHRS%logs\ahrs_YYYYmmdd_HHMMSS.csv"
echo Mexa nos sensores e volte a posicao inicial; feche com Q/ESC.
"%PY%" "%VIS%" --port COM4 --opencv --log auto
pause
goto menu

:fim
endlocal
exit /b 0
