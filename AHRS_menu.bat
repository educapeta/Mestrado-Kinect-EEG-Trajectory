@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
set "VIS=%~dp0tools\visualiza_vetores_ahrs.py"
:menu
cls
echo ============================================================
echo  TESTE AHRS 2x MPU6050 -- vetores 3D (item 142)
echo ============================================================
echo  1 - Ver vetores no CONSOLE (modo texto, sem janela)
echo  2 - Abrir JANELA 3D (OpenCV nativo)
echo  3 - Monitor serial do firmware de teste
echo  4 - Gravar FIRMWARE DE TESTE (precisa BOOT)
echo  5 - Voltar ao FIRMWARE DAS LUVAS
echo  0 - Sair
echo.
set "OPCAO="
set /p "OPCAO=Escolha uma opcao: "
if "%OPCAO%"=="1" goto texto
if "%OPCAO%"=="2" goto grafico
if "%OPCAO%"=="3" goto monitor
if "%OPCAO%"=="4" goto gravar_teste
if "%OPCAO%"=="5" goto luvas
if "%OPCAO%"=="0" goto fim
goto menu

:texto
cls
"%PY%" "%VIS%" --port COM4 --texto --duracao 30
pause
goto menu

:grafico
cls
"%PY%" "%VIS%" --port COM4
pause
goto menu

:monitor
cls
echo Feche o visualizador antes. Para sair do monitor: Ctrl+]
"%PIO%" device monitor -d firmware\esp32_teste_ahrs --baud 115200
pause
goto menu

:gravar_teste
cls
echo Coloque a placa em DOWNLOAD MODE:
echo   segure BOOT, aperte/solte EN, solte BOOT.
pause
"%PIO%" run -d firmware\esp32_teste_ahrs -t upload
pause
goto menu

:luvas
cls
"%PIO%" run -d firmware\esp32_duas_imus -t upload
pause
goto menu

:fim
endlocal
exit /b 0
