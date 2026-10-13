@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ===========================================================================
rem Gravador do esp32_duas_imus (1 ESP32 + 2 MPU6050; UDP 4210/4211).
rem
rem MOTIVO DO RETRY AUTOMATICO (09/10/2026): o auto-reset DESTA placa nao
rem entra em download mode -- o esptool falha com "Wrong boot mode (0x13)"
rem se o BOOT nao estiver pressionado durante o "Connecting........". Cada
rem execucao do esptool da' UMA chance (~12 s), entao este script TENTA DE
rem NOVO sozinho: o usuario so' precisa SEGURAR o BOOT e esperar.
rem
rem Uso:  gravar.bat [COMx]     (padrao COM4)
rem       set GRAVAR_TENTATIVAS=1  antes de rodar = testa UMA tentativa
rem ===========================================================================

set "PORT=%~1"
if not "%PORT%"=="" goto :tem_port
set /p "PORT=  Porta serial (ex: COM4 / COM5) [COM4]: "
if "%PORT%"=="" set "PORT=COM4"

:tem_port

if "%GRAVAR_TENTATIVAS%"=="" set "GRAVAR_TENTATIVAS=10"
set "TENTATIVAS=%GRAVAR_TENTATIVAS%"

echo ============================================================
echo   GRAVAR ESP32 COM 2 MPU6050    (UDP 4210=direita 4211=esquerda)
echo   Porta serial: %PORT%   Tentativas: %TENTATIVAS% (auto-retry)
echo   Wi-Fi do firmware: hotspot EDUARDO-PC 9049 -^> unicast 192.168.137.1
echo.
echo   COMO GRAVAR (NAO use o botao EN/RST):
echo     1) TECLE QUALQUER TECLA abaixo para comecar.
echo     2) SEGURE o BOOT quando aparecer "Connecting........"
echo        (segurar desde ja' tambem e' valido -- e' o mais seguro).
echo     3) Continue segurando ate' "Writing at 0x000...".
echo     4) SOLTE o BOOT; ao final, desligue/religue o USB da placa.
echo.
echo   O script repete sozinho ate' achar o BOOT; se esgotar as
echo   tentativas, e' so' rodar de novo.
echo ============================================================
echo.
pause

rem PlatformIO Core: 1) o caminho padrao (extensao do VS Code); 2) o `pio` do
rem .venv do projeto (este .bat vive em firmware\esp32_duas_imus -> ..\..\.venv);
rem 3) o "pio" que estiver no PATH.
set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
if not exist "%PIO%" set "PIO=%~dp0..\..\.venv\Scripts\pio.exe"
if not exist "%PIO%" set "PIO=pio"

for /L %%i in (1,1,%TENTATIVAS%) do (
    echo.
    echo ============================================================
    echo   TENTATIVA %%i de %TENTATIVAS% -- BOOT pressionado, por favor
    echo ============================================================
    "%PIO%" run -e esp32_duas_imus -t upload --upload-port %PORT%
    if not errorlevel 1 goto :sucesso
    echo.
    echo   Tentativa %%i FALHOU -- provavelmente o BOOT nao estava
    echo   pressionado durante o "Connecting". Proxima em 4 s...
    ping -n 5 127.0.0.1 >nul
)

echo.
echo ============================================================
echo   RESULTADO: FALHOU nas %TENTATIVAS% tentativas.
echo   Confira: cabo no COM certo; BOOT pressionado durante o
echo   "Connecting........"; nenhum outro programa usando %PORT%
echo   (feche outros terminais/uploads); alimentacao da placa.
echo   Rode o gravar.bat de novo quando quiser repetir.
echo ============================================================
pause
exit /b 1

:sucesso
echo.
echo ============================================================
echo   RESULTADO: SUCESSO! ESP32 com 2 MPU6050 gravado em %PORT%.
echo.
echo   PROXIMO PASSO: SOLTE o BOOT, desligue e religue o USB
echo   (sem o replug a placa sai em modo download e o firmware
echo   novo nao sobe).
echo.
echo   Conferir os IMUs:
echo     .venv\Scripts\python.exe tools\diagnostico_imu_serial.py %PORT% 115200 15
echo   (o sondeo varre os GPIOs e imprime o par SDA/SCL achado)
echo ============================================================
pause
exit /b 0
