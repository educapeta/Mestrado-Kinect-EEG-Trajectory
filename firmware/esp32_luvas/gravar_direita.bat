@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PORT=%~1"
if not "%PORT%"=="" goto :tem_port
set /p "PORT=  Porta serial (ex: COM6 / COM7) [COM6]: "
if "%PORT%"=="" set "PORT=COM6"

:tem_port

echo ============================================================
echo   GRAVAR LUVA DIREITA      (env luva_direita / UDP 4210)
echo   Porta serial: %PORT%
echo   Wi-Fi: hotspot EDUARDO-PC 9049 (PSK) -> unicast 192.168.137.1
echo   (eduroam DESLIGADA no firmware: client isolation derruba o UDP)
echo ============================================================
echo.
echo   COMO GRAVAR (NAO use o botao EN/RST):
echo     1) Acompanhe o terminal.
echo     2) Quando aparecer "Connecting........", SEGURE o BOOT.
echo     3) Continue segurando ate aparecer "Writing at 0x000...".
echo     4) Solte o BOOT.
echo.
echo   Uso:  gravar_direita.bat [COMx]     (padrao: COM6)
echo.
pause

rem PlatformIO Core: 1) o caminho padrao (extensao do VS Code); 2) o `pio` do
rem .venv do projeto (este .bat vive em firmware\esp32_luvas -> ..\..\.venv);
rem 3) o "pio" que estiver no PATH.
set "PIO=%USERPROFILE%\.platformio\penv\Scripts\pio.exe"
if not exist "%PIO%" set "PIO=%~dp0..\..\.venv\Scripts\pio.exe"
if not exist "%PIO%" set "PIO=pio"

"%PIO%" run -e luva_direita -t upload --upload-port %PORT%

echo.
echo ============================================================
if errorlevel 1 (
  echo   RESULTADO: FALHOU. Segure o BOOT durante o "Connecting".
) else (
  echo   RESULTADO: SUCESSO! Luva DIREITA gravada em %PORT% (UDP 4210).
)
echo ============================================================
pause
