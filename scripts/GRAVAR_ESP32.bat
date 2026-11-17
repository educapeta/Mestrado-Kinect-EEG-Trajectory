@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ===========================================================================
rem  GRAVAR ESP32 -- atalho para tools\gravar_esp32_com_boot.py (item 85).
rem
rem  Uso:   GRAVAR_ESP32.bat [COMx] [--sem-pulso] [--env env]
rem    - sem argumentos assume COM4 e o env esp32_duas_imus
rem    - voce SEGURA o BOOT; o script da' o pulso de RESET pela serial (EN) e
rem      grava com --before=no_reset (o auto-reset desta placa derruba o CP210x)
rem    - pode soltar o BOOT quando aparecer "Connecting..."
rem
rem  Se o pulso nao funcionar: segure o BOOT ENQUANTO pluga o USB, solte depois
rem  de ~2 s e rode:   GRAVAR_ESP32.bat COM4 --sem-pulso
rem ===========================================================================

if not exist ".venv\Scripts\python.exe" (
    echo ERRO: nao achei .venv\Scripts\python.exe neste projeto.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" tools\gravar_esp32_com_boot.py %*

echo.
pause
