@echo off
REM ============================================================================
REM  Suspensao seletiva de USB -- liga/desliga (Kinect v2 precisa dela DESLIGADA)
REM ============================================================================
REM  POR QUE EXISTE (bancada de 07/10/2026)
REM    O Kinect ficou com `sensor.IsAvailable = FALSE`: o SDK achava e abria o
REM    sensor, e ele nunca transmitia -- com o Kinect Studio mostrando imagem
REM    normalmente. O registro mostrou o proprio sensor marcado
REM    `DeviceSelectiveSuspended = 1` e o plano Equilibrado com a suspensao
REM    seletiva de USB ATIVADA (0x1 em CA e CC). O v2 nao volta a transmitir de
REM    forma confiavel depois de suspender.
REM
REM  ESTE SCRIPT NAO PRECISA DE ADMINISTRADOR -- medido em 07/10/2026: o
REM  `powercfg /setacvalueindex` funcionou no token NAO elevado (o usuario e' do
REM  grupo Administradores, com o token filtrado do UAC). Se na sua maquina der
REM  "acesso negado", rode este .bat como administrador (botao direito >
REM  Executar como administrador).
REM
REM  Uso:
REM    tools\suspensao_usb_seletiva.bat              -> mostra o estado
REM    tools\suspensao_usb_seletiva.bat desligar     -> desliga (recomendado)
REM    tools\suspensao_usb_seletiva.bat ligar        -> volta ao padrao
REM ============================================================================
setlocal
set SUB=2a737441-1930-4402-8d77-b2bebba308a3
set CFG=48e6b7a6-50f5-4782-a5d4-53bb8f07e226

if /I "%~1"=="desligar" goto desligar
if /I "%~1"=="ligar" goto ligar
if /I "%~1"=="off" goto desligar
if /I "%~1"=="on" goto ligar
goto estado

:desligar
echo Desligando a suspensao seletiva de USB (CA e CC)...
powercfg /setacvalueindex SCHEME_CURRENT %SUB% %CFG% 0
powercfg /setdcvalueindex SCHEME_CURRENT %SUB% %CFG% 0
powercfg /setactive SCHEME_CURRENT
echo.
echo Pronto. IMPORTANTE: DESLIGUE e RELIGUE a fonte do Kinect para o sensor sair
echo do estado suspenso em que ele ja' esta'. Depois rode:
echo     .venv\Scripts\python.exe tools\pos_reinicio_kinect.py
goto estado

:ligar
echo Voltando a suspensao seletiva de USB ao padrao (CA e CC = 1)...
powercfg /setacvalueindex SCHEME_CURRENT %SUB% %CFG% 1
powercfg /setdcvalueindex SCHEME_CURRENT %SUB% %CFG% 1
powercfg /setactive SCHEME_CURRENT
goto estado

:estado
echo.
echo === Estado atual (suspensao seletiva de USB) ===
powercfg /query SCHEME_CURRENT %SUB% %CFG% | findstr /I "Atuais Current"
echo.
echo   0x00000000 = DESLIGADA  ^(o que o Kinect v2 quer^)
echo   0x00000001 = LIGADA     ^(padrao do Windows^)
endlocal
