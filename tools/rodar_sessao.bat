@echo off
rem ============================================================================
rem Roda o paradigma ME/MI e FECHA a janela sozinha quando a sessao termina.
rem
rem Motivo (30/09/2026): abrir o PowerShell com -NoExit deixava a janela
rem parada depois da sessao; e o terminal do VS Code nunca fecha sozinho (e'
rem um terminal, nao uma janela). Aqui a saida fica visivel durante a sessao
rem (5 s no fim, 20 s se der erro) e a janela some sozinha.
rem
rem Uso (qualquer argumento do paradigma vale):
rem     tools\rodar_sessao.bat --source gnautilus --tela-cheia --aux-cameras auto
rem ============================================================================
setlocal
cd /d "%~dp0.."

".venv\Scripts\python.exe" -u eeg_motor_paradigm.py %*
set ERRO=%ERRORLEVEL%
echo.
if %ERRO% neq 0 (
    echo [rodar_sessao] paradigma terminou com ERRO %ERRO% - veja a saida acima.
    echo [rodar_sessao] fechando em 20 s...
    timeout /t 20 /nobreak >nul
) else (
    echo [rodar_sessao] sessao encerrada - fechando em 5 s...
    timeout /t 5 /nobreak >nul
)
exit /b %ERRO%
