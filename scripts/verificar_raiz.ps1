# Verifica se a raiz do projeto esta limpa.
# Uso: .\scripts\verificar_raiz.ps1
$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Venv nao encontrado em $py" }
& $py "tests\manual\test_estrutura_raiz.py"
exit $LASTEXITCODE
