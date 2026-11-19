param(
    [Parameter(Mandatory=$true)]
    [string]$SessaoDir,
    [ValidateSet("treino","teste-congelado")]
    [string]$Modo = "treino",
    [string]$Modelo = "",
    [string]$SalvarModelo = "",
    [string]$Saida = "",
    [switch]$Questionario
)

# Roda o feedback L/R adaptativo sobre uma sessão já adquirida (CSV+JSON).
# Modo treino: adapta com os labels do cue e pode salvar o modelo.
# Modo teste-congelado: carrega o modelo e NÃO adapta.

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\eeg;$raiz\src\mestradopy\vision"

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Venv nao encontrado em $py" }
if (-not (Test-Path $SessaoDir)) { throw "Sessao nao encontrada: $SessaoDir" }
if ($Modo -eq "teste-congelado" -and [string]::IsNullOrWhiteSpace($Modelo)) {
    throw "teste-congelado exige -Modelo caminho\\modelo.joblib"
}

$comando = @(
    "src\mestradopy\tools\feedback_adaptativo.py",
    "--sessao-dir", (Resolve-Path $SessaoDir).Path,
    "--modo", $Modo
)
if (-not [string]::IsNullOrWhiteSpace($Modelo)) { $comando += @("--modelo", $Modelo) }
if (-not [string]::IsNullOrWhiteSpace($SalvarModelo)) {
    $dir = Split-Path -Parent $SalvarModelo
    if (-not [string]::IsNullOrWhiteSpace($dir)) { New-Item -ItemType Directory -Force $dir | Out-Null }
    $comando += @("--salvar-modelo", $SalvarModelo)
}
if (-not [string]::IsNullOrWhiteSpace($Saida)) {
    New-Item -ItemType Directory -Force $Saida | Out-Null
    $comando += @("--saida", $Saida)
}
if ($Questionario) {
    $q = Join-Path $SessaoDir "questionario_estrategia.json"
    $comando += @("--questionario", $q)
}

Write-Host "=== FEEDBACK L/R - $Modo ==="
Write-Host "Sessao: $SessaoDir"
& $py @comando
exit $LASTEXITCODE
