param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("01","02","03")]
    [string]$Sessao = "01",
    [string]$Sujeito = "S01",
    [string]$PastaSaida = ""
)

# Aquisição do protocolo L/R adaptativo (ME -> MI).
# Sessões 01 e 02 = treino adaptativo; sessão 03 = teste congelado (mesma aquisição).
# 2 blocos x 10 trials/classe = 20/classe por sessão; 3 sessões = 60/classe no total.
# NUNCA rodar 60 trials/classe em sequência.

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\eeg;$raiz\src\mestradopy\vision;$raiz\src\mestradopy\acquisition"

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Venv nao encontrado em $py" }

if ([string]::IsNullOrWhiteSpace($PastaSaida)) {
    $PastaSaida = Join-Path $raiz "outputs\eeg\lr_adaptativo_s$Sessao"
}
New-Item -ItemType Directory -Force -Path $PastaSaida | Out-Null

$papel = if ($Sessao -eq "03") { "TESTE CONGELADO (nao adaptar)" } else { "TREINO ADAPTATIVO" }
Write-Host "=== PROTOCOLO L/R ADAPTATIVO - sessao $Sessao - $papel ==="
Write-Host "Sujeito: $Sujeito"
Write-Host "Trial: cue 1 s -> ME 4 s -> transicao 1 s -> MI 4 s -> ITI 1-2 s"
Write-Host "2 blocos x 10 trials/classe = 20/classe nesta sessao"
Write-Host "Total do protocolo: 3 sessoes x 20/classe = 60/classe"
Write-Host "Saida: $PastaSaida"
Write-Host "Feche o gRecorder/GtecSuite antes de continuar."
Write-Host ""

$comando = @(
    "src\mestradopy\tools\validacao_eeg_3classes.py",
    "--source", "gnautilus",
    "--blocos", "2",
    "--trials-por-classe", "10",
    "--classes", "esquerda,direita",
    "--me-sec", "4",
    "--mi-sec", "4",
    "--transicao-sec", "1",
    "--cue-sec", "1",
    "--repouso-sec", "2",
    "--iti-min", "1",
    "--iti-max", "2",
    "--baseline-olhos-sec", "60",
    "--sujeito", $Sujeito,
    "--sessao", $Sessao,
    "--pasta-saida", $PastaSaida
)

# Feedback online: s01 calibra no bloco 1; s02 adapta; s03 congelado.
$modeloDir = Join-Path $raiz "outputs\visao\lr_adaptativo"
New-Item -ItemType Directory -Force -Path $modeloDir | Out-Null
$comando += @("--feedback", "online")
if ($Sessao -eq "01") {
    $comando += @("--salvar-modelo", (Join-Path $modeloDir "modelo_s01.joblib"))
} elseif ($Sessao -eq "02") {
    $modelo01 = Join-Path $modeloDir "modelo_s01.joblib"
    if (-not (Test-Path $modelo01)) { throw "Modelo s01 nao encontrado: $modelo01" }
    $comando += @("--modelo", $modelo01,
                  "--salvar-modelo", (Join-Path $modeloDir "modelo_s02.joblib"))
} else {
    $modelo02 = Join-Path $modeloDir "modelo_s02.joblib"
    if (-not (Test-Path $modelo02)) { throw "Modelo s02 nao encontrado: $modelo02" }
    $comando += @("--modelo", $modelo02, "--congelado")
}

& $py @comando 2>&1 | Tee-Object -FilePath (Join-Path $PastaSaida "log_aquisicao_s$Sessao.txt")
if ($LASTEXITCODE -ne 0) { throw "Aquisicao falhou (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Aquisicao concluida. Proximo passo:"
if ($Sessao -eq "01") {
    Write-Host "  analisar/treinar feedback: .\scripts\rodar_feedback_lr.ps1 -SessaoDir `"$PastaSaida`" -Modo treino -SalvarModelo outputs\visao\lr_adaptativo\modelo_s01.joblib"
} elseif ($Sessao -eq "02") {
    Write-Host "  continuar treino: .\scripts\rodar_feedback_lr.ps1 -SessaoDir `"$PastaSaida`" -Modo treino -Modelo outputs\visao\lr_adaptativo\modelo_s01.joblib -SalvarModelo outputs\visao\lr_adaptativo\modelo_s02.joblib"
} else {
    Write-Host "  teste congelado: .\scripts\rodar_feedback_lr.ps1 -SessaoDir `"$PastaSaida`" -Modo teste-congelado -Modelo outputs\visao\lr_adaptativo\modelo_s02.joblib"
}
