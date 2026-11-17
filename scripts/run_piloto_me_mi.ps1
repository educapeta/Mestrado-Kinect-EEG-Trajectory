param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("01","02")]
    [string]$Sessao = "01",
    [string]$Sujeito = "S01",
    [string]$Device = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\eeg;$raiz\src\mestradopy\vision;$raiz\src\mestradopy\acquisition"

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Venv nao encontrado em $py."
}

$saida = Join-Path $raiz "outputs\eeg\piloto_me_mi_s$Sessao"
New-Item -ItemType Directory -Force -Path $saida | Out-Null

Write-Host "=== PILOTO ME 4s + MI 4s - sessao $Sessao ==="
Write-Host "Baseline: 60 s olhos abertos + 60 s olhos fechados"
Write-Host "Trial: cue 1 s -> ME 4 s -> transicao 1 s -> MI 4 s -> ITI 1-2 s"
Write-Host "2 blocos x 15 trials/classe = 30 trials/classe nesta sessao"
Write-Host "Saida: $saida"
Write-Host "Feche o gRecorder/GtecSuite antes de continuar."
Write-Host ""

$comando = @(
    "src\mestradopy\tools\validacao_eeg_3classes.py",
    "--source", "gnautilus",
    "--blocos", "2",
    "--trials-por-classe", "15",
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
    "--pasta-saida", $saida
)

& $py @comando 2>&1 | Tee-Object -FilePath (Join-Path $saida "log_aquisicao_s$Sessao.txt")
