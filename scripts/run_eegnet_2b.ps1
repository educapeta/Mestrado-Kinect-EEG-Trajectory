# BCIC-IV 2b: EEGNet 3ch/2cls (1.154 parâmetros)
# Uso:
#   .\run_eegnet_2b.ps1
#   .\run_eegnet_2b.ps1 -Caso benchmark
#   .\run_eegnet_2b.ps1 -Caso aplicar
#   .\run_eegnet_2b.ps1 -Caso aplicar -Epochs 100 -BatchSize 128

param(
    [ValidateSet("todos", "benchmark", "aplicar")]
    [string]$Caso = "todos",
    [string]$Device = "xpu",
    [int]$Epochs = 500,
    [int]$BatchSize = 32
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

Write-Host "Conferindo dispositivo..."
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'xpu', hasattr(torch, 'xpu') and torch.xpu.is_available())"

$saida2b = Join-Path $raiz "outputs\eeg\eegnet_faria_2b_3ch2cls"
$saidaBench = Join-Path $raiz "outputs\eeg\eegnet_faria_2b_benchmark"
New-Item -ItemType Directory -Force -Path $saida2b | Out-Null

if ($Caso -eq "todos" -or $Caso -eq "benchmark") {
    New-Item -ItemType Directory -Force -Path $saidaBench | Out-Null
    Write-Host "`n=== 2b: benchmark 4-fold, 3 canais / 2 classes ==="
    & $py src\mestradopy\tools\eegnet_faria_benchmark.py `
        --dataset 2b --n-canais 3 --n-classes 2 `
        --folds 4 --epochs $Epochs --device $Device --verbose `
        --saida $saidaBench 2>&1 |
        Tee-Object -FilePath (Join-Path $saidaBench "log_2b_3ch_2cls.txt")
}

if ($Caso -eq "todos" -or $Caso -eq "aplicar") {
    Write-Host "`n=== 2b: treino final + teste nos 40 trials left/right do piloto ==="
    & $py src\mestradopy\tools\eegnet_faria_2b_aplicar_mi.py `
        --device $Device --epochs $Epochs --batch-size $BatchSize `
        --progress-every 10 --saida $saida2b 2>&1 |
        Tee-Object -FilePath (Join-Path $saida2b "log_aplicacao_2b.txt")
}

Write-Host "`nConcluido."
Write-Host "Benchmark: $saidaBench"
Write-Host "Modelo/resultado: $saida2b"
