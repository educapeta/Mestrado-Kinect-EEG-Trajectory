# Executa o benchmark EEGNet (Faria) na Intel Arc via XPU.
# Uso:
#   .\run_eegnet_arc.ps1
#   .\run_eegnet_arc.ps1 -Caso 3ch4cls
#   .\run_eegnet_arc.ps1 -Epochs 50 -Caso 3ch4cls   # teste rápido

param(
    [string]$Device = "xpu",
    [int]$Epochs = 500,
    [ValidateSet("22ch4cls", "3ch4cls", "22ch2cls", "todos")]
    [string]$Caso = "todos"
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $raiz

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Venv nao encontrado em $py. Veja docs/RODAR_EEGNET_ARC.md."
}

Write-Host "Conferindo dispositivo..."
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('xpu', hasattr(torch, 'xpu') and torch.xpu.is_available())"

$saida = Join-Path $raiz "_eegnet_faria_500"
New-Item -ItemType Directory -Force -Path $saida | Out-Null

function Rodar($nome, $nCanais, $nClasses) {
    $log = Join-Path $saida "log_${nome}.txt"
    Write-Host "`n=== $nome | epochs=$Epochs | device=$Device ==="
    Write-Host "Log: $log"
    & $py tools\eegnet_faria_benchmark.py `
        --dataset 2a --n-canais $nCanais --n-classes $nClasses `
        --folds 4 --epochs $Epochs --device $Device --verbose `
        --saida $saida 2>&1 | Tee-Object -FilePath $log
}

if ($Caso -eq "todos" -or $Caso -eq "22ch4cls") { Rodar "2a_22ch_4cls" 22 4 }
if ($Caso -eq "todos" -or $Caso -eq "3ch4cls") { Rodar "2a_3ch_4cls" 3 4 }
if ($Caso -eq "todos" -or $Caso -eq "22ch2cls") { Rodar "2a_22ch_2cls" 22 2 }

$pasta2b = Join-Path $raiz "Datasets\BCICIV_2b_gdf"
if ($Caso -eq "todos" -and (Test-Path $pasta2b)) {
    $log = Join-Path $saida "log_2b_3ch_2cls.txt"
    Write-Host "`n=== 2b_3ch_2cls | epochs=$Epochs | device=$Device ==="
    & $py tools\eegnet_faria_benchmark.py `
        --dataset 2b --n-canais 3 --n-classes 2 `
        --folds 4 --epochs $Epochs --device $Device --verbose `
        --saida $saida 2>&1 | Tee-Object -FilePath $log
}

Write-Host "`nConcluido. Resultados em $saida"
