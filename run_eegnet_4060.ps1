# Executa a Table 4.1 (EEGNet do Faria) na GPU.
# Uso: .\run_eegnet_4060.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONWARNINGS = "ignore"
$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $raiz

$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Venv nao encontrado em $py. Veja docs/RODAR_EEGNET_4060.md."
}

Write-Host "Conferindo CUDA..."
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA indisponivel'"

$saida = Join-Path $raiz "_eegnet_faria_500"
New-Item -ItemType Directory -Force -Path $saida | Out-Null

Write-Host "`n=== 2a: 22 canais / 4 classes ==="
& $py tools\eegnet_faria_benchmark.py --dataset 2a --n-canais 22 --n-classes 4 --folds 4 --epochs 500 --device cuda --saida $saida

Write-Host "`n=== 2a: 3 canais / 4 classes ==="
& $py tools\eegnet_faria_benchmark.py --dataset 2a --n-canais 3 --n-classes 4 --folds 4 --epochs 500 --device cuda --saida $saida

Write-Host "`n=== 2a: 22 canais / 2 classes ==="
& $py tools\eegnet_faria_benchmark.py --dataset 2a --n-canais 22 --n-classes 2 --folds 4 --epochs 500 --device cuda --saida $saida

$pasta2b = Join-Path $raiz "Datasets\BCICIV_2b_gdf"
if (Test-Path $pasta2b) {
    Write-Host "`n=== 2b: 3 canais / 2 classes ==="
    & $py tools\eegnet_faria_benchmark.py --dataset 2b --n-canais 3 --n-classes 2 --folds 4 --epochs 500 --device cuda --saida $saida
} else {
    Write-Host "`n[2b] Pasta $pasta2b nao encontrada; pulei o caso 2b."
}

Write-Host "`nConcluido. Resultados em $saida"
