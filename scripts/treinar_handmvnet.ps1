param(
    [Parameter(Mandatory=$true)][string]$Dados,
    [string]$Validacao = "",
    [string]$Saida = "outputs\visao\handmvnet_treino",
    [int]$Epocas = 40,
    [string]$Device = "xpu",
    [int]$Batch = 2,
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\vision;$raiz\src\mestradopy\eeg"
$py = Join-Path $raiz ".venv\Scripts\python.exe"

$args = @("src\mestradopy\tools\handmvnet_train.py",
          "--data", $Dados, "--saida", $Saida,
          "--epochs", "$Epocas", "--device", $Device,
          "--batch-size", "$Batch")
if ($Validacao -ne "") { $args += @("--data-val", $Validacao) }
if ($Checkpoint -ne "") { $args += @("--checkpoint", $Checkpoint) }

Write-Host "Dados: $Dados"
Write-Host "Saida: $Saida | epochs=$Epocas | device=$Device"
& $py @args
