param(
    [string]$Saida = "F:\handmvnet",
    [int]$Cada = 5,
    [string]$Lado = "auto",
    [switch]$Mostrar
)

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\vision;$raiz\src\mestradopy\eeg"
$py = Join-Path $raiz ".venv\Scripts\python.exe"

$args = @("src\mestradopy\tools\handmvnet_capture.py",
          "--saida", $Saida, "--cada", "$Cada", "--lado", $Lado)
if ($Mostrar) { $args += "--mostrar" } else { $args += "--sem-janela" }

Write-Host "Destino: $Saida"
Write-Host "Amostra a cada $Cada frames; lado=$Lado"
& $py @args
