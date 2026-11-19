param(
    [string]$Sessao = "01",
    [switch]$SkipCameras
)

# Preflight antes de cada sessao EEG do protocolo L/R adaptativo.
# Uso:
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verificar_preflight_lr.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\verificar_preflight_lr.ps1 -SkipCameras

$ErrorActionPreference = "Stop"
$raiz = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $raiz
$env:PYTHONPATH = "$raiz\src\mestradopy;$raiz\src\mestradopy\core;$raiz\src\mestradopy\tools;$raiz\src\mestradopy\eeg;$raiz\src\mestradopy\vision"
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $raiz ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Venv nao encontrado em $py" }

Write-Host "=== PREFLIGHT PROTOCOLO L/R - sessao $Sessao ==="
Write-Host ""

Write-Host "[1/6] higiene da raiz..."
powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $raiz "scripts\verificar_raiz.ps1")
if ($LASTEXITCODE -ne 0) { throw "raiz suja" }

Write-Host "[2/6] plano do protocolo..."
& $py "src\mestradopy\tools\feedback_adaptativo.py" --plano
if ($LASTEXITCODE -ne 0) { throw "plano do feedback falhou" }

Write-Host "[3/6] teste do feedback (10/10)..."
& $py "tests\manual\test_feedback_adaptativo.py"
if ($LASTEXITCODE -ne 0) { throw "teste do feedback falhou" }

Write-Host "[4/6] teste do pipeline de video (73/73)..."
& $py "tests\manual\test_handmvnet_video_offline.py"
if ($LASTEXITCODE -ne 0) { throw "teste do video falhou" }

if ($SkipCameras) {
    Write-Host "[5/6] cameras: PULADO (-SkipCameras)."
} else {
    Write-Host "[5/6] cameras/calibracao (dry-run, sem gravar)..."
    & $py "src\mestradopy\tools\handmvnet_video_record.py" --dry-run
    if ($LASTEXITCODE -ne 0) { throw "dry-run das cameras falhou" }
}

Write-Host "[6/6] arquivos de calibracao..."
$calibs = @(
    "configs\calibracao\kinect_intrinsics.npz",
    "configs\calibracao\stereo_calibration.npz",
    "configs\calibracao\stereo_calibration_aux1.npz"
)
foreach ($c in $calibs) {
    if (-not (Test-Path (Join-Path $raiz $c))) { throw "calibracao ausente: $c" }
    Write-Host "  OK $c"
}

Write-Host ""
Write-Host "PREFLIGHT OK para a sessao $Sessao."
Write-Host "Proximo passo:"
Write-Host "  .\scripts\aquisicao_lr_adaptativo.ps1 -Sessao $Sessao -Sujeito S01"
