<#
    Publica as mudancas do projeto no GitHub com UM comando (commit + push).

    Uso (dentro da pasta do projeto):
        .\tools\publicar.ps1 "corrigi o bug do marcador na ME"
        .\tools\publicar.ps1 "ajustes no README" -SemPublicar   # so' commit local

    Se o Windows bloquear o script (ExecutionPolicy), rode assim:
        powershell -ExecutionPolicy Bypass -File tools\publicar.ps1 "mensagem"

    O que ele faz:
      1. mostra o que mudou desde o ultimo commit;
      2. avisa (e para) se algum arquivo novo passar de 50 MB (GitHub nao gosta
         de arquivo grande e provavelmente e' dado que deveria ficar no .gitignore);
      3. git add -A + git commit -m "<mensagem>";
      4. git push (a credencial do GitHub ja' esta' guardada na maquina).
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Mensagem,
    [switch]$SemPublicar
)

$ErrorActionPreference = 'Stop'
$raiz = Split-Path -Parent $PSScriptRoot
Set-Location $raiz

if (-not (Test-Path (Join-Path $raiz '.git'))) {
    Write-Host "ERRO: $raiz nao e' um repositorio git." -ForegroundColor Red
    exit 1
}

# 1) o que mudou?
$mudancas = @(git status --short)
if ($mudancas.Count -eq 0 -or -not $mudancas[0]) {
    Write-Host 'Nada para publicar: nenhuma alteracao desde o ultimo commit.' -ForegroundColor Yellow
    exit 0
}
Write-Host '--- alteracoes detectadas ---' -ForegroundColor Cyan
$mudancas | ForEach-Object { Write-Host "  $_" }

# 2) arquivo grande entrando por engano?
$grandes = @()
foreach ($linha in $mudancas) {
    $arquivo = ($linha.Substring(3).Trim()) -replace '^"(.*)"$', '$1'
    $item = Get-Item -LiteralPath $arquivo -ErrorAction SilentlyContinue
    if ($item -and -not $item.PSIsContainer -and $item.Length -gt 50MB) {
        $grandes += $item
    }
}
if ($grandes.Count -gt 0) {
    Write-Host 'ATENCAO: arquivos acima de 50 MB:' -ForegroundColor Red
    $grandes | ForEach-Object { Write-Host ("  {0} ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB)) }
    Write-Host 'Nao publique isso: ajuste o .gitignore e rode de novo.' -ForegroundColor Red
    exit 1
}

# 3) commit
git add -A
if ($LASTEXITCODE -ne 0) { Write-Host 'falha no git add' -ForegroundColor Red; exit 1 }
git commit -m $Mensagem
if ($LASTEXITCODE -ne 0) { Write-Host 'falha no git commit' -ForegroundColor Red; exit 1 }

if ($SemPublicar) {
    Write-Host 'Commit feito localmente (-SemPublicar: nada foi enviado ao GitHub).' -ForegroundColor Yellow
    exit 0
}
if (-not (git remote)) {
    Write-Host 'Nenhum remote configurado. Rode antes:' -ForegroundColor Red
    Write-Host '  git remote add origin https://github.com/educapeta/Mestrado-Kinect-EEG-Trajectory.git'
    exit 1
}

# 4) integrar o que houver no GitHub ANTES de enviar
#    (acontece quando voce edita um arquivo pelo site do GitHub: o remoto fica
#    "na frente" e o push seria recusado com "fetch first")
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
Write-Host "--- integrando o que estiver no GitHub (git pull --rebase origin $branch) ---" -ForegroundColor Cyan
git pull --rebase origin $branch
if ($LASTEXITCODE -ne 0) {
    Write-Host 'CONFLITO ao integrar as mudancas do GitHub: o rebase foi cancelado.' -ForegroundColor Red
    Write-Host 'Nada foi publicado. Seu commit esta salvo localmente; resolva o conflito' -ForegroundColor Red
    Write-Host '(ou me chame) e rode o script de novo.' -ForegroundColor Red
    git rebase --abort
    exit 1
}

# 5) push
git push
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Publicado no GitHub com sucesso.' -ForegroundColor Green
    git --no-pager log --oneline -1
} else {
    Write-Host 'O push falhou (internet/credencial?). O commit esta salvo localmente.' -ForegroundColor Red
}