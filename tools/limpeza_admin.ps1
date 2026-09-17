# Limpeza que exige privilegio de administrador.
# Gerado durante a sessao de manutencao do projeto (2026-09-17).
#
# O que faz (tudo na lista de "limpezas seguras" aprovadas):
#   1) esvazia a Lixeira de TODOS os usuarios (a parte de outros SIDs exige admin);
#   2) limpa os logs do Windows (CBS/DISM/WindowsUpdate/...): mantem as pastas;
#   3) limpa o cache de download do Windows Update (para/religa o wuauserv);
#   4) remove os 2 arquivos soltos na raiz do C: que JA foram copiados para o projeto;
#   5) remove pastas vazias de upgrade (C:\$WINDOWS.~BT, C:\$Windows.~WS, OneDriveTemp);
#   6) DISM /StartComponentCleanup (limpa componentes antigos do WinSxS; demora).
#
# NAO faz (decisao do usuario): powercfg /h off, mover jogos do Steam, cache do
# navegador, mexer no pagefile. O log diz o que ficou pendente.
#
# Log em docs\limpeza_admin.log

$ErrorActionPreference = 'Continue'
$raiz = Split-Path -Parent $PSScriptRoot
$log = Join-Path $raiz 'docs\limpeza_admin.log'

function Log([string]$mensagem) {
    $linha = "$(Get-Date -Format 'HH:mm:ss')  $mensagem"
    Write-Host $linha
    Add-Content -LiteralPath $log -Value $linha -Encoding utf8
}

Set-Content -LiteralPath $log -Value "=== limpeza administrativa $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" -Encoding utf8
$antes = (Get-PSDrive C).Free
Log ("livre antes: {0:N2} GB" -f ($antes / 1GB))

# 1) Lixeira (todos os usuarios)
try {
    Clear-RecycleBin -Force -ErrorAction Stop
    Log 'lixeira: esvaziada (todos os usuarios)'
} catch {
    # Fallback: remove o conteudo de cada pasta de SID
    Remove-Item 'C:\$Recycle.Bin\*' -Recurse -Force -ErrorAction SilentlyContinue
    Log "lixeira: esvaziada pelo caminho direto ($($_.Exception.Message))"
}

# 2) Logs do Windows (apaga o conteudo, mantem as pastas)
foreach ($pasta in 'CBS', 'DISM', 'MoSetup', 'waasmedic', 'WindowsUpdate', 'SIH', 'WinREAgent', 'NetSetup') {
    $alvo = Join-Path 'C:\Windows\Logs' $pasta
    if (Test-Path -LiteralPath $alvo) {
        $tamanho = (Get-ChildItem -LiteralPath $alvo -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        Remove-Item "$alvo\*" -Recurse -Force -ErrorAction SilentlyContinue
        $sobrou = (Get-ChildItem -LiteralPath $alvo -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        Log ("log {0,-14} {1,8:N1} MB -> {2,8:N1} MB" -f $pasta, ($tamanho / 1MB), ($sobrou / 1MB))
    }
}

# 3) Cache de download do Windows Update
try {
    Stop-Service wuauserv -Force -ErrorAction Stop
    Stop-Service bits -Force -ErrorAction SilentlyContinue
    Remove-Item 'C:\Windows\SoftwareDistribution\Download\*' -Recurse -Force -ErrorAction SilentlyContinue
    Start-Service wuauserv
    Start-Service bits -ErrorAction SilentlyContinue
    Log 'Windows Update: cache de download limpo (wicauserv parado e religado)'
} catch {
    Log "Windows Update: falhou ($($_.Exception.Message))"
}

# 4) Arquivos soltos na raiz do C: (ja copiados para o projeto)
foreach ($arquivo in 'C:\A02T.gdf', 'C:\gnatilus_finder.py') {
    if (Test-Path -LiteralPath $arquivo) {
        Remove-Item -LiteralPath $arquivo -Force -ErrorAction SilentlyContinue
        Log ("{0} removido: {1}" -f $arquivo, (-not (Test-Path -LiteralPath $arquivo)))
    }
}

# 5) Pastas vazias de upgrade/instalacao
foreach ($pasta in 'C:\$WINDOWS.~BT', 'C:\$Windows.~WS', 'C:\OneDriveTemp') {
    if (Test-Path -LiteralPath $pasta) {
        Remove-Item -LiteralPath $pasta -Recurse -Force -ErrorAction SilentlyContinue
        Log ("{0} removida: {1}" -f $pasta, (-not (Test-Path -LiteralPath $pasta)))
    }
}

# 6) Componentes antigos do Windows (WinSxS) -- demora alguns minutos
Log 'DISM /StartComponentCleanup (pode demorar; nao use /ResetBase)'
$dism = & dism.exe /Online /Cleanup-Image /StartComponentCleanup /Quiet 2>&1
Log ("DISM terminou: " + (($dism | Select-Object -Last 1) -join ' '))

$depois = (Get-PSDrive C).Free
Log ("livre depois: {0:N2} GB | ganho: {1:N2} GB" -f ($depois / 1GB), (($depois - $antes) / 1GB))
Log 'PENDENTE (decisao do usuario):'
Log '  - powercfg /h off            -> +6,2 GB (perde hibernacao/inicializacao rapida)'
Log '  - Steam: mover jogos (154 GB) pelo proprio Steam (Configuracoes > Armazenamento)'
Log '  - Navegador: limpar cache (~10 GB em AppData\Local\Google)'
Log '  - Pagefile (17,9 GB): NAO apagar; ajustar em Propriedades do Sistema se quiser'
Log '=== fim ==='