# Reset elevado do Kinect v2 (07/10/2026) -- roda VIA UAC (elevado).
# Desabilita/habilita o dispositivo (re-enumeracao = o "desligar/religar" pelo
# software) e reinicia o servico KinectMonitor. Log em C:\mestradopy\_elev.log.
$log = 'C:\mestradopy\_elev.log'
function marca([string]$m) {
    ("{0} {1}" -f (Get-Date -Format 'HH:mm:ss'), $m) | Add-Content -Path $log
}
'inicio' | Set-Content -Path $log

$comp = 'USB\VID_045E&PID_02C4\002457351347'
$filhos = @(
    'USB\VID_045E&PID_02C4&MI_00\6&31C278CF&0&0000',
    'USB\VID_045E&PID_02C4&MI_02\6&31C278CF&0&0002'
)

# 1) desabilita o composite (cascata nos filhos); se falhar, cada filho sozinho
$desligou = $false
try {
    Disable-PnpDevice -InstanceId $comp -Confirm:$false -ErrorAction Stop
    $desligou = $true
    marca "disable composite OK"
} catch {
    marca ("disable composite FALHOU: " + $_.Exception.Message)
}
if (-not $desligou) {
    foreach ($f in $filhos) {
        try {
            Disable-PnpDevice -InstanceId $f -Confirm:$false -ErrorAction Stop
            marca "disable filho OK: $f"
        } catch {
            marca ("disable filho FALHOU " + $f + " :: " + $_.Exception.Message)
        }
    }
}
Start-Sleep -Seconds 3

# 2) habilita de volta (composite e, por garantia, cada filho)
try {
    Enable-PnpDevice -InstanceId $comp -Confirm:$false -ErrorAction Stop
    marca "enable composite OK"
} catch {
    marca ("enable composite: " + $_.Exception.Message)
}
foreach ($f in $filhos) {
    try {
        Enable-PnpDevice -InstanceId $f -Confirm:$false -ErrorAction Stop
        marca "enable filho OK: $f"
    } catch {
        marca ("enable filho: " + $_.Exception.Message)
    }
}
Start-Sleep -Seconds 2

# 3) reinicia o servico do SDK (agora com direito)
try {
    Restart-Service -Name KinectMonitor -Force -ErrorAction Stop
    marca "KinectMonitor reiniciado"
} catch {
    marca ("KinectMonitor: " + $_.Exception.Message)
}

marca "feito"