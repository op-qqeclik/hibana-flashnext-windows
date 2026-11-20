# Restart the engine: stop our own serving processes, start run-flashnext.bat detached,
# then poll /health until 200 - never a fixed sleep.
#
# Touches only `python.exe` processes whose command line contains "hibana", and only
# files under %HIBANA_HOME%.
#
#   powershell -ExecutionPolicy Bypass -File launch\restart.ps1
#   powershell -ExecutionPolicy Bypass -File launch\restart.ps1 -Bat B:\hibana-win\run-flashnext-v9.bat
param(
    [string]$Bat = '',
    [int]$Port = 0,
    [int]$TimeoutSec = 1800
)

$ErrorActionPreference = 'Continue'
if (-not $env:HIBANA_HOME) { $env:HIBANA_HOME = 'B:\hibana-win' }
if (-not $Bat) { $Bat = (Join-Path $env:HIBANA_HOME 'run-flashnext.bat') }
if ($Port -le 0) {
    if ($env:HIBANA_PORT) { $Port = [int]$env:HIBANA_PORT } else { $Port = 30004 }
}
if (-not (Test-Path $Bat)) { Write-Error "launcher not found: $Bat"; exit 2 }

$res = Join-Path $env:HIBANA_HOME 'results'
if (-not (Test-Path $res)) { New-Item -ItemType Directory -Path $res | Out-Null }
$log = Join-Path $res 'restart.log'
function Log($m) { "$([DateTime]::Now.ToString('HH:mm:ss')) $m" | Tee-Object -FilePath $log -Append }

$ours = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*hibana serve*' }
foreach ($p in $ours) { Log "stopping pid $($p.ProcessId)"; Stop-Process -Id $p.ProcessId -Force }
Start-Sleep -Seconds 3

# The multiprocess children of a killed server can linger; only ours carry "hibana".
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*hibana*' } |
    ForEach-Object { Log "stopping leftover pid $($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }

for ($i = 0; $i -lt 20; $i++) {
    $smi = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
    if ([double]$smi -lt 50000) { Log "gpu released: $smi MiB used"; break }
    Start-Sleep -Seconds 3
}

Log "starting the engine: $Bat"
$task = 'hibana-flashnext'
# A task left in the Running state makes /run a silent no-op (seen twice). Always end + delete first.
& schtasks /end /tn $task 2>$null | Out-Null
& schtasks /delete /tn $task /f 2>$null | Out-Null
& schtasks /create /tn $task /tr "cmd /c $Bat" /sc once /st 23:58 /f | Out-Null
& schtasks /run /tn $task | Out-Null

$sw = [Diagnostics.Stopwatch]::StartNew()
$url = "http://127.0.0.1:$Port/health"
while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Seconds 5
    try {
        $h = Invoke-WebRequest -UseBasicParsing -Uri $url -TimeoutSec 5
        if ($h.StatusCode -eq 200) {
            Log ("serving after {0:N0} s ({1})" -f $sw.Elapsed.TotalSeconds, $url)
            exit 0
        }
    } catch { }
}
Log "TIMEOUT waiting for $url - check $env:HIBANA_HOME\results\flashnext.log"
exit 1
