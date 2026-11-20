# Engine status: health, GPU memory, our processes, and the tail of the current run's log.
#
#   powershell -ExecutionPolicy Bypass -File launch\status.ps1
param(
    [int]$Port = 0,
    [string]$Log = '',
    [int]$Lines = 30
)

if (-not $env:HIBANA_HOME) { $env:HIBANA_HOME = 'B:\hibana-win' }
if ($Port -le 0) {
    if ($env:HIBANA_PORT) { $Port = [int]$env:HIBANA_PORT } else { $Port = 30004 }
}
if (-not $Log) { $Log = Join-Path $env:HIBANA_HOME 'results\flashnext.log' }

$health = 'down'
try {
    $h = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
    $health = $h.StatusCode
} catch { }
Write-Output "health: $health ($Port)"
Write-Output ("gpu: " + (& nvidia-smi --query-gpu=memory.used --format=csv,noheader))
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*hibana*' } |
    ForEach-Object { Write-Output ("proc " + $_.ProcessId + " <- " + $_.ParentProcessId + " : " + $_.CommandLine) }

if (-not (Test-Path $Log)) { Write-Output "no log at $Log"; exit 0 }
$l = @(Get-Content $Log)
$i = 0
for ($k = 0; $k -lt $l.Count; $k++) { if ($l[$k] -match 'starting hibana') { $i = $k } }
Write-Output "---- log from the current start ($($l.Count - $i) lines)"
$l[([Math]::Max(0, $i))..([Math]::Max(0, $l.Count - 1))] | Select-Object -First $Lines
Write-Output "---- tail"
$l | Select-Object -Last 6
