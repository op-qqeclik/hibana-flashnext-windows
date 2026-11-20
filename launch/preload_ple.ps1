# Warm the 47.7 GB PLE table into the Windows file cache.
#
# Not needed with the current engine (`HIBANA_PLE_RAM=1` loads the table into the
# engine's own host buffers at start). Kept because it is still the right tool when
# you run an older build, or when you want the pages resident before a benchmark.
#
#   powershell -ExecutionPolicy Bypass -File launch\preload_ple.ps1
#   powershell -ExecutionPolicy Bypass -File launch\preload_ple.ps1 -Ple "B:\hibana-models\hub\models--RadixArk--Qwen3.8-Flash-Next-NVFP4\snapshots\7b719..."
#
param(
    [string]$Ple = '',
    [int]$BufferMB = 16
)

if (-not $Ple) {
    $cache = $env:HIBANA_HF_CACHE
    if (-not $cache) { $cache = 'B:\hibana-models' }
    $snaps = Get-ChildItem -Path (Join-Path $cache 'hub') -Directory -Filter 'models--*Flash-Next*' -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem (Join-Path $_.FullName 'snapshots') -Directory -ErrorAction SilentlyContinue }
    $cand = $snaps | Where-Object { (Get-ChildItem $_.FullName -Filter 'model-plefp8-*.safetensors' -ErrorAction SilentlyContinue) } |
        Select-Object -First 1
    if (-not $cand) {
        Write-Error "PLE shards (model-plefp8-*.safetensors) not found under $cache\hub - pass -Ple <snapshot dir>"
        exit 2
    }
    $Ple = $cand.FullName
}

$files = Get-ChildItem $Ple -Filter 'model-plefp8-*.safetensors'
if (-not $files) { Write-Error "no model-plefp8-*.safetensors in $Ple"; exit 2 }

Write-Output ("PLE table: {0} shards in {1}" -f $files.Count, $Ple)
$sw = [Diagnostics.Stopwatch]::StartNew()
$buf = New-Object byte[] ($BufferMB * 1MB)
$total = 0
foreach ($f in $files) {
    $fs = [IO.File]::Open($f.FullName, 'Open', 'Read', 'ReadWrite')
    try {
        while (($n = $fs.Read($buf, 0, $buf.Length)) -gt 0) { $total += $n }
    } finally { $fs.Dispose() }
    Write-Output ("{0} -> {1:N1} GB total, {2:N1}s" -f $f.Name, ($total / 1GB), $sw.Elapsed.TotalSeconds)
}
$sw.Stop()
Write-Output ("preloaded {0:N1} GB in {1:N1}s = {2:N0} MB/s" -f ($total / 1GB), $sw.Elapsed.TotalSeconds, ($total / 1MB / $sw.Elapsed.TotalSeconds))
Write-Output "expect ~15 s and 3.3 GB/s on a local NVMe; on a USB drive this is 4-20 MB/s (see docs\05 D2)"
