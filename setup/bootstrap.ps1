#Requires -Version 5.1
<#
.SYNOPSIS
    One-command install for native Windows inference of Qwen3.8-Flash-Next on one RTX PRO 6000.

.DESCRIPTION
    Idempotent. Runs in five stages and skips what is already done:

      1  preflight        12 checks (setup\preflight.ps1)
      2  junction         C:\cuda13 -> real CUDA 13.0 (no spaces in the path)
      3  toolchain        uv venv + torch cu130 + triton-windows + flashinfer (+ wheel patches)
      4  engine patches   port\winpatch.py against your Hibana checkout (26 idempotent patches)
      5  weights          verify presence (optionally download them with -FetchWeights)

    What this script will NOT do: download the Hibana engine (you must own a copy) or
    patch anything outside the paths in setup\config.ps1.

.PARAMETER FetchWeights
    Also run `hf download` for the trunk weights (135 GB) into %HIBANA_MODELS%\hub.
    The W3 3-bit set (48 GB) is fetched only if W3_REPO is set in config.ps1.

.PARAMETER Preload
    Run launch\preload_ple.ps1 after the weights are in place. Not needed with the
    current engine (`HIBANA_PLE_RAM=1` loads the 47.7 GB table into RAM at start).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\bootstrap.ps1
    powershell -ExecutionPolicy Bypass -File setup\bootstrap.ps1 -FetchWeights
#>
[CmdletBinding()]
param(
    [switch]$FetchWeights,
    [switch]$Preload,
    [switch]$SkipPreflight
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$cfgFile = Join-Path $repo 'setup\config.ps1'
$cfg = @{}
if (Test-Path $cfgFile) {
    . $cfgFile
} else {
    . (Join-Path $repo 'setup\config.example.ps1')
    Write-Host "[i] using defaults; copy setup\config.example.ps1 to config.ps1 to change paths" -ForegroundColor Yellow
}

$hm   = $cfg.HIBANA_HOME
$hmd  = $cfg.HIBANA_MODELS
$w3   = $cfg.HIBANA_W3
$cudaH = $cfg.HIBANA_CUDA_HOME
$cudaR = $cfg.HIBANA_CUDA_REAL
$msvc = $cfg.HIBANA_MSVC_PATH

function Step { param([int]$n, [string]$msg) Write-Host ''; Write-Host ("=== [{0}/5] {1}" -f $n, $msg) -ForegroundColor Cyan }
function Ok   { param([string]$msg) Write-Host ("    ok  {0}" -f $msg) -ForegroundColor Green }
function Info { param([string]$msg) Write-Host ("    ..  {0}" -f $msg) }
function Die  { param([string]$msg) Write-Host ("    xx  {0}" -f $msg) -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1 preflight
if (-not $SkipPreflight) {
    Step 1 'preflight'
    & (Join-Path $repo 'setup\preflight.ps1') -NoColor
    if ($LASTEXITCODE -ne 0) { Die 'preflight failed - fix the FAIL lines, then re-run' }
    Ok 'preflight passed'
} else {
    Step 1 'preflight (skipped)'
}

# ---------------------------------------------------------------- 2 junction
Step 2 'CUDA path without spaces'
if (Test-Path $cudaH) {
    Ok ("{0} already exists" -f $cudaH)
} else {
    if (-not (Test-Path $cudaR)) { Die ("real CUDA not found: {0}" -f $cudaR) }
    Info ("mklink /J {0} {1}" -f $cudaH, $cudaR)
    & cmd /c mklink /J "$cudaH" "$cudaR" | Out-Null
    if (-not (Test-Path $cudaH)) { Die 'mklink failed (needs a local NTFS volume; junctions do not need admin)' }
    Ok 'junction created'
}

# ---------------------------------------------------------------- 3 toolchain
Step 3 'toolchain (venv + packages + wheel patches)'
$venvPy = Join-Path $hm '.venv\Scripts\python.exe'
$setup = Join-Path $repo 'port\setup_windows.py'
$env:HIBANA_CHECKOUT  = $hm
$env:HIBANA_CUDA_HOME = $cudaH
$env:HIBANA_CUDA_REAL = $cudaR
$env:HIBANA_MSVC_PATH = $msvc

if (Test-Path $venvPy) {
    Info 'venv exists - re-running the installer is cheap and idempotent'
}
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) { Die 'uv not on PATH - see docs\01-porting.md section 3' }
Info 'uv run --no-project --python 3.12 port\setup_windows.py'
& uv run --no-project --python 3.12 python $setup
if ($LASTEXITCODE -ne 0) { Die 'toolchain install failed - see the first error above, then docs\05-troubleshooting.md' }
if (-not (Test-Path $venvPy)) { Die ("venv python still missing: {0}" -f $venvPy) }
Ok ("venv ready: {0}" -f $venvPy)

# ---------------------------------------------------------------- 4 engine patches
Step 4 'engine patches (idempotent)'
if (-not (Test-Path (Join-Path $hm 'hibana\cli.py'))) {
    Die ("no engine checkout at {0} - set HIBANA_HOME in setup\config.ps1" -f $hm)
}
& $venvPy (Join-Path $repo 'port\winpatch.py') $hm
if ($LASTEXITCODE -ne 0) { Die 'engine patching failed - the anchor that did not match is printed above' }
Ok 'engine patched'
if (-not (Test-Path (Join-Path $hm 'win_compat.h'))) {
    Copy-Item (Join-Path $repo 'port\win_compat.h') (Join-Path $hm 'win_compat.h') -Force
    Info 'copied win_compat.h into the checkout'
}
Copy-Item (Join-Path $repo 'port\_plat.py') (Join-Path $hm 'hibana\_plat.py') -Force
Ok 'shim in place'

# ---------------------------------------------------------------- 5 weights
Step 5 'weights'
$hub = Join-Path $hmd 'hub'
$haveTrunk = $false
if (Test-Path $hub) {
    $haveTrunk = ((Get-ChildItem $hub -Recurse -Filter 'model.safetensors.index.json' -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0)
}
if ($haveTrunk) {
    Ok ("trunk snapshot present under {0}" -f $hub)
} elseif ($FetchWeights) {
    $hf = Join-Path $hm '.venv\Scripts\hf.exe'
    if (-not (Test-Path $hf)) { Die 'hf CLI missing from the venv (pip install huggingface_hub failed)' }
    Info ("HF_HOME={0} ; hf download {1}" -f $hmd, $cfg.HF_REPO)
    $env:HF_HOME = $hmd
    & $hf download $cfg.HF_REPO
    if ($LASTEXITCODE -ne 0) { Die 'download failed - check HF token / mirror (see docs\05 D4)' }
} else {
    Info 'trunk weights missing. Either:'
    Info ("  set HF_HOME={0}; hf download {1}" -f $hmd, $cfg.HF_REPO)
    Info  '  ...or copy an existing HF cache (hub\models--...-snapshots\<rev>\) to that path'
    Info  '  then re-run:  powershell -File setup\bootstrap.ps1'
}

$haveW3 = Test-Path (Join-Path $w3 'config.json')
if ($haveW3) {
    Ok ("W3 3-bit present: {0}" -f $w3)
} elseif ($cfg.W3_REPO -and $FetchWeights) {
    $hf = Join-Path $hm '.venv\Scripts\hf.exe'
    Info ("hf download {0}" -f $cfg.W3_REPO)
    & $hf download $cfg.W3_REPO --local-dir $w3
} else {
    Info ("W3 3-bit missing: {0} (333 files, ~48 GB) - set W3_REPO in config.ps1 or copy it by hand" -f $w3)
}

if ($Preload) {
    Info 'warming the PLE table into the file cache'
    & (Join-Path $repo 'launch\preload_ple.ps1') -Ple (Join-Path $hmd 'ple')
}

# ---------------------------------------------------------------- done
Write-Host ''
Write-Host '=== done ===' -ForegroundColor Cyan
Write-Host '  start the server:  launch\run-flashnext.bat'
Write-Host ("  endpoint:          http://127.0.0.1:{0}/v1" -f $cfg.HIBANA_PORT)
Write-Host '  verify:            python port\verify.py'
Write-Host '  bit-exactness:     python bench\refcheck.py cmp base     (must print IDENTICAL)'
Write-Host '  five-point sweep:  python bench\sweep.py 1,2,4,8,16 700'
Write-Host ''
Write-Host '  First start takes ~4 min (weights + PLE + graph capture). Do not kill it.' -ForegroundColor Yellow
