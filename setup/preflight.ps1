#Requires -Version 5.1
<#
.SYNOPSIS
    12 pre-flight checks for running Hibana Flash-Next natively on Windows.

.DESCRIPTION
    Read-only. Tells you exactly what is missing before you waste an hour on a
    failed install. Exit code 0 = all checks passed or warned, 1 = at least one
    hard requirement is missing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup\preflight.ps1
#>
[CmdletBinding()]
param(
    [switch]$NoColor
)

$ErrorActionPreference = 'Continue'
$script:fail = 0
$script:warn = 0

function Write-Result {
    param([string]$Status, [string]$Name, [string]$Detail)
    switch ($Status) {
        'PASS' { $color = 'Green' }
        'WARN' { $color = 'Yellow'; $script:warn++ }
        'FAIL' { $color = 'Red'; $script:fail++ }
        default { $color = 'Gray' }
    }
    if ($NoColor) { $color = 'Gray' }
    Write-Host ('  [{0}] {1,-34} {2}' -f $Status, $Name, $Detail) -ForegroundColor $color
}

function Get-DriveFreeGB {
    param([string]$Path)
    try {
        $letter = $Path.Substring(0, 1)
        $drive = Get-PSDrive -Name $letter -ErrorAction Stop
        return [math]::Round($drive.Free / 1GB, 1)
    } catch {
        return $null
    }
}

Write-Host ''
Write-Host 'Hibana Flash-Next / Windows port - preflight' -ForegroundColor Cyan
Write-Host '---------------------------------------------------------------'

# ---- config ---------------------------------------------------------------
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$cfgFile = Join-Path $repo 'setup\config.ps1'
$cfg = @{}
if (Test-Path $cfgFile) {
    . $cfgFile
    Write-Host ("  config: {0}" -f $cfgFile)
} else {
    . (Join-Path $repo 'setup\config.example.ps1')
    Write-Host "  config: defaults (copy setup\config.example.ps1 to config.ps1 to edit)"
}
$home_ = $cfg.HIBANA_HOME
$models = $cfg.HIBANA_MODELS
$w3 = $cfg.HIBANA_W3
$cudaHome = $cfg.HIBANA_CUDA_HOME
$cudaReal = $cfg.HIBANA_CUDA_REAL
$msvc = $cfg.HIBANA_MSVC_PATH
$port = $cfg.HIBANA_PORT
Write-Host '---------------------------------------------------------------'

# 1 - OS / architecture
$os = [Environment]::OSVersion.Version
$is64 = [Environment]::Is64BitOperatingSystem
if ($is64 -and $os.Major -ge 10) {
    Write-Result 'PASS' 'Windows x64' ("version {0}.{1} build {2}" -f $os.Major, $os.Minor, $os.Build)
} else {
    Write-Result 'FAIL' 'Windows x64' ("found {0} (64-bit={1}); need Windows 10/11 x64" -f $os, $is64)
}

# 2/3 - GPU
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Write-Result 'FAIL' 'nvidia-smi' 'not on PATH - install the NVIDIA display driver 6xx+'
} else {
    Write-Result 'PASS' 'nvidia-smi' $smi.Source
    try {
        $line = (& nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
        $parts = $line -split '\s*,\s*'
        $gpuName = $parts[0]
        $cap = $parts[1]
        $vramMiB = [int]($parts[2] -replace '[^0-9]', '')
        $driver = $parts[3]
        if ($gpuName -match 'RTX PRO 6000' -or $cap -eq '12.0') {
            Write-Result 'PASS' 'GPU model' ("{0} (SM {1})" -f $gpuName, $cap)
        } else {
            Write-Result 'FAIL' 'GPU model' ("{0} (SM {1}) - this profile needs an RTX PRO 6000 Blackwell (SM 12.0)" -f $gpuName, $cap)
        }
        if ($vramMiB -ge 90000) {
            Write-Result 'PASS' 'VRAM' ("{0} MiB" -f $vramMiB)
        } else {
            Write-Result 'FAIL' 'VRAM' ("{0} MiB - the 3-bit profile needs >= 90 000 MiB" -f $vramMiB)
        }
        Write-Result 'PASS' 'driver' $driver
    } catch {
        Write-Result 'FAIL' 'GPU query' ("nvidia-smi failed: {0}" -f $_.Exception.Message)
    }
}

# 4 - host RAM
try {
    $ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction Stop).TotalPhysicalMemory / 1GB, 1)
    if ($ramGB -ge 120) {
        Write-Result 'PASS' 'host RAM' ("{0} GB" -f $ramGB)
    } else {
        Write-Result 'FAIL' 'host RAM' ("{0} GB - the 47.7 GB PLE table needs >= 128 GB" -f $ramGB)
    }
} catch {
    Write-Result 'WARN' 'host RAM' ("could not query: {0}" -f $_.Exception.Message)
}

# 5 - CUDA 13
if (Test-Path $cudaReal) {
    $nvcc = Join-Path $cudaReal 'bin\nvcc.exe'
    if (Test-Path $nvcc) {
        $ver = (& $nvcc --version 2>$null | Select-String -Pattern 'release ([0-9]+\.[0-9]+)' | Select-Object -First 1)
        if ($ver -and $ver.Matches[0].Groups[1].Value -eq '13.0') {
            Write-Result 'PASS' 'CUDA toolkit' ("{0} (13.0)" -f $cudaReal)
        } else {
            Write-Result 'FAIL' 'CUDA toolkit' ("{0} - need exactly 13.0 (torch cu130 / triton 3.7.1)" -f $cudaReal)
        }
    } else {
        Write-Result 'FAIL' 'CUDA toolkit' ("{0}\bin\nvcc.exe missing" -f $cudaReal)
    }
} else {
    Write-Result 'FAIL' 'CUDA toolkit' ("not found: {0}" -f $cudaReal)
}
if (Test-Path $cudaHome) {
    Write-Result 'PASS' 'CUDA junction' $cudaHome
} else {
    Write-Result 'WARN' 'CUDA junction' ("{0} does not exist - bootstrap.ps1 creates it with mklink /J" -f $cudaHome)
}

# 6 - MSVC
$vcvars = Join-Path $msvc 'Common7\Tools\VsDevCmd.bat'
if (Test-Path $vcvars) {
    Write-Result 'PASS' 'MSVC 2019/2022' $msvc
} else {
    Write-Result 'FAIL' 'MSVC 2019/2022' ("{0} - install VS 2022 Build Tools; CUDA 13 rejects VS 2026" -f $msvc)
}

# 7 - uv
if (Get-Command uv -ErrorAction SilentlyContinue) {
    $uvv = (& uv --version 2>$null | Select-Object -First 1)
    Write-Result 'PASS' 'uv' $uvv
} else {
    Write-Result 'FAIL' 'uv' 'not on PATH - install.ps1 from astral.sh/uv (powershell -c "irm https://astral.sh/uv/install.ps1 | iex")'
}

# 8 - engine checkout
$cli = Join-Path $home_ 'hibana\cli.py'
if (Test-Path $cli) {
    Write-Result 'PASS' 'engine checkout' $home_
} else {
    Write-Result 'FAIL' 'engine checkout' ("{0} - set HIBANA_HOME to a Hibana 0.1.0 checkout you may patch" -f $home_)
}
$plat = Join-Path $home_ 'hibana\_plat.py'
if (Test-Path $plat) {
    Write-Result 'PASS' 'port patches' 'hibana\_plat.py present (winpatch.py already applied)'
} else {
    Write-Result 'WARN' 'port patches' 'not applied yet - bootstrap.ps1 runs port\winpatch.py'
}
$venvPy = Join-Path $home_ '.venv\Scripts\python.exe'
if (Test-Path $venvPy) {
    Write-Result 'PASS' 'venv' $venvPy
} else {
    Write-Result 'WARN' 'venv' 'missing - bootstrap.ps1 runs port\setup_windows.py'
}

# 9/10 - disk
$freeModels = Get-DriveFreeGB $models
if ($freeModels -eq $null) {
    Write-Result 'FAIL' 'disk (models)' ("drive for {0} not found" -f $models)
} elseif ($freeModels -ge 200) {
    Write-Result 'PASS' 'disk (models)' ("{0} GB free at {1}" -f $freeModels, $models)
} else {
    Write-Result 'FAIL' 'disk (models)' ("{0} GB free - need >= 200 GB (135 GB trunk + 48 GB W3 + slack)" -f $freeModels)
}
$freeHome = Get-DriveFreeGB $home_
if ($freeHome -ne $null) {
    if ($freeHome -ge 20) {
        Write-Result 'PASS' 'disk (engine)' ("{0} GB free" -f $freeHome)
    } else {
        Write-Result 'FAIL' 'disk (engine)' ("{0} GB free - need >= 20 GB for the venv" -f $freeHome)
    }
}

# 11 - weights
$hub = Join-Path $models 'hub'
$snapCount = 0
if (Test-Path $hub) {
    $snapCount = (Get-ChildItem $hub -Recurse -Filter 'config.json' -ErrorAction SilentlyContinue | Measure-Object).Count
}
if ($snapCount -gt 0) {
    Write-Result 'PASS' 'weights (snapshot)' ("{0} config.json under {1}" -f $snapCount, $hub)
} else {
    Write-Result 'WARN' 'weights (snapshot)' ("none under {0} - see setup\bootstrap.ps1 -FetchWeights" -f $hub)
}
$w3cfg = Join-Path $w3 'config.json'
if (Test-Path $w3cfg) {
    Write-Result 'PASS' 'weights (W3 3-bit)' $w3
} else {
    Write-Result 'WARN' 'weights (W3 3-bit)' ("not found: {0} - 48 GB, 333 files" -f $w3)
}

# 12 - port
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    Write-Result 'WARN' 'port' ("{0} already listening (an engine is running?)" -f $port)
} else {
    Write-Result 'PASS' 'port' ("{0} free" -f $port)
}
Write-Host ''
Write-Host ("  {0} failed, {1} warning(s)" -f $script:fail, $script:warn)
if ($script:fail -eq 0) {
    Write-Host '  ready: run setup\bootstrap.ps1 next' -ForegroundColor Green
    exit 0
}
Write-Host '  fix the FAIL lines above before continuing' -ForegroundColor Red
exit 1
