@echo off
rem ============================================================================
rem  Hibana Flash-Next on a single RTX PRO 6000 (Windows native, no WSL/Docker)
rem  Log: %HIBANA_HOME%\results\flashnext.log
rem
rem  Paths default to the layout this port was built on. Override any of them by
rem  setting the variable before calling this script (or edit the defaults below).
rem  See setup\config.example.ps1 for the same values in PowerShell form.
rem ============================================================================
setlocal EnableExtensions

if not defined HIBANA_HOME      set "HIBANA_HOME=B:\hibana-win"
if not defined HIBANA_MODELS    set "HIBANA_MODELS=B:\hibana-models"
if not defined HIBANA_W3        set "HIBANA_W3=%HIBANA_MODELS%\w3"
if not defined HIBANA_CUDA_HOME set "HIBANA_CUDA_HOME=C:\cuda13"
if not defined HIBANA_MSVC_PATH set "HIBANA_MSVC_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if not defined HIBANA_PORT      set "HIBANA_PORT=30004"
if not defined HIBANA_MAX_BATCH set "HIBANA_MAX_BATCH=16"
if not defined HIBANA_CONTEXT   set "HIBANA_CONTEXT=262144"

rem CUDA 13 needs an MSVC 2019-2022 host compiler. If the box also has a newer
rem toolset, vswhere -latest picks that one and cudafe++ dies with 0xC0000005,
rem so the 2022 Build Tools dev prompt is entered first and inherited by
rem torch / flashinfer / nvcc alike.
call "%HIBANA_MSVC_PATH%\Common7\Tools\VsDevCmd.bat" -arch=x64 >nul 2>&1
set DISTUTILS_USE_SDK=1
set HIBANA_WIN_COMPAT=%HIBANA_HOME%\win_compat.h
set HIBANA_HF_CACHE=%HIBANA_MODELS%
set HIBANA_RESULTS=%HIBANA_HOME%\results
set HIBANA_W3_CHECKPOINT=%HIBANA_W3%
set HIBANA_CUDA_ARCH=12.0a
set HIBANA_HOST_FLOOR_GIB=2

rem Distilled draft head (optional). A partial export is legal: tensors it omits
rem fall through to the checkpoint. Leave the variable unset for the stock head.
if not defined HIBANA_MTP_WEIGHTS if exist "%HIBANA_MODELS%\mtp\mtp_weights.v16reason.safetensors" set "HIBANA_MTP_WEIGHTS=%HIBANA_MODELS%\mtp\mtp_weights.v16reason.safetensors"

set HIBANA_SPEC=8
rem Each draft step is one host<->device handshake, and under WDDM a handshake
rem costs ~1.85 ms while an extra draft is worth only ~0.35 token. Hence deep on
rem c1/c2, shallow on c8/c16. Measured table: docs\02-tuning.md section 2.2.
rem HIBANA_PLE_BATCHED=1 measured worse (streaming the columns overlaps the
rem draft chain; one batch does not).
set HIBANA_SPEC_DEPTHS=1:8,2:6,4:3,8:3,16:4

rem verify-state stays at the engine default (reconstruct): snapshots reserves
rem per-request state copies and does not fit beside the 3-bit experts in 96 GiB.
set HIBANA_VISION=1
rem The checkpoint's 16.7 MP vision tile does not fit beside the pools: a 4K
rem capture costs 1.3 GiB and wedges admission. A quarter ceiling keeps 1080p and
rem 1440p native and 4K at ~1.4x downscale.
set HIBANA_VISION_MAX_PIXELS=4194304

set HIBANA_GDN_PREFILL=triton
rem HIBANA_KV_FORMAT=nvfp4 needs flashinfer's FP4-KV module, which does not
rem compile in this port; fp8 is the working format.
set HIBANA_KV_FORMAT=fp8
set HIBANA_VRAM_RESERVE_GIB=2
set HIBANA_KV_POOL_TOKENS=524288
set HIBANA_STATE_SLOTS=36
set HIBANA_SSD=0

set CUDA_HOME=%HIBANA_CUDA_HOME%
set CUDA_PATH=%HIBANA_CUDA_HOME%
set PATH=%HIBANA_CUDA_HOME%\bin;%HIBANA_HOME%\.venv\Scripts;%PATH%
set PYTHONUTF8=1

rem Requests that carry no sampling fields get this temperature
rem (server.py DEFAULT_TEMPERATURE).
set HIBANA_DEFAULT_TEMPERATURE=0.2
rem The reasoning channel is closed after this many tokens, or once its trailing
rem window of 48 tokens repeats 3x (engine/loopguard.py).
set HIBANA_THINKING_BUDGET=4096
set HIBANA_THINKING_GUARD=48:3:16

rem flashinfer plan timings in the log; set to 0 for quiet logs.
set _FI_TIMING=1

if not exist "%HIBANA_RESULTS%" mkdir "%HIBANA_RESULTS%"
if not exist "%HIBANA_HOME%\.venv\Scripts\python.exe" (
  echo [ERROR] no venv at "%HIBANA_HOME%\.venv" - run setup\bootstrap.ps1 first. 1>&2
  exit /b 2
)
cd /d "%HIBANA_HOME%"

echo [%DATE% %TIME%] starting hibana flashnext >> "%HIBANA_RESULTS%\flashnext.log"
"%HIBANA_HOME%\.venv\Scripts\python.exe" -m hibana serve flashnext --no-fetch --host 0.0.0.0 --port %HIBANA_PORT% --max-batch %HIBANA_MAX_BATCH% --context-length %HIBANA_CONTEXT% >> "%HIBANA_RESULTS%\flashnext.log" 2>&1
echo [%DATE% %TIME%] exited with %ERRORLEVEL% >> "%HIBANA_RESULTS%\flashnext.log"
endlocal
