# ============================================================================
#  config.example.ps1  --  copy to config.ps1 and edit.
#
#  Every script in setup\ dot-sources setup\config.ps1 if it exists, otherwise
#  it falls back to the defaults below (the layout this port was built on).
#  launch\run-flashnext.bat reads the same values from environment variables;
#  bootstrap.ps1 exports them before calling it, so you normally only edit here.
# ============================================================================

$cfg = @{
    # --- engine checkout (code + .venv live here; you must already own a copy) ---
    HIBANA_HOME     = 'B:\hibana-win'
    HIBANA_MODELS   = 'B:\hibana-models'          # HF cache layout: hub\models--<org>--<name>\snapshots\<rev>\
    HIBANA_W3       = 'B:\hibana-models\w3'       # W3 3-bit expert weights (~48 GB, 333 files)

    # --- toolchain ---
    HIBANA_CUDA_HOME = 'C:\cuda13'                                                     # junction, no space in path
    HIBANA_CUDA_REAL = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'       # real CUDA 13.0 install
    HIBANA_MSVC_PATH = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' # CUDA 13 needs 2019-2022

    # --- server ---
    HIBANA_PORT      = 30004
    HIBANA_MAX_BATCH = 16
    HIBANA_CONTEXT   = 262144

    # --- one-command weights download (only used by bootstrap.ps1 -FetchWeights) ---
    HF_REPO  = 'RadixArk/Qwen3.8-Flash-Next-NVFP4'
    W3_REPO  = ''    # set this to the W3 3-bit repo id from the same publisher, or copy the files by hand

    # --- optional distilled draft head (see docs\03) ---
    HIBANA_MTP_WEIGHTS = 'B:\hibana-models\mtp\mtp_weights.v16reason.safetensors'
}
