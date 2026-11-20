$ErrorActionPreference='Continue'
Set-Location B:\hibana-win
$env:UV_LINK_MODE='copy'
$env:PYTHONUTF8='1'
Write-Output "=== step1 uv venv ==="
& uv venv .venv --python 3.12 2>&1 | Out-String
if(-not (Test-Path .venv\Scripts\python.exe)){ Write-Output "FATAL venv missing"; exit 1 }
& .venv\Scripts\python.exe -c "import sys;print('python',sys.version)" 2>&1
Write-Output "=== step2 torch cu130 ==="
& uv pip install --python .venv\Scripts\python.exe "torch==2.13.0+cu130" --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match 2>&1 | Select-Object -Last 8 | Out-String
& .venv\Scripts\python.exe -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1
Write-Output "=== step3 rest ==="
& uv pip install --python .venv\Scripts\python.exe "numpy>=2.3,<2.4" "safetensors>=0.5" "tokenizers>=0.21" "jinja2>=3.1" "fastapi>=0.141,<0.142" "uvicorn>=0.52,<0.53" "xgrammar==0.2.8" "huggingface_hub>=1.30" "pillow>=11" "ninja==1.13.2" "einops==0.8.2" "pytest==9.1.1" "httpx==0.28.1" 2>&1 | Select-Object -Last 10 | Out-String
Write-Output "=== step4 flashinfer/b12x/triton ==="
& uv pip install --python .venv\Scripts\python.exe "flashinfer-python==0.6.17" "b12x==1.5.0" 2>&1 | Select-Object -Last 10 | Out-String
& uv pip install --python .venv\Scripts\python.exe "triton-windows" 2>&1 | Select-Object -Last 6 | Out-String
Write-Output "=== step5 import checks ==="
& .venv\Scripts\python.exe -c "import flashinfer;print('flashinfer ok',flashinfer.__version__)" 2>&1 | Select-Object -Last 5
& .venv\Scripts\python.exe -c "import b12x;print('b12x ok',b12x.__version__ if hasattr(b12x,'__version__') else '')" 2>&1 | Select-Object -Last 5
& .venv\Scripts\python.exe -c "import triton;print('triton',triton.__version__)" 2>&1 | Select-Object -Last 5
Write-Output "=== step6 pip install -e ==="
& uv pip install --python .venv\Scripts\python.exe -e . --no-deps 2>&1 | Select-Object -Last 6 | Out-String
& .venv\Scripts\python.exe -c "import hibana;print('hibana import ok')" 2>&1 | Select-Object -Last 8
Write-Output "=== done ==="
