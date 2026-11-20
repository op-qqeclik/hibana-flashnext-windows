$ErrorActionPreference='Continue'
Set-Location B:\hibana-win
$py = "B:\hibana-win\.venv\Scripts\python.exe"
$env:UV_LINK_MODE='copy'
function Run($desc, $block) { Write-Output "=== $desc ==="; & $block 2>&1 | Select-Object -Last 12 | Out-String }
Write-Output "=== python/torch ==="
& $py -c "import sys,torch;print(sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.version.cuda);print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '')" 2>&1 | Select-Object -Last 6
Write-Output "=== triton 3.7.1 pin ==="
& uv pip install --python $py "triton-windows==3.7.1.post27" 2>&1 | Select-Object -Last 5
& $py -c "import triton;print('triton',triton.__version__)" 2>&1 | Select-Object -Last 3
Write-Output "=== flashinfer without nccl4py ==="
& uv pip install --python $py --no-deps "flashinfer-python==0.6.17" 2>&1 | Select-Object -Last 4
& uv pip install --python $py "apache-tvm-ffi>=0.1.6,<0.2" click "cuda-python>=12.0" "cuda-tile>=1.4.0" einops ninja numpy "nvidia-cudnn-frontend>=1.25.0" "nvidia-cutlass-dsl>=4.5.0" nvidia-ml-py "packaging>=24.2" requests tabulate tqdm 2>&1 | Select-Object -Last 6
& $py -c "import flashinfer;print('flashinfer',flashinfer.__version__)" 2>&1 | Select-Object -Last 12
Write-Output "=== b12x ==="
& uv pip install --python $py "b12x==1.5.0" 2>&1 | Select-Object -Last 6
& $py -c "import b12x;print('b12x ok')" 2>&1 | Select-Object -Last 8
Write-Output "=== done ==="
