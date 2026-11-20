$ErrorActionPreference='Continue'
Set-Location B:\hibana-win
$py = "B:\hibana-win\.venv\Scripts\python.exe"
$env:UV_LINK_MODE='copy'
Write-Output "=== torch ==="
& $py -c "import sys,torch;print('py',sys.version.split()[0]);print('torch',torch.__version__,'cuda-avail',torch.cuda.is_available(),'cu',torch.version.cuda);print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none');print('cap',torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '');x=torch.randn(4,4,device='cuda');print('matmul ok',float(x.sum().item()))" 2>&1 | Select-Object -Last 8
Write-Output "=== deps (no cutlass) ==="
& uv pip install --python $py apache-tvm-ffi click cuda-python cuda-tile einops ninja numpy nvidia-cudnn-frontend nvidia-ml-py packaging requests tabulate tqdm 2>&1 | Select-Object -Last 4
Write-Output "=== cutlass-dsl 4.8.1 (win libs) ==="
& uv pip install --python $py "nvidia-cutlass-dsl==4.8.1" "nvidia-cutlass-dsl-libs-cu13" 2>&1 | Select-Object -Last 5
Write-Output "=== flashinfer import ==="
& $py -c "import flashinfer;print('flashinfer',flashinfer.__version__)" 2>&1 | Select-Object -Last 12
