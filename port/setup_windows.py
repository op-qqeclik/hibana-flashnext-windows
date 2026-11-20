"""Hibana on Windows: set up the native toolchain and patch the third-party wheels for it.

    python setup_windows.py            # venv + packages + patches (idempotent)
    python setup_windows.py --verify   # JIT smoke test (flashinfer rmsnorm on the GPU)

Why each patch exists (the Windows port's findings, 2026-11-16):

* flashinfer's own ninja generator is Linux-only (`c++`, `-fPIC`, `-MMD`, `prefer -isystem`), and its build lines
  carry Windows drive letters unescaped, which ninja rejects outright ("expected build command name"). tvm_ffi, the
  ABI flashinfer compiles against, ships a Windows-capable builder (cl/nvcc inside an MSVC dev prompt, `deps=msvc`,
  `.dll` output), so `JitSpecNvcc` is redirected onto it.
* tvm_ffi's generated build.ninja needs `C$:` for drive letters, `-Xcompiler=a,b` (one comma-separated value: nvcc
  takes a single argument), and its dev prompt is pinned to an MSVC 2019-2022 install — CUDA 13 rejects the 2026
  toolset and nvcc's `cudafe++` crashes on it.
* flashinfer's headers assume POSIX: `ushort`/`uint` from <sys/types.h>, `M_PI` without `_USE_MATH_DEFINES`,
  `__attribute__((always_inline))`, and `int64 % IntFastDiv` resolving to an int overload. `win_compat.h` supplies the
  first two by force-include; the other two are rewritten in place.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Defaults are the layout this port was built on (see setup/config.example.ps1); override with env vars
# so the same installer works on another machine without editing source.
CHECKOUT = Path(os.environ.get("HIBANA_CHECKOUT", "B:/hibana-win"))
VENV = CHECKOUT / ".venv"
PY = VENV / "Scripts" / "python.exe"
CUDA_HOME = Path(os.environ.get("HIBANA_CUDA_HOME", r"C:\cuda13"))   # a junction, so no path has a space
CUDA_TARGET = Path(os.environ.get("HIBANA_CUDA_REAL",
                                  r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"))
MSVC = Path(os.environ.get("HIBANA_MSVC_PATH",
                           r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"))

PACKAGES = [
    # torch first, from its own index: PyPI's torch is CPU-only
    ("torch==2.13.0+cu130", ["--index-url", "https://download.pytorch.org/whl/cu130",
                             "--extra-index-url", "https://pypi.org/simple",
                             "--index-strategy", "unsafe-best-match"]),
    ("numpy>=2.3,<2.4", []), ("safetensors>=0.5", []), ("tokenizers>=0.21", []), ("jinja2>=3.1", []),
    ("fastapi>=0.141,<0.142", []), ("uvicorn>=0.52,<0.53", []), ("xgrammar==0.2.8", []),
    ("huggingface_hub>=1.30", []), ("pillow>=11", []), ("ninja==1.13.2", []), ("einops==0.8.2", []),
    ("pytest==9.1.1", []), ("httpx==0.28.1", []),
    ("triton-windows==3.7.1.post27", []),          # the same Triton version the kernels are written for
    ("apache-tvm-ffi>=0.1.6,<0.2", []), ("click", []), ("cuda-python>=12.0", []), ("cuda-tile>=1.4.0", []),
    ("nvidia-cudnn-frontend>=1.25.0", []), ("nvidia-ml-py", []), ("packaging>=24.2", []), ("requests", []),
    ("tabulate", []), ("tqdm", []),
]


def run(*cmd, cwd=None, check=True, env=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip()[-2000:])
    if result.returncode and check:
        print(result.stderr.strip()[-3000:], file=sys.stderr)
        raise SystemExit(f"failed: {' '.join(str(c) for c in cmd)}")
    return result


def ensure_cuda_junction() -> None:
    if CUDA_HOME.exists():
        return
    run("cmd", "/c", "mklink", "/J", str(CUDA_HOME), str(CUDA_TARGET))


def ensure_venv() -> None:
    if PY.exists():
        print("venv exists:", VENV)
        return
    run("uv", "venv", str(VENV), "--python", "3.12")


def install_packages() -> None:
    for spec, extra in PACKAGES:
        run("uv", "pip", "install", "--python", str(PY), spec, *extra)


def install_flashinfer() -> None:
    # flashinfer pins nccl4py (manylinux only), which is for multi-rank NCCL: install without deps, then its real
    # dependencies one by one (all of them have win_amd64 wheels)
    run("uv", "pip", "install", "--python", str(PY), "--no-deps", "flashinfer-python==0.6.17")
    # b12x pins nvidia-cutlass-dsl==4.7.1, whose libs publish no Windows wheels: Flash-Next serves its experts through
    # its own Triton/GPTQ path, so b12x stays uninstalled (Bolt and Nemotron would need it).
    run("uv", "pip", "install", "--python", str(PY), "--no-deps", "b12x==1.5.0", check=False)


# ------------------------------------------------------------------------------------------------- the third parties

FLASHINFER_JIT_PATCH = '''

# --- hibana Windows port: JitSpecNvcc builds through tvm_ffi's cross-platform ninja builder
if __import__("sys").platform == "win32":  # pragma: no cover - exercised on the RTX PRO 6000 box
    import os as _os
    import sys as _sys
    import sysconfig as _sysconfig

    import tvm_ffi as _tvm_ffi
    from tvm_ffi.cpp import extension as _tvm_ext
    import tvm_ffi.libinfo as _libinfo
    from . import env as _jit_env

    _KEEP = ("-D", "-gencode=", "--expt-relaxed-constexpr", "-Xfatbin", "-static-global-template-stub",
             "--use_fast_math", "-maxrregcount", "-Xptxas", "-lineinfo")

    def _win_includes():
        dirs = [_sysconfig.get_path("include"), _libinfo.find_include_path(), _libinfo.find_dlpack_include_path(),
                _jit_env.FLASHINFER_INCLUDE_DIR, _jit_env.FLASHINFER_CSRC_DIR, _jit_env.SPDLOG_INCLUDE_DIR]
        dirs += list(_jit_env.CCCL_INCLUDE_DIRS) + list(_jit_env.CUTLASS_INCLUDE_DIRS)
        out, seen = [], set()
        for d in dirs:
            p = str(Path(str(d)).resolve())
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _win_tune(flags):
        return [f for f in (flags or []) if any(f.startswith(k) for k in _KEEP)]

    def _win_lib_path(self):
        return Path(self.build_dir) / f"{self.name}.dll"

    def _win_is_compiled(self):
        return Path(self.build_dir, f"{self.name}.dll").exists()

    def _win_build(self, verbose=None, need_lock=False):
        build_dir = Path(self.build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        compat = str(Path(_os.environ.get("HIBANA_WIN_COMPAT", r"B:/hibana-win/win_compat.h")))
        includes = [f"/I{d}" for d in _win_includes()]
        cflags = ["/std:c++17", "/MD", "/O2", "/EHsc", "/DNDEBUG", f"/FI{compat}",
                  "-DPy_LIMITED_API=0x03090000"] + includes
        cuda_flags = ["-std=c++17", "-Xcompiler=/MD,/EHsc,/Zc:__cplusplus,/Zc:preprocessor",
                      "-allow-unsupported-compiler", f"--pre-include={compat}", "-DNDEBUG",
                      "-DPy_LIMITED_API=0x03090000"]
        cuda_flags += _win_tune(self.extra_cuda_cflags) + [f"-I{d}" for d in _win_includes()]
        cuda_lib = str(Path(_os.environ.get("CUDA_HOME") or r"C:/cuda13") / "lib" / "x64")
        logger.info("building %s on Windows through tvm_ffi (%d sources)", self.name, len(self.sources))
        _tvm_ext.build(name=self.name, sources=[str(s) for s in self.sources], extra_cflags=cflags,
                       extra_cuda_cflags=cuda_flags,
                       extra_ldflags=[f"/LIBPATH:{cuda_lib}", "cudart.lib", "cuda.lib"],
                       build_directory=str(build_dir), output=str(build_dir / f"{self.name}.dll"), backend="cuda")

    def _win_load(self, so_path=None):
        return _tvm_ffi.load_module(str(so_path or _win_lib_path(self)))

    JitSpecNvcc.jit_library_path = property(_win_lib_path)
    JitSpecNvcc.aot_path = property(lambda self: Path(self.build_dir) / f"{self.name}.dll")
    JitSpecNvcc.is_aot = property(lambda self: False)
    JitSpecNvcc.get_library_path = _win_lib_path
    JitSpecNvcc.is_compiled = property(_win_is_compiled)
    JitSpecNvcc.build = _win_build
    JitSpecNvcc.load = _win_load
'''

TVMFFI_PATCHES = [
    # drive letters in build lines
    ('''def build_ninja(build_dir: str) -> None:
    """Build the module in the given build directory using ninja."""
    command = ["ninja", "-v"]''',
     '''def _escape_win_build_paths(ninja_file: str) -> None:
    """Write "C:" back as "C$:" on build lines: ninja takes the first colon as the outputs' separator, so an unescaped
    drive letter reads as an empty rule name ("expected build command name"). $out/$in expansion turns it back."""
    path = Path(ninja_file)
    escaped = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("build ") or line.startswith("default "):
            line = re.sub(r"([A-Za-z]):(?=[\\\\/])", r"\\1$:", line)
        escaped.append(line)
    path.write_text("\\n".join(escaped) + "\\n", encoding="utf-8")


def build_ninja(build_dir: str) -> None:
    """Build the module in the given build directory using ninja."""
    if IS_WINDOWS:
        _escape_win_build_paths(os.path.join(build_dir, "build.ninja"))
    command = ["ninja", "-v"]'''),
    # one comma-separated -Xcompiler value
    ('default_cuda_cflags = ["-Xcompiler", "/std:c++17", "/O2"]',
     'default_cuda_cflags = ["-Xcompiler=/std:c++17,/O2"]  # nvcc takes one comma-separated value'),
    # pin the MSVC install (CUDA 13 rejects the 2026 toolset)
    ('''        # Find the Visual Studio installation path
        vs_install_path = subprocess.run(''',
     '''        # Find the Visual Studio installation path (HIBANA_MSVC_PATH pins one: CUDA 13 supports MSVC 2019-2022 and
        # nvcc's cudafe++ crashes on the 2026 toolset, and a box can carry both)
        vs_install_path = os.environ.get("HIBANA_MSVC_PATH", "").strip()
        if vs_install_path and not Path(vs_install_path).exists():
            vs_install_path = ""
        if vs_install_path:
            pass
        else:
          vs_install_path = subprocess.run('''),
]

HEADER_PATCHES = [
    ("flashinfer/data/include/flashinfer/norm/fused_qk_rmsnorm_rope.cuh",
     "int const token_idx_in_seq = tokenIdx % seq_len;",
     "int const token_idx_in_seq = static_cast<int>(tokenIdx) % seq_len;  // MSVC: int64 % IntFastDiv is ambiguous"),
]


def patch_flashinfer(site: Path) -> None:
    core = site / "flashinfer/jit/core.py"
    text = core.read_text(encoding="utf-8")
    marker = "# --- hibana Windows port: JitSpecNvcc builds through tvm_ffi"
    if marker not in text:
        core.write_text(text + FLASHINFER_JIT_PATCH, encoding="utf-8", newline="\n")
        print("patched", core)
    if "from __future__ import annotations" not in text:
        raise SystemExit("flashinfer/jit/core.py has no __future__ import: Path is used by the patch")
    # gcc's inline attribute in the vendored headers
    for path in (site / "flashinfer/data").rglob("*.cuh"):
        body = path.read_text(encoding="utf-8", errors="replace")
        if "__attribute__((always_inline))" in body:
            path.write_text(body.replace("__attribute__((always_inline))", "__forceinline__"),
                            encoding="utf-8", newline="\n")
            print("rewrote inline attribute in", path.name)
    for rel, old, new in HEADER_PATCHES:
        path = site / rel
        body = path.read_text(encoding="utf-8", errors="replace")
        if old in body:
            path.write_text(body.replace(old, new), encoding="utf-8", newline="\n")
            print("patched", path.name)


def patch_tvm_ffi(site: Path) -> None:
    path = site / "tvm_ffi/cpp/extension.py"
    text = path.read_text(encoding="utf-8")
    if "import re" not in text.split("def build_ninja")[0]:
        text = text.replace("import subprocess", "import re\nimport subprocess", 1)
    for old, new in TVMFFI_PATCHES:
        if new.splitlines()[0] in text:
            continue
        if old not in text:
            print("WARNING: tvm_ffi anchor missing:", old.splitlines()[0][:60])
            continue
        text = text.replace(old, new)
        print("patched tvm_ffi:", old.splitlines()[0][:50])
    path.write_text(text, encoding="utf-8", newline="\n")


def write_compat_header() -> None:
    target = CHECKOUT / "win_compat.h"
    if target.exists():
        return
    shutil.copyfile(ROOT / "win_compat.h", target)
    print("wrote", target)


def apply_checkout_patches() -> None:
    env = dict(os.environ, PYTHONUTF8="1")
    run(str(PY), str(CHECKOUT / "winpatch.py"), str(CHECKOUT), env=env)


def verify() -> int:
    env = dict(os.environ)
    env.update({"CUDA_HOME": str(CUDA_HOME), "CUDA_PATH": str(CUDA_HOME), "HIBANA_MSVC_PATH": str(MSVC),
                "HIBANA_WIN_COMPAT": str(CHECKOUT / "win_compat.h"),
                "PATH": f"{CUDA_HOME / 'bin'};{VENV / 'Scripts'};" + env.get("PATH", "")})
    code = ("import torch, flashinfer;"
            "x=torch.randn(2,8,64,dtype=torch.bfloat16,device='cuda');w=torch.ones(64,dtype=torch.bfloat16,device='cuda');"
            "y=flashinfer.norm.rmsnorm(x,w);torch.cuda.synchronize();"
            "print('flashinfer JIT ok', tuple(y.shape), float(y.abs().mean()))")
    result = subprocess.run([str(PY), "-c", code], env=env, capture_output=True, text=True)
    print(result.stdout[-2000:] or result.stderr[-3000:])
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="JIT smoke test only")
    args = parser.parse_args()
    site = VENV / "Lib" / "site-packages"
    if args.verify:
        return verify()
    ensure_cuda_junction()
    ensure_venv()
    install_packages()
    install_flashinfer()
    patch_flashinfer(site)
    patch_tvm_ffi(site)
    write_compat_header()
    apply_checkout_patches()
    print(f"\nsetup done. Serve with {CHECKOUT / 'run-flashnext.bat'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
