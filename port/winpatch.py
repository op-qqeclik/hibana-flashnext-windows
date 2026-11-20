"""Apply the Windows port patches to a hibana checkout (idempotent).

    python winpatch.py B:\\hibana-win

Every patch names the file and checks its anchor, so re-running is safe and a drifted anchor fails loudly.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
PKG = ROOT / "hibana"
applied: list[str] = []
skipped: list[str] = []


def note(path: Path, what: str) -> None:
    applied.append(f"{path.relative_to(ROOT)}: {what}")


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text, encoding="utf-8", newline="\n")


def sub(rel: str, old: str, new: str, what: str, count: int = 1) -> None:
    text = read(rel)
    if new in text:
        skipped.append(f"{rel}: {what} (already)")
        return
    if old not in text:
        raise SystemExit(f"ANCHOR MISSING in {rel} for: {what}\n---\n{old[:400]}\n---")
    text = text.replace(old, new, count)
    write(rel, text)
    note(ROOT / rel, what)


def replace_function(rel: str, name: str, new_source: str, what: str | None = None) -> None:
    """Swap one top-level function's whole body (from its `def` to the line before the next top-level statement)."""
    text = read(rel)
    pattern = re.compile(rf"^def {re.escape(name)}\(.*?(?=^(?:def |class |@|[A-Za-z_#]|\Z))", re.S | re.M)
    match = pattern.search(text)
    if match is None:
        raise SystemExit(f"function {name} not found in {rel}")
    text = text[:match.start()] + new_source.rstrip("\n") + "\n\n\n" + text[match.end():]
    write(rel, text)
    note(ROOT / rel, what or f"rewrote {name}()")


# ------------------------------------------------------------------------------------------------ 0. the shim module
shutil.copyfile(HERE / "_plat.py", PKG / "_plat.py")
note(PKG / "_plat.py", "installed the platform shims")

# ------------------------------------------------------------------------------------------------ 1. hardware target
(ROOT / "hibana/hardware/gb10.py").write_text(read("hibana/hardware/gb10.py") + '''

# The Windows port serves the same engines on one discrete card, so the host probe accepts that card too: its compute
# capability is 12.0 (SM 120, the same kernel generation as the Spark's 12.1) and its memory is device memory, not the
# host's (the preflight checks VRAM instead of MemTotal when the card is discrete).
import os as _os
import sys as _sys

DISCRETE_GPU = _sys.platform == "win32"
ACCEPTED_GPUS = {GPU_NAME: COMPUTE_CAPABILITY,
                 "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": (12, 0)}
MIN_DEVICE_MEMORY_KIB = 60_000_000  # what a discrete card needs for the 3-bit Flash-Next profile (53 GiB resident)
UNIFIED_PHYSICAL_MEMORY = not DISCRETE_GPU
''', encoding="utf-8", newline="\n")
note(ROOT / "hibana/hardware/gb10.py", "accepted GPU table, discrete-memory flag")

# ------------------------------------------------------------------------------------------------ 2. ledger budget
replace_function("hibana/cache/ledger.py", "mem_available_bytes", '''def mem_available_bytes() -> int:
    """What the KV and state pools may use: the host's MemAvailable on a Spark (its GPU allocates from it), the
    device's free memory on a discrete card, where the weights are already resident (`hibana._plat`)."""
    from hibana._plat import pool_available_bytes

    return pool_available_bytes()''')

sub("hibana/cache/ledger.py", '''def host_floor_bytes() -> int:''', '''def host_floor_bytes() -> int:''', "floor helper kept")

# ------------------------------------------------------------------------------------------ 3. launcher preflight
replace_function("hibana/launch/preflight.py", "meminfo", '''def meminfo() -> dict[str, int]:
    from hibana._plat import meminfo as _meminfo

    return _meminfo()''')

replace_function("hibana/launch/preflight.py", "check_host", '''def check_host(floor_kib: int | None = None) -> dict[str, str]:
    """The host can serve this target: one supported NVIDIA GPU with a recent driver, the memory the load needs above
    the floor (`floor_kib`, default `guard.floor_kib()`), CUDA's nvcc for the first-use kernel builds, and the free
    disk for the model. On a Spark the budget is host MemAvailable; on the Windows port's discrete card it is the
    card's free memory and its total, since the pools and the weights live there."""
    from hibana._plat import IS_WINDOWS, load_headroom_bytes
    from hibana.hardware.gb10 import ACCEPTED_GPUS, MIN_DEVICE_MEMORY_KIB

    need = LOAD_PEAK_KIB + (guard.floor_kib() if floor_kib is None else floor_kib)
    if not IS_WINDOWS and platform.machine() != "aarch64":
        raise PreflightError("host architecture must be aarch64 (a DGX Spark)")
    if shutil.which("nvidia-smi") is None:
        raise PreflightError("nvidia-smi is not installed")
    rows = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip().splitlines()
    if len(rows) != 1:
        raise PreflightError("exactly one NVIDIA GPU is required")
    name, cc, driver, vram_mib = (x.strip() for x in rows[0].split(","))
    if (name, cc) not in {(k, ".".join(str(p) for p in v)) for k, v in ACCEPTED_GPUS.items()}:
        raise PreflightError(f"GPU must be one of {sorted(ACCEPTED_GPUS)}, found {name} {cc}")
    if int(driver.split(".")[0]) < 580:
        raise PreflightError("NVIDIA driver must be 580 or newer")
    mem = meminfo()
    available = load_headroom_bytes()
    if IS_WINDOWS:
        device_kib = int(float(vram_mib)) * 1024
        if device_kib < MIN_DEVICE_MEMORY_KIB:
            raise PreflightError(f"the card needs at least {MIN_DEVICE_MEMORY_KIB // 2**20} GiB of memory for this profile")
        if available < need * 1024:
            raise PreflightError(f"at least {need // 2**20} GiB of free device memory is required before the model loads "
                                 f"(have {available // 2**30} GiB); something else is using the GPU")
    else:
        if mem["MemTotal"] < 120_000_000:
            raise PreflightError("at least 120 GB host memory is required")
        if mem["MemAvailable"] < need:
            raise PreflightError(f"at least {need // 2**20} GiB must be available before the model loads (have {mem['MemAvailable'] // 2**20} GiB); "
                                 "something else is using the memory")
    if not cuda_nvcc():
        raise PreflightError("the CUDA toolkit's nvcc is missing (it builds the kernels on first use): install CUDA 13, "
                             "or point CUDA_HOME at it")
    return {"gpu": f"{name} (SM {cc}), driver {driver}",
            "memory": (f"{int(float(vram_mib)) // 1024} GiB device, {available / 2**30:.1f} GiB free" if IS_WINDOWS
                       else f"{mem['MemTotal'] // 2**20} GiB total, {mem['MemAvailable'] // 2**20} GiB available")}''')

sub("hibana/launch/preflight.py", '''HEADROOM = 10 * 2**30''', '''HEADROOM = 10 * 2**30''', "headroom kept")

# a cross-platform nvcc probe next to HEADROOM
sub("hibana/launch/preflight.py", '''class PreflightError(RuntimeError):
    pass''', '''class PreflightError(RuntimeError):
    pass


def cuda_nvcc() -> str | None:
    """nvcc for the first-use kernel builds: the toolkit on PATH, else $CUDA_HOME, else the Spark's /usr/local/cuda."""
    found = shutil.which("nvcc")
    if found:
        return found
    for candidate in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if candidate and (Path(candidate) / "bin" / "nvcc.exe").is_file():
            return str(Path(candidate) / "bin" / "nvcc.exe")
        if candidate and (Path(candidate) / "bin" / "nvcc").is_file():
            return str(Path(candidate) / "bin" / "nvcc")
    return None''', "nvcc probe")

# ----------------------------------------------------------------------------------------------- 4. memory guard
replace_function("hibana/launch/guard.py", "available_kib", '''def available_kib() -> int:
    from hibana._plat import meminfo

    return meminfo()["MemAvailable"]''')

replace_function("hibana/launch/guard.py", "_stop_self", '''def _stop_self() -> None:
    from hibana._plat import terminate_self

    terminate_self()
    time.sleep(15)
    os._exit(1)''')

sub("hibana/launch/guard.py", '''    def stop() -> None:
        try:
            os.killpg(os.getpgid(args.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(15)
        if alive():
            os.kill(args.pid, signal.SIGKILL)''', '''    def stop() -> None:
        from hibana._plat import IS_WINDOWS, kill_tree

        if IS_WINDOWS:
            kill_tree(args.pid)
            return
        try:
            os.killpg(os.getpgid(args.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        time.sleep(15)
        if alive():
            os.kill(args.pid, signal.SIGKILL)''', "kill_tree on Windows")

sub("hibana/launch/guard.py", '''    def alive() -> bool:
        try:
            os.kill(args.pid, 0)
            return True
        except ProcessLookupError:
            return False''', '''    def alive() -> bool:
        from hibana._plat import IS_WINDOWS

        if IS_WINDOWS:
            import subprocess as _sp
            out = _sp.run(["tasklist", "/FI", f"PID eq {args.pid}", "/NH"], capture_output=True, text=True).stdout
            return str(args.pid) in out
        try:
            os.kill(args.pid, 0)
            return True
        except ProcessLookupError:
            return False''', "tasklist liveness on Windows")

# -------------------------------------------------------------------------------------------- 5. server memory gem
replace_function("hibana/api/server.py", "_memory_gib", '''def _memory_gib() -> dict:
    """What the process holds beyond the ledger's pools: PyTorch's allocated and reserved bytes, now and at their peak
    (reserved only grows: freed blocks stay cached), and the process's resident host memory."""
    import torch
    from hibana._plat import rss_bytes

    gib = lambda v: round(v / 2**30, 2)  # noqa: E731
    return {"allocated": gib(torch.cuda.memory_allocated()), "reserved": gib(torch.cuda.memory_reserved()),
            "max_allocated": gib(torch.cuda.max_memory_allocated()), "max_reserved": gib(torch.cuda.max_memory_reserved()),
            "rss": gib(rss_bytes())}''')

# ---------------------------------------------------------------------------------------------- 6. SSD tier locks
sub("hibana/cache/ssd_store.py", "import fcntl\n", "", "drop the fcntl import")
# the store's flock is rewritten by fix_ssd.py (it sits inside a try/except whose indentation matters)

# ----------------------------------------------------------------------------------------- 7. free-space reserve
sub("hibana/engine/executor.py", '''        fs = os.statvfs(store.base)
        reserve = min(fs.f_blocks * fs.f_frsize // 10, 64 * GIB)
        room = tier.used + max(0, fs.f_bavail * fs.f_frsize - reserve)''', '''        from hibana._plat import disk_usage

        total, free, usable = disk_usage(store.base)
        reserve = min(total // 10, 64 * GIB)
        room = tier.used + max(0, usable - reserve)''', "shutil.disk_usage instead of statvfs")

# ------------------------------------------------------------------------------------------------ 8. target check
sub("hibana/target.py", '''        if os.uname().machine != "aarch64":
            raise RuntimeError("host architecture must be aarch64")''', '''        from hibana._plat import IS_WINDOWS, meminfo

        if not IS_WINDOWS and os.uname().machine != "aarch64":
            raise RuntimeError("host architecture must be aarch64")''', "arch check only off Windows")

sub("hibana/target.py", '''        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            mem[key] = int(rest.split()[0])''', '''        mem = meminfo()''', "meminfo from the shim")

sub("hibana/target.py", '''        if mem["MemTotal"] < MIN_TOTAL_MEMORY_KIB:
            raise RuntimeError("at least 120 GB host memory is required")''', '''        from hibana.hardware.gb10 import DISCRETE_GPU

        if not DISCRETE_GPU and mem["MemTotal"] < MIN_TOTAL_MEMORY_KIB:
            raise RuntimeError("at least 120 GB host memory is required")''', "host memory only on a Spark")

sub("hibana/target.py", '''            if name != GPU_NAME or cap != COMPUTE_CAPABILITY:
                raise RuntimeError(f"GPU must be {GPU_NAME} SM{COMPUTE_CAPABILITY}, found {name} {cap}")''', '''            from hibana.hardware.gb10 import ACCEPTED_GPUS

            if (name, cap) not in ACCEPTED_GPUS.items():
                raise RuntimeError(f"GPU must be one of {sorted(ACCEPTED_GPUS)}, found {name} SM{cap}")''', "accepted GPUs")

# -------------------------------------------------------------------------------------------- 9. PLE table on Windows
sub("hibana/model/flashnext/ple.py", "_PAGE = mmap.PAGESIZE", "_PAGE = getattr(mmap, \"PAGESIZE\", 4096)", "page size without PAGESIZE")
sub("hibana/model/flashnext/ple.py", '''        for m in self.maps:
            m._mmap.madvise(mmap.MADV_RANDOM)  # a fault reads its own page, not a readahead window of neighbours''', '''        from hibana._plat import madvise_random

        for m in self.maps:
            madvise_random(m._mmap)  # a fault reads its own page, not a readahead window of neighbours''', "madvise via the shim")
sub("hibana/model/flashnext/ple.py", '''            self.maps.append(np.memmap(snap / file, dtype=np.uint8, mode="r", offset=base + begin, shape=(count, ROW_BYTES)))
            self.fds.append(os.open(snap / file, os.O_RDONLY))''', '''            self.maps.append(np.memmap(snap / file, dtype=np.uint8, mode="r", offset=base + begin, shape=(count, ROW_BYTES)))
            self.paths.append(snap / file)
            self.fds.append(os.open(snap / file, os.O_RDONLY | getattr(os, "O_BINARY", 0)))''', "keep shard paths")
sub("hibana/model/flashnext/ple.py", '''        self.maps: list[np.memmap] = []''', '''        self.maps: list[np.memmap] = []
        self.paths: list[Path] = []''', "paths field")
sub("hibana/model/flashnext/ple.py", '''        if n:
            shard = np.searchsorted(self.starts, flat, side="right") - 1''', '''        from hibana._plat import IS_WINDOWS

        if n and IS_WINDOWS:
            return self._gather_threads(flat, out, reshape=rows.shape)
        if n:
            shard = np.searchsorted(self.starts, flat, side="right") - 1''', "Windows gather route")
sub("hibana/model/flashnext/ple.py", '''    def prefetch(self, rows: np.ndarray) -> None:''', '''    def _gather_threads(self, flat: np.ndarray, out: np.ndarray, reshape=None) -> np.ndarray:
        """Windows: the same rows, read by a thread pool with one handle per thread (`preadv2` has no equivalent, and a
        page-faulting memmap gather would serialise on the fault)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        shard = np.searchsorted(self.starts, flat, side="right") - 1
        local = flat - self.starts[shard]
        starts = np.asarray(self.starts, dtype=np.int64)
        offsets = np.asarray(self.offsets, dtype=np.int64)
        local_handle = threading.local()

        def one(i: int) -> None:
            s = int(shard[i])
            handles = getattr(local_handle, "handles", None)
            if handles is None:
                handles = local_handle.handles = {}
            handle = handles.get(s)
            if handle is None:
                handle = handles[s] = open(self.paths[s], "rb", buffering=0)
            handle.seek(int(offsets[s] + local[i] * ROW_BYTES))
            data = handle.read(ROW_BYTES)
            if len(data) != ROW_BYTES:
                raise OSError(f"PLE shard {s}: short read at row {int(local[i])}")
            out[i, :] = np.frombuffer(data, dtype=np.uint8)
            self.misses += 1

        with ThreadPoolExecutor(max_workers=32, thread_name_prefix="ple-gather") as pool:
            list(pool.map(one, range(flat.shape[0])))
        out = out.reshape(*flat.shape, ROW_BYTES)
        return out.reshape(*reshape, ROW_BYTES) if reshape is not None else out

    def prefetch(self, rows: np.ndarray) -> None:''', "threaded gather for Windows")
sub("hibana/model/flashnext/ple.py", '''        fds, advise, page, need = self.fds, os.posix_fadvise, _PAGE, os.POSIX_FADV_WILLNEED
        for s, offset in keys.tolist():
            advise(fds[s], offset, page, need)''', '''        from hibana._plat import willneed

        fds, page = self.fds, _PAGE
        for s, offset in keys.tolist():
            willneed(fds[s], offset, page)''', "fadvise WILLNEED via the shim")
sub("hibana/model/flashnext/ple.py", '''            fd, start = self.fds[s], self.offsets[s] + (a - lo) * ROW_BYTES
            os.posix_fadvise(fd, start, (b - a) * ROW_BYTES, os.POSIX_FADV_SEQUENTIAL)''', '''            from hibana._plat import dontneed, pread, sequential

            fd, start = self.fds[s], self.offsets[s] + (a - lo) * ROW_BYTES
            sequential(fd, start, (b - a) * ROW_BYTES)''', "fadvise SEQUENTIAL via the shim")
sub("hibana/model/flashnext/ple.py", '''                    r = os.preadv(fd, [flat[got:want]], start + (i - a) * ROW_BYTES + got)
                    if r <= 0:
                        raise OSError(f"PLE shard {s}: short read at row {i}")
                    got += r''', '''                    chunk = pread(fd, want - got, start + (i - a) * ROW_BYTES + got)
                    if not chunk:
                        raise OSError(f"PLE shard {s}: short read at row {i}")
                    flat[got:want] = chunk
                    got += len(chunk)''', "pread via the shim")
sub("hibana/model/flashnext/ple.py", '''            os.posix_fadvise(fd, start, (b - a) * ROW_BYTES, os.POSIX_FADV_DONTNEED)''', '''            dontneed(fd, start, (b - a) * ROW_BYTES)''', "fadvise DONTNEED via the shim")

# -------------------------------------------------------------------------------------- 10. engine runtime env
sub("hibana/launch/run.py", '''    venv_bin = str(Path(sys.executable).parent)
    path = os.environ.get("PATH", "/usr/bin:/bin")
    return {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"), "PATH": f"{venv_bin}:/usr/local/cuda/bin:{path}"}''', '''    from hibana._plat import IS_WINDOWS

    venv_bin = Path(sys.executable).parent
    sep = ";" if IS_WINDOWS else ":"
    if IS_WINDOWS:
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if not cuda_home:
            roots = sorted(Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v13*"), reverse=True)
            cuda_home = str(roots[0]) if roots else r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"
        path = os.environ.get("PATH", r"C:/Windows/System32")
    else:
        cuda_home, path = os.environ.get("CUDA_HOME", "/usr/local/cuda"), os.environ.get("PATH", "/usr/bin:/bin")
    return {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "CUDA_HOME": cuda_home, "PATH": sep.join([str(venv_bin), str(Path(cuda_home) / "bin"), path])}''', "Windows CUDA_HOME and PATH")

print("\n".join(f"  patched {a}" for a in applied) if applied else "  nothing patched")
if skipped:
    print("\n".join(f"  kept    {s}" for s in skipped))
