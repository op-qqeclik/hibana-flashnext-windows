"""Platform shims for running the engine off DGX OS (the Windows port).

The engine was written for one DGX Spark: aarch64, /proc, POSIX file I/O and one physical memory shared by the host
and the GPU. Everything the port needs to differ is here, so the engine's own code keeps its Linux paths:

* `meminfo()`      MemTotal / MemAvailable in KiB (``/proc/meminfo`` on Linux, ``GlobalMemoryStatusEx`` on Windows).
* `pool_available_bytes()` what the KV pages and state slots may use: host MemAvailable on a Spark (unified), the GPU's
  free device memory on a discrete card (the weights are already resident there).
* `load_headroom_bytes()` what a weight load may consume: the same number, measured before the load.
* `disk_usage()`  ``os.statvfs`` / ``shutil.disk_usage`` as one tuple.
* `rss_bytes()`   the process's resident size.
* `flock()/funlock()`, `fadvise()`, `madvise_random()`, `pread_rows()`: the POSIX calls the SSD tier, the PLE table and
  the weight reader make, replaced (or dropped) on Windows.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
KIB = 1024


# --------------------------------------------------------------------------------------------------------------- memory

def _meminfo_windows() -> dict[str, int]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    # Windows' commit limit is what a large allocation actually runs into; report it as MemAvailable's twin so the
    # guard sees the same kind of number a Spark's MemAvailable is.
    return {"MemTotal": status.ullTotalPhys // KIB, "MemFree": status.ullAvailPhys // KIB,
            "MemAvailable": status.ullAvailPhys // KIB,
            "CommitLimit": status.ullTotalPageFile // KIB, "CommitAvailable": status.ullAvailPageFile // KIB}


def meminfo() -> dict[str, int]:
    """MemTotal and MemAvailable (KiB), the keys the engine reads; Linux from /proc/meminfo, Windows from the API."""
    if not IS_WINDOWS:
        out: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key] = int(rest.split()[0])
        return out
    return _meminfo_windows()


def _cuda_free_bytes() -> int | None:
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.mem_get_info()[0])
    except Exception:
        return None


def pool_available_bytes() -> int:
    """What the pools may take. On a Spark the GPU allocates from host memory, so MemAvailable is the budget; on a
    discrete card the pools live in device memory, so it is the free device memory after the weights are resident."""
    if IS_WINDOWS:
        free = _cuda_free_bytes()
        if free is not None:
            return free
    return meminfo()["MemAvailable"] * KIB


def load_headroom_bytes() -> int:
    """The budget a weight load is checked against, measured before it runs."""
    if IS_WINDOWS:
        free = _cuda_free_bytes()
        if free is not None:
            return free
    return meminfo()["MemAvailable"] * KIB


def rss_bytes() -> int:
    """The process's resident memory (the PLE table's page cache is not part of it)."""
    if not IS_WINDOWS:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        return 0
    return int(counters.WorkingSetSize)


def disk_usage(path) -> tuple[int, int, int]:
    """(total, free-for-all, free-for-this-user) bytes, the fields `os.statvfs` callers read, as f_blocks/f_frsize."""
    usage = shutil.disk_usage(str(path))
    return usage.total, usage.free, usage.free


def cpu_count() -> int:
    return os.cpu_count() or 1


# -------------------------------------------------------------------------------------------------------- POSIX calls

def fadvise(fd: int, offset: int, length: int, advice: int | None = None) -> None:
    """``os.posix_fadvise`` where it exists (WILLNEED / DONTNEED are the only advice the engine passes)."""
    advise = getattr(os, "posix_fadvise", None)
    if advise is None:
        return
    try:
        advise(fd, offset, length, advice if advice is not None else 0)
    except OSError:
        pass


def willneed(fd: int, offset: int, length: int) -> None:
    fadvise(fd, offset, length, getattr(os, "POSIX_FADV_WILLNEED", 3))


def dontneed(fd: int, offset: int = 0, length: int = 0) -> None:
    fadvise(fd, offset, length, getattr(os, "POSIX_FADV_DONTNEED", 4))


def sequential(fd: int, offset: int, length: int) -> None:
    fadvise(fd, offset, length, getattr(os, "POSIX_FADV_SEQUENTIAL", 2))


def madvise_random(mapping) -> None:
    """Ask for one page per fault instead of a readahead window; a no-op where mmap has no madvise."""
    advise = getattr(mapping, "madvise", None)
    if advise is None:
        return
    try:
        import mmap
        advise(getattr(mmap, "MADV_RANDOM", 1))
    except (OSError, ValueError):
        pass


def page_size() -> int:
    return getattr(__import__("mmap"), "PAGESIZE", 4096)


def flock_exclusive(fd: int) -> bool:
    """A non-blocking exclusive lock on `fd`; True when it was taken."""
    if not IS_WINDOWS:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    import msvcrt
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return True
    except OSError:
        return False


def funlock(fd: int) -> None:
    if not IS_WINDOWS:
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        return
    import msvcrt
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def pread(fd: int, nbytes: int, offset: int) -> bytes:
    """``os.pread``: one read at an offset without moving the file pointer."""
    if not IS_WINDOWS:
        return os.pread(fd, nbytes, offset)
    import msvcrt
    handle = msvcrt.get_osfhandle(fd)
    import ctypes
    from ctypes import wintypes
    buf = ctypes.create_string_buffer(nbytes)
    done = wintypes.DWORD(0)
    overlapped = ctypes.c_ulonglong(offset)
    ok = ctypes.windll.kernel32.ReadFile(ctypes.c_void_p(handle), buf, nbytes, ctypes.byref(done),
                                         ctypes.byref(overlapped))
    if not ok and ctypes.get_last_error() != 0:
        raise OSError(ctypes.get_last_error(), "ReadFile")
    return buf.raw[:done.value]


def terminate_self(sig_delivery: float = 0.0) -> None:
    """Stop this process the way SIGTERM would: Windows has no signals, so ask the server to drain, else exit."""
    if not IS_WINDOWS:
        import signal
        os.kill(os.getpid(), signal.SIGTERM)
        return
    import ctypes
    ctypes.windll.kernel32.TerminateProcess(ctypes.windll.kernel32.GetCurrentProcess(), 1)


def kill_tree(pid: int) -> None:
    """Stop a process and its children (SIGTERM to the process group on Linux, taskkill on Windows)."""
    import subprocess
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    import signal
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
