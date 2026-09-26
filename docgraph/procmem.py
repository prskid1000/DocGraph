"""Process memory helpers: give freed memory back to the OS, report the
process's working set, and map read-mostly arrays from files.

* trim()          gc + compact the C heap + shrink the working set. On
                  Windows: HeapCompact over the process heaps and
                  SetProcessWorkingSetSize(-1, -1) (pages go to the standby
                  list and fault back cheaply if touched again); on glibc:
                  malloc_trim(0). The biggest single win is torch: its CUDA
                  kernels and DLL images (~1.4 GB of working set after the
                  first encode) are touched once and never again.
* memory()        working set / private bytes of this process, in MB.
* mmap_npz(path)  the arrays of an uncompressed .npz as read-only memory
                  maps (file-backed: they cost working set only while hot
                  and are dropped by the OS first under pressure).
"""
from __future__ import annotations

import gc
import logging
import sys
import threading
import time
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

_lock = threading.Lock()
_last_trim = 0.0


def trim(reason: str = "", min_interval: float = 0.0) -> bool:
    """Collect garbage and return free memory to the OS. Rate-limited by
    `min_interval` seconds; returns whether it ran."""
    global _last_trim
    now = time.monotonic()
    if min_interval and now - _last_trim < min_interval:
        return False
    if not _lock.acquire(blocking=False):
        return False
    try:
        _last_trim = now
        gc.collect()
        if sys.platform == "win32":
            _trim_windows()
        else:
            _trim_posix()
        if reason:
            log.debug("memory trimmed (%s): %s", reason, memory())
        return True
    except Exception as exc:  # never fail a caller over housekeeping
        log.debug("memory trim failed: %s", exc)
        return False
    finally:
        _lock.release()


_DLLS: dict = {}


def _dll(name: str):
    """A private ctypes handle: argtypes set here never leak into other
    users of ctypes.windll.<name>."""
    d = _DLLS.get(name)
    if d is None:
        import ctypes
        d = _DLLS[name] = ctypes.WinDLL(name)
    return d


def _trim_windows() -> None:
    import ctypes
    from ctypes import wintypes
    k = _dll("kernel32")
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.GetProcessHeaps.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    k.GetProcessHeaps.restype = wintypes.DWORD
    k.HeapCompact.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.SetProcessWorkingSetSize.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t]
    n = k.GetProcessHeaps(0, None)
    if n:
        heaps = (wintypes.HANDLE * n)()
        n = k.GetProcessHeaps(n, heaps)
        for i in range(n):
            try:
                k.HeapCompact(heaps[i], 0)
            except Exception:
                pass
    k.SetProcessWorkingSetSize(k.GetCurrentProcess(), ctypes.c_size_t(-1), ctypes.c_size_t(-1))


def _trim_posix() -> None:
    import ctypes
    import ctypes.util
    name = ctypes.util.find_library("c")
    if not name:
        return
    libc = ctypes.CDLL(name)
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)


def memory() -> dict:
    """{"rss_mb", "private_mb"} of this process (0 when unknown)."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                            ("d", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]
            k = _dll("kernel32")
            k.GetCurrentProcess.restype = wintypes.HANDLE
            p = _dll("psapi")
            p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            m = _PMC()
            m.cb = ctypes.sizeof(m)
            p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(m), m.cb)
            return {"rss_mb": round(m.WorkingSetSize / 2 ** 20), "private_mb": round(m.PrivateUsage / 2 ** 20)}
        except Exception:
            return {"rss_mb": 0, "private_mb": 0}
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            d = dict(line.split(":", 1) for line in fh if ":" in line)
        return {"rss_mb": round(int(d["VmRSS"].split()[0]) / 1024),
                "private_mb": round(int(d.get("RssAnon", "0 kB").split()[0]) / 1024)}
    except Exception:
        return {"rss_mb": 0, "private_mb": 0}


def mmap_npz(path: Path) -> dict | None:
    """Arrays of an uncompressed (np.savez) archive as read-only memmaps;
    None when the file is compressed / unreadable (callers fall back to
    np.load)."""
    import numpy as np
    from numpy.lib import format as npf
    path = Path(path)
    out: dict = {}
    try:
        with zipfile.ZipFile(path) as zf, open(path, "rb") as fh:
            for info in zf.infolist():
                if info.compress_type != zipfile.ZIP_STORED or not info.filename.endswith(".npy"):
                    return None
                fh.seek(info.header_offset)
                local = fh.read(30)
                if local[:4] != b"PK\x03\x04":
                    return None
                n_name = int.from_bytes(local[26:28], "little")
                n_extra = int.from_bytes(local[28:30], "little")
                start = info.header_offset + 30 + n_name + n_extra
                fh.seek(start)
                version = npf.read_magic(fh)
                if version == (1, 0):
                    shape, fortran, dtype = npf.read_array_header_1_0(fh)
                else:
                    shape, fortran, dtype = npf.read_array_header_2_0(fh)
                if dtype.hasobject:
                    return None
                data_off = fh.tell()
                key = info.filename[:-4]
                n = int(np.prod(shape)) if shape else 1
                if n == 0:
                    out[key] = np.zeros(shape, dtype=dtype)
                else:
                    out[key] = np.memmap(path, dtype=dtype, mode="r", offset=data_off, shape=shape,
                                         order="F" if fortran else "C")
    except Exception:
        return None
    return out
