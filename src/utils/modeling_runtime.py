"""Bounded DuckDB connections and stage memory measurements."""
import ctypes
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
import duckdb

@contextmanager
def connection(runtime, work):
    temp = Path(runtime.get("temp_directory") or Path(work) / "duckdb_spill")
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(config={"memory_limit": runtime.get("memory_limit", "2GB"), "threads": int(runtime.get("threads", 2))})
    try:
        con.execute("SET temp_directory = ?", [str(temp.resolve())])
        con.execute("SET preserve_insertion_order=false")
        yield con
    finally:
        con.close()


def rss_bytes():
    if os.name == "nt":
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD),
                        *[(n, ctypes.c_size_t) for n in ("peak", "rss", "qpeak", "q", "ppeak", "p", "page", "pagepeak")]]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return int(counters.rss)
        return None
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None


class Measurement:
    """Current RSS samples from this stage, not lifetime process maximum."""
    def __enter__(self):
        self.started = time.perf_counter(); self.peak = rss_bytes()
        self.stop_event = threading.Event()
        def sample():
            while not self.stop_event.wait(0.2):
                current = rss_bytes()
                if current is not None:
                    self.peak = max(self.peak or 0, current)
        self.thread = threading.Thread(target=sample, daemon=True); self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set(); self.thread.join()
        current = rss_bytes()
        if current is not None:
            self.peak = max(self.peak or 0, current)
        self.seconds = time.perf_counter() - self.started

    def metrics(self):
        return {"duration_seconds": self.seconds, "peak_rss_bytes": self.peak}
