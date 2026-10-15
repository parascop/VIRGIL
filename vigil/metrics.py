"""Thread-safe runtime metrics: stage latencies (p50/p95), counters, gauges, FPS,
memory probes. Warm-up/model-load times are recorded separately from
steady-state latencies; unavailable hardware counters return ``None`` rather
than a guess."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

from .util import percentile

STAGES = ("ingest_to_detect", "detect", "trigger_to_vlm_start", "vlm", "embed", "store", "trigger_to_store")


class Metrics:
    def __init__(self, window: int = 2000):
        self._lock = threading.Lock()
        self._lat: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, Any] = {}
        self.warmup: Dict[str, float] = {}  # model load / warm-up seconds (NOT steady state)
        self._frame_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self.errors: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.peak_rss_mb = 0.0

    def observe(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._lat[stage].append(float(seconds))

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self.counters[name] += n

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            self.gauges[name] = value

    def frame_processed(self, camera_id: str) -> None:
        with self._lock:
            self._frame_times[camera_id].append(time.monotonic())

    def fps(self, camera_id: str) -> float:
        with self._lock:
            ts = list(self._frame_times[camera_id])
        if len(ts) < 2 or ts[-1] - ts[0] <= 0:
            return 0.0
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def error(self, source: str, message: str) -> None:
        with self._lock:
            self.errors.append({"t": time.time(), "source": source, "message": message[:300]})
            self.counters["errors"] += 1

    def stage_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        with self._lock:
            lat = {k: list(v) for k, v in self._lat.items()}
        out = {}
        for k, v in lat.items():
            out[k] = {"n": len(v), "p50_ms": _ms(percentile(v, 50)), "p95_ms": _ms(percentile(v, 95)),
                      "max_ms": _ms(max(v)) if v else None, "last_ms": _ms(v[-1]) if v else None}
        return out

    def sample_memory(self) -> Dict[str, Optional[float]]:
        rss = process_rss_mb()
        if rss:
            self.peak_rss_mb = max(self.peak_rss_mb, rss)
        return {"rss_mb": rss, "peak_rss_mb": self.peak_rss_mb or None, **gpu_memory()}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters, gauges, warm = dict(self.counters), dict(self.gauges), dict(self.warmup)
            errors = list(self.errors)
        return {"counters": counters, "gauges": gauges, "warmup_s": warm, "stages": self.stage_stats(),
                "memory": self.sample_memory(), "errors": errors}


def _ms(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v * 1000.0, 2)


def process_rss_mb() -> Optional[float]:
    """Resident set size of this process in MB (None if not measurable)."""
    try:
        import psutil  # optional

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        if sys.platform == "darwin":
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes on macOS (peak)
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 1e6
    except Exception:  # noqa: BLE001
        pass
    return None


def gpu_memory() -> Dict[str, Optional[float]]:
    """GPU memory facts. ``torch_*`` is this process only; ``smi_*`` (NVIDIA)
    is whole-GPU use including Ollama. All None when not measurable."""
    out: Dict[str, Optional[float]] = {"torch_peak_mb": None, "smi_used_mb": None, "smi_total_mb": None}
    try:
        import torch

        if torch.cuda.is_available():
            out["torch_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out["torch_peak_mb"] = torch.mps.driver_allocated_memory() / 1e6
    except Exception:  # noqa: BLE001
        pass
    used, total = nvidia_smi_memory()
    out["smi_used_mb"], out["smi_total_mb"] = used, total
    return out


def nvidia_smi_memory() -> tuple:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            used, total = r.stdout.strip().splitlines()[0].split(",")
            return float(used), float(total)
    except Exception:  # noqa: BLE001
        pass
    return None, None
