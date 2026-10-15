"""Hardware detection, profile auto-selection and device resolution.

Ollama allocates VLM memory on its own: the PyTorch device chosen here only
controls YOLO and CLIP. Ollama placement is *observed* (``/api/ps``), never
assumed from the PyTorch result.
"""
from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("vigil.device")


def _run(cmd: list, timeout: float = 5.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def total_ram_mb() -> Optional[float]:
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024.0
        if sys.platform == "darwin":
            v = _run(["sysctl", "-n", "hw.memsize"])
            return int(v) / 1e6 if v else None
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 1e6
    except Exception:  # noqa: BLE001
        pass
    return None


def detect_hardware() -> Dict[str, Any]:
    """Collect facts. Missing components are reported, never assumed."""
    hw: Dict[str, Any] = {
        "os": platform.system(), "os_release": platform.release(), "os_version": platform.version()[:80],
        "python": platform.python_version(), "python_arch": platform.architecture()[0],
        "machine": platform.machine(), "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "ram_mb": total_ram_mb(),
        "torch": None, "cuda_available": False, "cuda_runtime": None, "mps_built": False,
        "mps_available": False, "gpu_name": None, "gpu_vram_mb": None, "torch_error": None,
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "nvidia_smi": None,
    }
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if smi:
        name, mem, drv = [x.strip() for x in smi.splitlines()[0].split(",")]
        hw["nvidia_smi"] = {"name": name, "vram_mb": float(mem), "driver": drv}
    try:
        import torch

        hw["torch"] = torch.__version__
        hw["cuda_available"] = bool(torch.cuda.is_available())
        hw["cuda_runtime"] = torch.version.cuda
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            hw["mps_built"] = bool(mps.is_built())
            hw["mps_available"] = bool(mps.is_available())
        if hw["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            hw["gpu_name"] = props.name
            hw["gpu_vram_mb"] = props.total_memory / 1e6
        elif hw["mps_available"]:
            hw["gpu_name"] = "Apple GPU (MPS)"
            hw["gpu_vram_mb"] = hw["ram_mb"]  # unified memory
    except Exception as e:  # noqa: BLE001
        hw["torch_error"] = f"{type(e).__name__}: {e}"
    if hw["gpu_name"] is None and hw["nvidia_smi"]:
        hw["gpu_name"] = hw["nvidia_smi"]["name"]
        hw["gpu_vram_mb"] = hw["nvidia_smi"]["vram_mb"]
    return hw


def select_profile(hw: Dict[str, Any], configs_dir: Path) -> Tuple[Path, str]:
    """Pick a known profile for the hardware; returns (path, reason)."""
    gpu = (hw.get("gpu_name") or "").lower()
    if "2050" in gpu:
        return configs_dir / "profile_rtx2050.yaml", f"GPU '{hw['gpu_name']}' matches RTX 2050"
    if "1650" in gpu:
        return configs_dir / "profile_gtx1650.yaml", f"GPU '{hw['gpu_name']}' matches GTX 1650"
    if hw.get("apple_silicon"):
        return configs_dir / "profile_mac_m4.yaml", "Apple Silicon detected"
    if gpu:
        return configs_dir / "profile_gtx1650.yaml", f"unknown GPU '{hw['gpu_name']}': using the conservative GTX 1650 profile"
    return configs_dir / "profile_gtx1650.yaml", "no GPU detected: using the conservative profile with CPU fallback"


def probe_device(device: str) -> Tuple[bool, str]:
    """Run a tiny tensor op to confirm the device really works."""
    try:
        import torch

        x = torch.ones(8, 8, device=device)
        float((x @ x).sum().item())
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def resolve_device(preference: str, hw: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """auto = cuda -> mps -> cpu; explicit preferences are probed and fall back
    to CPU with a reason instead of crashing later with incompatible state."""
    hw = hw or detect_hardware()
    pref = (preference or "auto").lower()
    if pref == "auto":
        order = (["cuda:0"] if hw["cuda_available"] else []) + (["mps"] if hw["mps_available"] else []) + ["cpu"]
    else:
        order = [pref, "cpu"]
    for dev in order:
        if dev == "cpu":
            return "cpu", ("requested cpu" if pref == "cpu" else f"fell back to cpu (wanted {pref})")
        ok, why = probe_device(dev)
        if ok:
            return dev, f"{dev} probe ok"
        log.warning("device %s unusable: %s", dev, why)
    return "cpu", "fallback"
