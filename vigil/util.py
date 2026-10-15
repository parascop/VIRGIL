"""Small shared helpers: atomic file writes, timezone formatting, percentiles."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional
from zoneinfo import ZoneInfo

import numpy as np


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write via a temp file in the same directory, then ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=1, default=str).encode("utf-8"))


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def atomic_write_image(path: str | Path, image: np.ndarray, quality: int = 85) -> None:
    atomic_write_bytes(path, encode_jpeg(image, quality))


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100); None for empty input."""
    vals: List[float] = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def get_tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def fmt_ts(epoch: float, tz_name: str = "UTC", fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.fromtimestamp(float(epoch), tz=ZoneInfo(tz_name)).strftime(fmt)


def setup_logging(level: str = "INFO", debug: bool = False) -> None:
    import logging

    logging.basicConfig(
        level=logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s",
    )
