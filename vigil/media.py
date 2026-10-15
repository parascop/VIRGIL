"""Helpers for turning a stored event time back into something viewable:
frame previews and short browser-playable clips extracted from the source file."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("vigil.media")


def extract_frame(path: str | Path, t_s: float) -> Optional[np.ndarray]:
    """Frame at ``t_s`` seconds (BGR) or None if the file/time is unusable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def codec_fourcc(path: str | Path) -> str:
    cap = cv2.VideoCapture(str(path))
    try:
        v = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)).strip().lower() if v else ""
    finally:
        cap.release()


def browser_playable(path: str | Path) -> Optional[bool]:
    """True for H.264 in MP4 (plays in every browser), False for codecs that most
    browsers cannot decode (e.g. OpenCV's mp4v), None if unknown."""
    if Path(path).suffix.lower() not in (".mp4", ".m4v", ".mov", ".webm"):
        return False
    fourcc = codec_fourcc(path)
    if not fourcc:
        return None
    return fourcc in ("avc1", "h264", "x264", "vp80", "vp90", "vp09")


def extract_clip(src: str | Path, t0: float, t1: float, out: str | Path) -> Tuple[Optional[Path], str]:
    """Write ``src[t0:t1]`` to ``out``. Prefers ffmpeg/H.264 (browser playable);
    otherwise falls back to OpenCV mp4v (may not play in browsers).
    Returns (path | None, human readable note)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = max(0.0, t0)
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.2f}", "-i", str(src), "-t", f"{max(0.5, t1 - t0):.2f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(out)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out, "H.264 clip extracted with ffmpeg"
            log.warning("ffmpeg failed: %s", r.stderr[:200])
        except Exception as e:  # noqa: BLE001
            log.warning("ffmpeg error: %s", e)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return None, "source video could not be opened"
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
    n = 0
    while n < int((t1 - t0) * fps):
        ok, img = cap.read()
        if not ok:
            break
        vw.write(img)
        n += 1
    vw.release()
    cap.release()
    if n == 0:
        return None, "no frames in that time range"
    return out, "clip written with OpenCV (mp4v): some browsers cannot play this codec; install ffmpeg for H.264"
