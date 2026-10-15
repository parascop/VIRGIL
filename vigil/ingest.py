"""Video ingestion: files, webcams and RTSP/network streams.

* One ``CameraSource`` thread per camera, single time-based sampling point
  (``target_fps``), bounded drop-oldest output queue, bounded per-camera ring
  buffer (~4 s of downscaled frames), RTSP/webcam reconnect with exponential
  backoff, clean shutdown via a stop event.
* Files support realtime pacing (``speed=1``), faster-than-real-time pacing
  (``speed=N``) and unpaced offline processing (``speed=0``; the queue then
  blocks instead of dropping so no frames are lost for evaluation).
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .config import CameraConfig
from .types import Frame

# TCP transport is far more reliable than UDP for RTSP over Wi-Fi.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402

log = logging.getLogger("vigil.ingest")


# --------------------------------------------------------------- primitives
class DropOldestQueue:
    """Thread-safe bounded queue.

    ``overflow="drop_oldest"`` evicts the oldest item to admit a new one and
    counts it; ``overflow="block"`` applies back-pressure (used for offline
    file processing so nothing is lost).
    """

    def __init__(self, maxsize: int = 30, overflow: str = "drop_oldest"):
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self.overflow = overflow
        self._dq: Deque[Any] = deque()
        self._cv = threading.Condition()
        self.dropped = 0
        self.max_depth = 0

    def put(self, item: Any, stop: Optional[threading.Event] = None) -> int:
        """Returns number of items dropped to make room (0 or 1)."""
        with self._cv:
            dropped = 0
            if len(self._dq) >= self.maxsize:
                if self.overflow == "block":
                    while len(self._dq) >= self.maxsize:
                        if stop is not None and stop.is_set():
                            return 0
                        self._cv.wait(timeout=0.1)
                else:
                    self._dq.popleft()
                    self.dropped += 1
                    dropped = 1
            self._dq.append(item)
            self.max_depth = max(self.max_depth, len(self._dq))
            self._cv.notify_all()
            return dropped

    def get(self, timeout: float = 0.0) -> Optional[Any]:
        with self._cv:
            if not self._dq and timeout > 0:
                self._cv.wait(timeout=timeout)
            if not self._dq:
                return None
            item = self._dq.popleft()
            self._cv.notify_all()
            return item

    def qsize(self) -> int:
        with self._cv:
            return len(self._dq)

    def clear(self) -> None:
        with self._cv:
            self._dq.clear()
            self._cv.notify_all()


class RingBuffer:
    """Keeps roughly the last ``seconds`` of (downscaled) frames, bounded both
    by time and by ``max_frames`` so memory can never grow without limit."""

    def __init__(self, seconds: float = 4.0, fps: float = 10.0, max_side: int = 640):
        self.seconds = seconds
        self.max_frames = max(4, int(math.ceil(seconds * fps * 1.5)) + 2)
        self.max_side = max_side
        self._frames: Deque[Frame] = deque()
        self._lock = threading.Lock()

    def add(self, frame: Frame) -> None:
        img = frame.image
        h, w = img.shape[:2]
        s = self.max_side / max(h, w)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
        f = Frame(camera_id=frame.camera_id, index=frame.index, t=frame.t, image=small,
                  t_media=frame.t_media, t_captured=frame.t_captured)
        with self._lock:
            self._frames.append(f)
            while len(self._frames) > self.max_frames or (self._frames and f.t - self._frames[0].t > self.seconds):
                self._frames.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def pick(self, t_end: float, span_s: float, n: int) -> List[Frame]:
        """n frames evenly spaced over [t_end - span_s, t_end] (nearest frame to
        each target time, last one is the trigger frame), chronological, unique."""
        with self._lock:
            frames = [f for f in self._frames if f.t <= t_end + 1e-6]
        if not frames:
            return []
        if n <= 1:
            return [frames[-1]]
        targets = [t_end - span_s + i * span_s / (n - 1) for i in range(n)]
        chosen: List[Frame] = []
        for tt in targets:
            best = min(frames, key=lambda f: abs(f.t - tt))
            if not chosen or best.index != chosen[-1].index:
                chosen.append(best)
        chosen.sort(key=lambda f: f.t)
        return chosen


# ------------------------------------------------------------------- probing
def probe_video(path: str) -> Dict[str, Any]:
    """Inspect a video file without decoding it fully."""
    info: Dict[str, Any] = {"ok": False, "path": str(path), "error": "", "fps": 0.0, "frame_count": 0,
                            "duration_s": 0.0, "width": 0, "height": 0}
    if not Path(path).exists():
        info["error"] = "file does not exist"
        return info
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            info["error"] = "OpenCV could not open the file (corrupt or unsupported codec)"
            return info
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        info.update(fps=float(fps) if fps and math.isfinite(fps) and 0 < fps < 1000 else 0.0,
                    frame_count=max(fc, 0), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            info["error"] = "file opened but no frame could be decoded (empty or corrupt)"
            return info
        info["width"], info["height"] = int(frame.shape[1]), int(frame.shape[0])
        if info["fps"] and info["frame_count"] > 0:
            info["duration_s"] = info["frame_count"] / info["fps"]
        info["ok"] = True
        return info
    finally:
        cap.release()


# -------------------------------------------------------------------- source
def _open_capture(src: Any) -> "cv2.VideoCapture":
    if isinstance(src, int) and sys.platform == "win32":
        return cv2.VideoCapture(src, cv2.CAP_DSHOW)  # much faster open on Windows
    return cv2.VideoCapture(src)


class CameraSource(threading.Thread):
    """Reads one source and feeds the shared pipeline. ``state`` is one of
    connecting | connected | reconnecting | unavailable | finished | failed."""

    def __init__(self, camera: CameraConfig, out_q: DropOldestQueue, ring: RingBuffer, stop: threading.Event,
                 ingest_cfg: Dict[str, Any], speed: float = 1.0, loop: bool = False):
        super().__init__(name=f"ingest-{camera.id}", daemon=True)
        self.camera = camera
        self.out_q = out_q
        self.ring = ring
        self.stop_event = stop
        self.cfg = ingest_cfg
        self.speed = float(speed)
        self.loop = loop
        self.state = "connecting"
        self.error = ""
        self.info: Dict[str, Any] = {}
        self.frames_emitted = 0
        self.frames_read = 0
        self.dropped = 0
        self.reconnects = 0
        self.last_frame_wall = 0.0
        self.eof = threading.Event()
        self._cap: Optional["cv2.VideoCapture"] = None

    # -- public ------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        stale = (self.state == "connected" and not self.camera.is_file
                 and time.time() - self.last_frame_wall > self.cfg["stale_after_s"])
        return {"camera_id": self.camera.id, "state": "unavailable" if stale else self.state,
                "error": self.error, "frames_emitted": self.frames_emitted, "dropped": self.dropped,
                "reconnects": self.reconnects, "last_frame_wall": self.last_frame_wall, "info": self.info,
                "source": "file" if self.camera.is_file else ("webcam" if self.camera.is_webcam else "network")}

    # -- thread body ---------------------------------------------------------
    def run(self) -> None:
        try:
            if self.camera.is_file:
                self._run_file()
            else:
                self._run_live()
        except Exception as e:  # noqa: BLE001 - never let a source thread kill the app
            log.exception("[%s] ingest crashed", self.camera.id)
            self.state, self.error = "failed", f"{type(e).__name__}: {e}"
        finally:
            self._release()
            self.eof.set()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001
                pass
            self._cap = None

    def _emit(self, image: np.ndarray, t: float, t_media: Optional[float], idx: int) -> None:
        fr = Frame(camera_id=self.camera.id, index=idx, t=t, image=image, t_media=t_media,
                   t_captured=time.time())
        self.ring.add(fr)
        d = self.out_q.put(fr, self.stop_event)
        self.dropped += d
        self.frames_emitted += 1
        self.last_frame_wall = time.time()

    # -- files -------------------------------------------------------------
    def _run_file(self) -> None:
        path = str(self.camera.source)
        info = probe_video(path)
        self.info = info
        if not info["ok"]:
            self.state, self.error = "failed", info["error"]
            log.error("[%s] %s: %s", self.camera.id, path, info["error"])
            return
        interval = 1.0 / self.camera.target_fps
        while not self.stop_event.is_set():
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                self.state, self.error = "failed", "could not open file"
                return
            self.state = "connected"
            fps = info["fps"]
            wall0 = time.time()
            paced_t0 = time.time()
            next_sample = 0.0
            idx = 0
            while not self.stop_event.is_set():
                if not self._cap.grab():
                    break
                self.frames_read += 1
                pos = self._cap.get(cv2.CAP_PROP_POS_MSEC)
                media_t = pos / 1000.0 if (pos and pos > 0) else (idx / fps if fps else idx * interval)
                idx += 1
                if media_t + 1e-6 < next_sample:
                    continue
                ok, img = self._cap.retrieve()
                if not ok or img is None:
                    continue
                next_sample += interval  # fixed grid avoids drift on VFR / odd fps sources
                if next_sample <= media_t:  # after a long gap re-anchor instead of bursting
                    next_sample = media_t + interval
                if self.speed > 0:  # pace to wall clock / speed
                    wait = paced_t0 + media_t / self.speed - time.time()
                    if wait > 0 and self.stop_event.wait(wait):
                        break
                self._emit(img, wall0 + media_t, media_t, idx)
            self._release()
            if not self.loop or self.stop_event.is_set():
                break
        if self.state != "failed":
            self.state = "finished"

    # -- webcam / RTSP -------------------------------------------------------
    def _run_live(self) -> None:
        rc = self.cfg["reconnect"]
        backoff = rc["initial_s"]
        attempts = 0
        interval = 1.0 / self.camera.target_fps
        while not self.stop_event.is_set():
            self.state = "connecting" if attempts == 0 else "reconnecting"
            self._cap = _open_capture(self.camera.source)
            if not self._cap.isOpened():
                self._release()
                attempts += 1
                self.state, self.error = "unavailable", "could not open source"
                if rc["max_attempts"] and attempts >= rc["max_attempts"]:
                    self.state = "failed"
                    return
                self.reconnects += 1
                log.warning("[%s] open failed, retry in %.1fs", self.camera.id, backoff)
                if self.stop_event.wait(backoff):
                    return
                backoff = min(rc["max_s"], backoff * rc["factor"])
                continue
            self.info = {"width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                         "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                         "fps": float(self._cap.get(cv2.CAP_PROP_FPS) or 0)}
            self.state, self.error = "connected", ""
            self.last_frame_wall = time.time()
            backoff, attempts = rc["initial_s"], 0
            next_sample, idx, fails = 0.0, 0, 0
            while not self.stop_event.is_set():
                if not self._cap.grab():
                    fails += 1
                    if fails >= 30 or time.time() - self.last_frame_wall > self.cfg["stale_after_s"]:
                        self.state, self.error = "unavailable", "stream stopped delivering frames"
                        break
                    self.stop_event.wait(0.05)
                    continue
                fails = 0
                now = time.time()
                self.last_frame_wall = now
                if now < next_sample:
                    continue
                ok, img = self._cap.retrieve()
                if not ok or img is None:
                    continue
                next_sample = now + interval
                idx += 1
                self._emit(img, now, None, idx)
            self._release()
            if self.stop_event.is_set():
                return
            self.reconnects += 1
            attempts = 1  # signals "reconnecting" on the next loop
            if self.stop_event.wait(backoff):
                return
            backoff = min(rc["max_s"], backoff * rc["factor"])
