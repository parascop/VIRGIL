"""Object detection + tracking.

* ``TrackBook``: pure-Python per-camera track history (no heavy imports).
* ``Detector``: one shared Ultralytics YOLO model. Detection is a single shared
  ``predict`` call (guarded by a lock) and **ByteTrack runs per camera** using
  Ultralytics' own ``BYTETracker`` class (the same implementation behind
  ``model.track(tracker="bytetrack.yaml", persist=True)``). ``model.track`` with
  ``persist=True`` keeps one tracker per *model*, which would mix track IDs when
  several cameras share the model; per-camera trackers avoid that.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import Observation, TrackState

log = logging.getLogger("vigil.tracker")

Det = Tuple[int, int, str, Tuple[float, float, float, float], float]  # id, cls_id, cls_name, bbox, conf


class TrackBook:
    """Maintains TrackState history for one camera. Track ids never leak across
    cameras because each camera owns its own TrackBook."""

    def __init__(self, camera_id: str, lost_timeout_s: float = 1.5, history: int = 300):
        self.camera_id = camera_id
        self.lost_timeout_s = lost_timeout_s
        self.history = history
        self.tracks: Dict[int, TrackState] = {}

    def update(self, t: float, dets: Sequence[Det]) -> List[TrackState]:
        """Add one frame of tracked detections; returns tracks still alive
        (seen this frame or within the loss tolerance)."""
        for tid, cid, cname, bbox, conf in dets:
            tr = self.tracks.get(tid)
            if tr is None:
                tr = TrackState(track_id=tid, camera_id=self.camera_id, cls_id=cid, cls_name=cname,
                                history=deque(maxlen=self.history), first_t=t, last_t=t)
                self.tracks[tid] = tr
            tr.history.append(Observation(t=t, bbox=tuple(float(v) for v in bbox), conf=float(conf)))
            tr.last_t = t
        for tid in [k for k, v in self.tracks.items() if t - v.last_t > self.lost_timeout_s]:
            del self.tracks[tid]  # track terminated
        return list(self.tracks.values())

    def visible(self, t: float) -> List[TrackState]:
        return [tr for tr in self.tracks.values() if tr.last_t == t]


class DetectorError(RuntimeError):
    pass


class Detector:
    """Shared YOLO detector with per-camera ByteTrack."""

    def __init__(self, cfg: Dict[str, Any], device: str):
        self.cfg = cfg
        self.device = device
        self._model = None
        self._lock = threading.Lock()
        self._trackers: Dict[str, Any] = {}
        self._tracker_args = None
        self.names: Dict[int, str] = {}
        self.load_seconds: float = 0.0
        self.warmup_seconds: float = 0.0

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load weights (never at import time). Re-callable after a failure."""
        import time

        t0 = time.time()
        try:
            from ultralytics import YOLO
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"ultralytics is not installed: {e}") from e
        from pathlib import Path

        from .config import PROJECT_ROOT

        model_path = str(self.cfg["model"])
        mp = Path(model_path)
        if not mp.is_absolute() and (PROJECT_ROOT / mp).exists():
            model_path = str(PROJECT_ROOT / mp)  # repo-relative weights (models/yolov8n.pt)
        try:
            self._model = YOLO(model_path)
        except Exception as e:  # noqa: BLE001
            raise DetectorError(f"could not load YOLO weights '{model_path}': {e}") from e
        self.names = dict(getattr(self._model, "names", {}) or {})
        self.load_seconds = time.time() - t0
        t1 = time.time()
        try:  # warm-up (not counted as steady-state latency)
            self._predict(np.zeros((self.cfg["imgsz"], self.cfg["imgsz"], 3), np.uint8))
        except Exception as e:  # noqa: BLE001
            log.warning("detector warm-up failed on %s: %s", self.device, e)
            raise
        self.warmup_seconds = time.time() - t1
        log.info("YOLO %s loaded on %s in %.2fs (warm-up %.2fs)", model_path, self.device,
                 self.load_seconds, self.warmup_seconds)

    def reload_on(self, device: str) -> None:
        """Safe fallback: drop the model and trackers, rebuild on another device."""
        with self._lock:
            self._model = None
            self._trackers.clear()
            self.device = device
        self.load()

    def _predict(self, image: np.ndarray):
        kw: Dict[str, Any] = dict(imgsz=self.cfg["imgsz"], conf=self.cfg["conf"], iou=self.cfg["iou"],
                                  classes=self.cfg.get("classes"), verbose=False, device=self.device)
        if self.cfg.get("half") and str(self.device).startswith("cuda"):
            kw["half"] = True
        return self._model.predict(image, **kw)

    def _make_tracker(self):
        from types import SimpleNamespace

        from ultralytics.cfg import get_cfg  # noqa: F401  (ensures cfg module available)
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, yaml_load
        from ultralytics.utils.checks import check_yaml

        if self._tracker_args is None:
            self._tracker_args = IterableSimpleNamespace(**yaml_load(check_yaml(self.cfg["tracker"])))
        return BYTETracker(args=self._tracker_args, frame_rate=int(self.cfg.get("track_fps", 10)))

    def reset_camera(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)

    # -- inference ---------------------------------------------------------
    def track(self, camera_id: str, image: np.ndarray) -> List[Det]:
        """Detect + track on one frame of one camera."""
        if self._model is None:
            raise DetectorError("detector not loaded")
        with self._lock:  # single shared model: serialise inference
            res = self._predict(image)[0]
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = self._trackers[camera_id] = self._make_tracker()
            boxes = res.boxes.cpu().numpy()
            tracks = tracker.update(boxes, image)
        out: List[Det] = []
        for row in tracks:  # x1, y1, x2, y2, id, conf, cls, idx
            x1, y1, x2, y2, tid, conf, cls = (float(v) for v in row[:7])
            cid = int(cls)
            name = self.cfg.get("class_names", {}).get(cid) or self.names.get(cid, str(cid))
            out.append((int(tid), cid, str(name), (x1, y1, x2, y2), conf))
        return out

    def detect_only(self, image: np.ndarray) -> List[Tuple[int, str, Tuple[float, float, float, float], float]]:
        """Plain detection (diagnostics / smoke test)."""
        if self._model is None:
            raise DetectorError("detector not loaded")
        with self._lock:
            res = self._predict(image)[0]
        b = res.boxes.cpu().numpy()
        out = []
        for xyxy, conf, cls in zip(b.xyxy, b.conf, b.cls):
            cid = int(cls)
            out.append((cid, self.names.get(cid, str(cid)), tuple(float(v) for v in xyxy), float(conf)))
        return out


def annotate(image: np.ndarray, dets: Sequence[Det], camera: Any, triggers_active: Sequence[Any] = (),
             status_text: str = "") -> np.ndarray:
    """Draw boxes, track ids, zones, entry lines and trigger banners on a copy."""
    import cv2

    img = image.copy()
    H, W = img.shape[:2]
    for z in camera.zones:
        pts = np.array([[int(x * W), int(y * H)] for x, y in z["polygon"]], np.int32)
        colour = {"restricted": (0, 165, 255), "prohibited": (0, 0, 255), "crowd": (255, 200, 0)}[z["type"]]
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], colour)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
        cv2.polylines(img, [pts], True, colour, 2)
        cv2.putText(img, f"{z['type']}:{z['name']}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    for ln in camera.entry_lines:
        p1 = (int(ln["p1"][0] * W), int(ln["p1"][1] * H))
        p2 = (int(ln["p2"][0] * W), int(ln["p2"][1] * H))
        cv2.line(img, p1, p2, (0, 255, 255), 2)
        # arrow showing the allowed direction (normal to the line)
        mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        nlen = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = dy / nlen, -dx / nlen  # left normal on screen
        sign = -1.0 if ln.get("allowed_direction", "positive") == "positive" else 1.0
        cv2.arrowedLine(img, (mx, my), (int(mx + sign * nx * 40), int(my + sign * ny * 40)), (0, 255, 255), 2, tipLength=0.4)
    for tid, cid, cname, (x1, y1, x2, y2), conf in dets:
        colour = (60, 220, 60) if cname == "person" else (255, 160, 40)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)
        cv2.putText(img, f"{cname}#{tid} {conf:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    y = 22
    for tg in triggers_active:
        cv2.rectangle(img, (0, y - 16), (W, y + 6), (0, 0, 160), -1)
        cv2.putText(img, f"CANDIDATE {tg.category.upper()} ({tg.heuristic_score:.2f})", (6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    if status_text:
        cv2.putText(img, status_text, (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img
