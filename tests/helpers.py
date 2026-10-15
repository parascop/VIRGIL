"""Shared test helpers: synthetic video, scripted fake detector/verifier."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from vigil.config import CameraConfig, heuristic_params, load_profile
from vigil.heuristics import HeuristicEngine
from vigil.tracker import TrackBook
from vigil.types import VLMRequest, VLMResult


def make_profile(tmp: Path, **over: Any) -> Dict[str, Any]:
    p = load_profile(None)
    for k in ("sqlite", "chroma", "keyframes", "live", "clips", "jobs", "cache"):
        p["paths"][k] = str(tmp / k if k not in ("sqlite",) else tmp / "vigil.db")
    p["detector"]["track_lost_timeout_s"] = 1.5
    p["live_frame_interval_s"] = 0.0
    p["status_write_interval_s"] = 0.2
    p["cooldown_s"] = 30
    for k, v in over.items():
        p[k].update(v) if isinstance(v, dict) else p.__setitem__(k, v)
    return p


def make_video(path: Path, n_frames: int = 30, fps: int = 10, size=(160, 120)) -> str:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened(), "cannot create test video"
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.putText(img, str(i), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        w.write(img)
    w.release()
    return str(path)


def cam(cid: str = "cam1", **kw: Any) -> CameraConfig:
    return CameraConfig(id=cid, source=kw.pop("source", "x.mp4"), **kw)


def engine_for(profile: Dict[str, Any], camera: CameraConfig) -> HeuristicEngine:
    return HeuristicEngine(camera, heuristic_params(profile, camera), profile["categories"])


def run_engine(engine: HeuristicEngine, frames: Sequence[Sequence[tuple]], fps: float = 10.0,
               wh=(640, 480), t0: float = 1000.0, book: Optional[TrackBook] = None):
    """frames[k] = list of (track_id, cls_name, bbox). Returns list of triggers."""
    book = book or TrackBook(engine.cam)
    out = []
    for k, dets in enumerate(frames):
        t = t0 + k / fps
        tracked = [(tid, 0 if cls == "person" else 24, cls, bb, 0.9) for tid, cls, bb in dets]
        tracks = book.update(t, tracked)
        out += engine.update(t, tracks, wh)
    return out


def box(cx: float, cy: float, w: float, h: float):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class FakeDetector:
    """Returns scripted detections per call; counts calls per camera."""

    def __init__(self, script: Dict[str, List[List[tuple]]], fail_every: int = 0, device: str = "cpu"):
        self.script = script
        self.calls: Dict[str, int] = {}
        self.fail_every = fail_every
        self.device = device
        self.names = {0: "person"}

    def reset_camera(self, cid: str) -> None:
        self.calls[cid] = 0

    def track(self, cid: str, image: np.ndarray):
        k = self.calls.get(cid, 0)
        self.calls[cid] = k + 1
        if self.fail_every and (k + 1) % self.fail_every == 0:
            raise RuntimeError("synthetic inference failure")
        seq = self.script.get(cid, [])
        frame = seq[k] if k < len(seq) else []
        return [(tid, 0, name, bb, 0.9) for tid, name, bb in frame]


class FakeVerifier:
    def __init__(self, mode: str = "verify", delay: float = 0.0):
        self.mode, self.delay, self.calls = mode, delay, 0
        self.model = "fake-vlm"

    def check(self):
        return {"reachable": True, "model": "fake-vlm", "model_available": True, "error": ""}

    def verify(self, req: VLMRequest) -> VLMResult:
        self.calls += 1
        t0 = time.time()
        time.sleep(self.delay)
        if self.mode == "timeout":
            return VLMResult(ok=False, model="fake-vlm", t_start=t0, t_end=time.time(), error="timeout: inference timed out")
        if self.mode == "reject":
            return VLMResult(ok=True, category="none", vlm_confidence=0.9, summary="Nothing unusual visible.",
                             objects=["person"], reason="no event", raw="{}", model="fake-vlm", t_start=t0, t_end=time.time())
        return VLMResult(ok=True, category=req.trigger.category, vlm_confidence=0.8, summary="A person is lying on the floor.",
                         objects=["person"], reason="visible", raw="{}", model="fake-vlm", t_start=t0, t_end=time.time())


def person_down_script(n_up: int = 20, n_still: int = 35):
    """Track 1 stands, falls in 3 frames, then lies still."""
    frames: List[List[tuple]] = []
    for _ in range(n_up):
        frames.append([(1, "person", box(300, 250, 40, 100))])
    frames.append([(1, "person", box(305, 265, 70, 80))])
    frames.append([(1, "person", box(310, 290, 90, 55))])
    frames.append([(1, "person", box(310, 310, 100, 40))])
    for _ in range(n_still):
        frames.append([(1, "person", box(310, 310, 100, 40))])
    return frames


class MockOllama:
    """Tiny HTTP server mimicking the Ollama endpoints Vigil uses (TEST FIXTURE)."""

    def __init__(self, responses: Optional[List[Any]] = None, delay: float = 0.0, version: str = "0.9.0",
                 models: Sequence[str] = ("moondream:latest",)):
        self.responses = list(responses or [])
        self.delay, self.version, self.models = delay, version, list(models)
        self.requests: List[Dict[str, Any]] = []
        self.post_status = 200
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                return

            def _send(self, code: int, obj: Any) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/version":
                    self._send(200, {"version": outer.version})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.models]})
                elif self.path == "/api/ps":
                    self._send(200, {"models": []})
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(body)
                time.sleep(outer.delay)
                if outer.post_status != 200:
                    self._send(outer.post_status, {"error": "model runner crashed"})
                    return
                r = outer.responses.pop(0) if outer.responses else {"category": "none", "vlm_confidence": 0.5, "summary": "", "objects": [], "reason": ""}
                content = r if isinstance(r, str) else json.dumps(r)
                self._send(200, {"message": {"role": "assistant", "content": content}})

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a: Any):
        self.server.shutdown()
        self.server.server_close()
