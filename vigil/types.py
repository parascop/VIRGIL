"""Typed objects exchanged between Vigil pipeline components.

Only plain dataclasses live here; there are no model or I/O dependencies so
that every other module (and every unit test) can import this cheaply.
"""
from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np

# The five target categories plus the "none" label used for rejected candidates.
CATEGORIES: Tuple[str, ...] = (
    "person down",
    "fighting",
    "tailgating",
    "crowd incident",
    "suspicious activity",
)
NONE_CATEGORY = "none"
VLM_CATEGORIES: Tuple[str, ...] = CATEGORIES + (NONE_CATEGORY,)
VERDICTS: Tuple[str, ...] = ("verified", "rejected", "unverified")

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def new_event_id() -> str:
    """Stable, sortable-enough unique id for an event."""
    return "ev_" + uuid.uuid4().hex[:16]


@dataclass
class Frame:
    """One sampled video frame.

    ``t`` is the *stream time* in epoch seconds. For live sources it is wall
    clock time. For files it is ``wall_start + media_time`` so that offline
    (faster-than-real-time) processing keeps deterministic relative timing.
    """

    camera_id: str
    index: int
    t: float
    image: np.ndarray
    t_media: Optional[float] = None  # seconds into the file (None for live)
    t_captured: float = 0.0  # wall clock when the frame was read (latency base)


@dataclass
class Observation:
    t: float
    bbox: BBox
    conf: float = 1.0


@dataclass
class TrackState:
    """A tracked object with a bounded observation history."""

    track_id: int
    camera_id: str
    cls_id: int
    cls_name: str
    history: Deque[Observation] = field(default_factory=lambda: deque(maxlen=300))
    first_t: float = 0.0
    last_t: float = 0.0

    # ---- geometry helpers -------------------------------------------------
    @property
    def bbox(self) -> BBox:
        return self.history[-1].bbox

    @property
    def width(self) -> float:
        b = self.bbox
        return max(1e-6, b[2] - b[0])

    @property
    def height(self) -> float:
        b = self.bbox
        return max(1e-6, b[3] - b[1])

    @property
    def aspect_ratio(self) -> float:
        """Width / height of the latest box (>1 means wider than tall)."""
        return self.width / self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        b = self.bbox
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    @property
    def foot(self) -> Tuple[float, float]:
        """Bottom-centre of the box: the best 2-D proxy for floor position."""
        b = self.bbox
        return ((b[0] + b[2]) / 2.0, b[3])

    @property
    def age(self) -> float:
        return self.last_t - self.first_t

    def _obs_at_or_before(self, t: float) -> Optional[Observation]:
        # 1 ms tolerance: window boundaries must not depend on float rounding (epoch-sized
        # timestamps lose ~2e-7 s, which used to flip a 4-frame window into a 5-frame one).
        for o in reversed(self.history):
            if o.t <= t + 1e-3:
                return o
        return None

    def velocity(self, window_s: float) -> Tuple[float, float]:
        """Centroid velocity in px/s over roughly the last ``window_s`` seconds."""
        if len(self.history) < 2:
            return (0.0, 0.0)
        last = self.history[-1]
        ref = self._obs_at_or_before(last.t - window_s)
        if ref is None:
            ref = self.history[0]
        dt = last.t - ref.t
        if dt <= 1e-6:
            return (0.0, 0.0)
        c1 = _centroid(last.bbox)
        c0 = _centroid(ref.bbox)
        return ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)

    def speed_norm(self, window_s: float) -> float:
        """Speed in body-heights per second (camera-distance independent)."""
        vx, vy = self.velocity(window_s)
        return math.hypot(vx, vy) / self.height

    def size_change_rate(self, window_s: float) -> float:
        """Rate of change of box width+height, in body-heights per second."""
        if len(self.history) < 2:
            return 0.0
        last = self.history[-1]
        ref = self._obs_at_or_before(last.t - window_s) or self.history[0]
        dt = last.t - ref.t
        if dt <= 1e-6:
            return 0.0
        dw = abs((last.bbox[2] - last.bbox[0]) - (ref.bbox[2] - ref.bbox[0]))
        dh = abs((last.bbox[3] - last.bbox[1]) - (ref.bbox[3] - ref.bbox[1]))
        return (dw + dh) / dt / self.height


def _centroid(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


@dataclass
class Trigger:
    """A heuristic *candidate* event. Never ground truth."""

    trigger_type: str  # e.g. person_down, fighting, tailgating, crowd_count ...
    category: str  # mapped target category
    camera_id: str
    t: float  # stream time of the trigger frame
    track_ids: Tuple[int, ...] = ()
    heuristic_score: float = 0.0
    bboxes: Dict[int, BBox] = field(default_factory=dict)
    reason: str = ""
    zone: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    dedupe_ids: Optional[Tuple[int, ...]] = None  # key override (crowd uses ())
    t_media: Optional[float] = None
    frame_index: int = 0

    def dedupe_key(self) -> Tuple[str, str, frozenset, str]:
        ids = self.track_ids if self.dedupe_ids is None else self.dedupe_ids
        return (self.camera_id, self.trigger_type, frozenset(ids), self.zone)


@dataclass
class VLMRequest:
    event_id: str
    trigger: Trigger
    composite: np.ndarray  # BGR composite image (already resized)
    trigger_frame: np.ndarray
    sample_count: int
    t_trigger: float  # wall-clock epoch when the trigger fired
    frame_times: Tuple[float, ...] = ()
    source_path: str = ""
    priority: float = 0.0
    t_enqueued: float = 0.0


@dataclass
class VLMResult:
    ok: bool
    category: str = NONE_CATEGORY
    vlm_confidence: float = 0.0
    summary: str = ""
    objects: list = field(default_factory=list)
    reason: str = ""
    raw: str = ""
    model: str = ""
    t_start: float = 0.0
    t_end: float = 0.0
    error: str = ""
    attempts: int = 1
    summary_truncated: bool = False


@dataclass
class EventRecord:
    """Row of the ``events`` table (SQLite is the source of truth)."""

    id: str
    timestamp: float
    camera_id: str
    category: str
    confidence_score: float
    summary: str
    keyframe_path: str
    verdict: str = "unverified"
    trigger_type: str = ""
    heuristic_score: float = 0.0
    vlm_confidence: float = 0.0
    track_ids: list = field(default_factory=list)
    bboxes: dict = field(default_factory=dict)
    vlm_raw: str = ""
    vlm_model: str = ""
    duration_s: float = 0.0
    t_trigger: float = 0.0
    t_vlm_start: float = 0.0
    t_vlm_end: float = 0.0
    t_stored: float = 0.0
    # extras beyond the required column list
    source_path: str = ""
    media_time_s: Optional[float] = None
    error: str = ""
    index_status: str = "pending"  # pending | indexed | failed | skipped
    index_error: str = ""
    meta: dict = field(default_factory=dict)
