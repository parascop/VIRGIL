"""Rule-based candidate-event generation (person down, fighting, tailgating,
crowd incident, suspicious activity) plus the duplicate-suppression cooldown.

Everything here is a *candidate*, never ground truth; the VLM verifies it.
All thresholds come from the merged profile/camera YAML (``params``); there are
no unexplained constants. Known sensitivities are documented per heuristic.

Pure Python (no torch/cv2) so it is unit-testable on any machine.
"""
from __future__ import annotations

import itertools
import logging
import math
import statistics
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from .config import CameraConfig
from .types import BBox, TrackState, Trigger

log = logging.getLogger("vigil.heuristics")

Point = Tuple[float, float]


# ------------------------------------------------------------------ geometry
def point_in_polygon(pt: Point, poly: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def side_of_line(p1: Point, p2: Point, pt: Point) -> float:
    """Signed side of ``pt`` relative to p1->p2 (see cameras.yaml for meaning)."""
    return (p2[0] - p1[0]) * (pt[1] - p1[1]) - (p2[1] - p1[1]) * (pt[0] - p1[0])


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orient(a1, a2, b1), orient(a1, a2, b2)
    o3, o4 = orient(b1, b2, a1), orient(b1, b2, a2)
    return (o1 * o2 <= 0) and (o3 * o4 <= 0) and not (o1 == o2 == 0)


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


# --------------------------------------------------------------- person down
@dataclass
class _FallState:
    ar_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=16))
    upright: Optional[Tuple[float, float, float, float]] = None  # (t, cx, cy, height)
    down_since: Optional[float] = None
    drop_px: float = 0.0
    upright_h: float = 1.0
    last_seen: float = 0.0
    last_emit: float = -1e9


class PersonDownDetector:
    """Aspect-ratio flip + downward drop + persistence.

    Sensitivities (documented limitation): a camera looking steeply down makes
    standing people look wide, a side-on camera makes lying people look wide
    only if the head-to-toe axis is across the image; occlusion by furniture can
    shrink boxes; crawling is rejected only by the stationarity test; sitting
    usually fails the ``ar_down_min`` and drop tests; detector jitter is reduced
    with a median filter over ``smoothing_window`` frames. The VLM is the real
    arbiter.
    """

    def __init__(self, p: Dict[str, Any]):
        self.p = p
        self.state: Dict[int, _FallState] = {}

    def _inherit_upright(self, st: _FallState, tr: TrackState, t: float, cx: float, cy: float) -> None:
        """A person who falls often changes shape so much that the tracker drops
        the box and starts a *new* track id that is already horizontal. If a
        track that was upright vanished moments ago close to where this new
        horizontal track appeared, inherit its upright reference. Deliberately
        conservative (time + distance limits, one claim per lost track); the
        VLM still has to confirm the event."""
        p = self.p
        best, best_d = None, None
        for tid, other in self.state.items():
            if tid == tr.track_id or other.upright is None or other.last_seen >= t:
                continue
            ut, ucx, ucy, uh = other.upright
            if t - ut > p["transition_window_s"]:
                continue
            d = math.hypot(cx - ucx, cy - ucy) / uh
            if d <= p.get("reassoc_max_dist_heights", 1.5) and (best_d is None or d < best_d):
                best, best_d = other, d
        if best is not None:
            st.upright = best.upright
            best.upright = None  # a lost track can be claimed only once

    def update(self, t: float, tracks: Sequence[TrackState], cam: str) -> List[Trigger]:
        p = self.p
        out: List[Trigger] = []
        seen = set()
        for tr in tracks:
            if tr.cls_name != "person" or tr.last_t != t:
                continue
            seen.add(tr.track_id)
            fresh = tr.track_id not in self.state
            st = self.state.setdefault(tr.track_id, _FallState())
            st.last_seen = t
            st.ar_hist.append(tr.aspect_ratio)
            k = int(p["smoothing_window"])
            ar = statistics.median(list(st.ar_hist)[-k:])
            cx, cy = tr.centroid
            if fresh and p.get("reassociate", True) and ar >= p["ar_down_min"]:
                self._inherit_upright(st, tr, t, cx, cy)
            if tr.age < p["min_track_age_s"] and st.upright is None and ar >= p["ar_upright_max"]:
                continue
            if ar < p["ar_upright_max"]:
                st.upright = (t, cx, cy, tr.height)
                st.down_since = None
                continue
            if st.down_since is None:
                if ar >= p["ar_down_min"] and st.upright is not None:
                    ut, _ucx, ucy, uh = st.upright
                    drop = cy - ucy  # image y grows downward
                    if t - ut <= p["transition_window_s"] and drop >= p["min_drop_height_ratio"] * uh:
                        st.down_since = t
                        st.drop_px = drop
                        st.upright_h = uh
                continue
            # candidate already in "down" state: must stay horizontal AND still
            still = tr.speed_norm(p["velocity_window_s"])
            if ar < p["ar_hold_min"] or still > p["stationary_speed_max"]:
                st.down_since = None
                continue
            held = t - st.down_since
            if held >= p["hold_s"] and t - st.last_emit >= 0:
                w = p["weights"]
                persistence = _clip01(held / (2.0 * p["hold_s"]))
                displacement = _clip01(st.drop_px / (p["full_drop_height_ratio"] * st.upright_h))
                stillness = _clip01(1.0 - still / max(1e-9, p["stationary_speed_max"]))
                score = _clip01((w["persistence"] * persistence + w["displacement"] * displacement
                                 + w["stillness"] * stillness) / max(1e-9, sum(w.values())))
                out.append(Trigger(
                    trigger_type="person_down", category="", camera_id=cam, t=t,
                    track_ids=(tr.track_id,), heuristic_score=score,
                    bboxes={tr.track_id: tr.bbox},
                    reason=(f"track {tr.track_id} flipped to horizontal (AR={ar:.2f}), dropped "
                            f"{st.drop_px / st.upright_h:.2f} body-heights, held still {held:.1f}s"),
                    meta={"held_s": held, "aspect_ratio": ar, "t_event_start": st.down_since}))
        # forget tracks that vanished for longer than the tolerated gap
        for tid in list(self.state):
            if tid not in seen and t - self.state[tid].last_seen > max(self.p["max_gap_s"], 0.0):
                self.state[tid].down_since = None
                if t - self.state[tid].last_seen > 30.0:
                    del self.state[tid]
        return out


# ------------------------------------------------------------------ fighting
@dataclass
class _PairState:
    active_since: Optional[float] = None
    last_active: float = -1e9
    peak_energy: float = 0.0


class FightingDetector:
    """Two close people with sustained high *relative* motion.

    Relative motion = |v_a - v_b| / mean_height + w * mean box-size change rate.
    Two people walking together have ~0 relative motion; a hug or handshake is
    close but low-energy; a crowd of dancers is a known false-positive source
    that the VLM must reject.
    """

    def __init__(self, p: Dict[str, Any]):
        self.p = p
        self.pairs: Dict[Tuple[int, int], _PairState] = {}

    def update(self, t: float, tracks: Sequence[TrackState], cam: str) -> List[Trigger]:
        p = self.p
        people = [tr for tr in tracks if tr.cls_name == "person" and tr.last_t == t]
        out: List[Trigger] = []
        active_pairs = set()
        for a, b in itertools.combinations(people, 2):
            key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
            mean_h = (a.height + b.height) / 2.0
            ca, cb = a.centroid, b.centroid
            dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
            close = iou(a.bbox, b.bbox) >= p["iou_min"] or dist < p["distance_height_ratio"] * mean_h
            if not close:
                continue
            w = p["velocity_window_s"]
            va, vb = a.velocity(w), b.velocity(w)
            rel = math.hypot(va[0] - vb[0], va[1] - vb[1]) / mean_h
            size = (a.size_change_rate(w) + b.size_change_rate(w)) / 2.0
            energy = rel + p["bbox_change_weight"] * size
            if energy < p["motion_energy_min"]:
                continue
            st = self.pairs.setdefault(key, _PairState())
            if st.active_since is None or t - st.last_active > p["grace_s"]:
                st.active_since = t
                st.peak_energy = 0.0
            st.last_active = t
            st.peak_energy = max(st.peak_energy, energy)
            active_pairs.add(key)
            dur = t - st.active_since
            if dur >= p["min_duration_s"]:
                wts = p["weights"]
                s = (wts["energy"] * _clip01(energy / (2.0 * p["motion_energy_min"]))
                     + wts["duration"] * _clip01(dur / (3.0 * p["min_duration_s"]))
                     + wts["closeness"] * _clip01(1.0 - dist / max(1e-9, p["distance_height_ratio"] * mean_h)))
                s = _clip01(s / max(1e-9, sum(wts.values())))
                out.append(Trigger(
                    trigger_type="fighting", category="", camera_id=cam, t=t,
                    track_ids=key, heuristic_score=s,
                    bboxes={a.track_id: a.bbox, b.track_id: b.bbox},
                    reason=(f"tracks {key[0]} and {key[1]} close (d={dist / mean_h:.2f} heights) with "
                            f"relative motion energy {energy:.2f} for {dur:.1f}s"),
                    meta={"energy": energy, "duration_s": dur, "t_event_start": st.active_since}))
        for key in list(self.pairs):
            if key not in active_pairs and t - self.pairs[key].last_active > p["grace_s"]:
                self.pairs[key].active_since = None
                if t - self.pairs[key].last_active > 60.0:
                    del self.pairs[key]
        return out


# ---------------------------------------------------------------- tailgating
class TailgatingDetector:
    """Two *distinct* tracks crossing a directed entry line within ``window_s``.

    Without a badge/authorisation signal this only yields "possible tailgating
    that needs verification"; it never claims an authorisation violation.
    """

    def __init__(self, p: Dict[str, Any], lines: Sequence[Dict[str, Any]]):
        self.p = p
        self.lines = list(lines)
        self.last_pt: Dict[Tuple[str, int], Point] = {}
        self.crossings: Dict[str, Deque[Tuple[int, float, BBox]]] = {
            ln["name"]: deque(maxlen=64) for ln in self.lines}
        self.reported: set = set()

    def update(self, t: float, tracks: Sequence[TrackState], cam: str, wh: Tuple[int, int]) -> List[Trigger]:
        W, H = wh
        out: List[Trigger] = []
        for ln in self.lines:
            p1 = (ln["p1"][0], ln["p1"][1])
            p2 = (ln["p2"][0], ln["p2"][1])
            want_positive = ln.get("allowed_direction", "positive") == "positive"
            for tr in tracks:
                if tr.cls_name != "person" or tr.last_t != t or tr.age < self.p["min_track_age_s"]:
                    continue
                fx, fy = tr.foot
                pt = (fx / W, fy / H)
                key = (ln["name"], tr.track_id)
                prev = self.last_pt.get(key)
                self.last_pt[key] = pt
                if prev is None:
                    continue
                s0, s1 = side_of_line(p1, p2, prev), side_of_line(p1, p2, pt)
                if s0 == 0 or s1 == 0 or (s0 > 0) == (s1 > 0):
                    continue
                if not segments_intersect(prev, pt, p1, p2):
                    continue
                crossed_positive = s0 < 0 < s1
                if crossed_positive != want_positive:
                    continue  # wrong direction (exit) is not an entry
                self.crossings[ln["name"]].append((tr.track_id, t, tr.bbox))
                out.extend(self._check(ln["name"], tr.track_id, t, cam))
        return out

    def _check(self, line: str, tid: int, t: float, cam: str) -> List[Trigger]:
        out: List[Trigger] = []
        win = self.p["window_s"]
        for other, ot, obox in list(self.crossings[line]):
            if other == tid or not (0 <= t - ot <= win):
                continue
            key = (line, other, tid)
            if key in self.reported:
                continue
            self.reported.add(key)
            gap = t - ot
            score = _clip01(1.0 - gap / max(win, 1e-9) * 0.7)
            mine = [b for i, c, b in self.crossings[line] if i == tid and c == t]
            bb = {other: obox}
            if mine:
                bb[tid] = mine[-1]
            out.append(Trigger(
                trigger_type="tailgating", category="", camera_id=cam, t=t,
                track_ids=(other, tid), heuristic_score=score, bboxes=bb, zone=line,
                reason=(f"tracks {other} and {tid} crossed entry line '{line}' {gap:.2f}s apart "
                        f"(window {win}s); no badge data, so this is only a *possible* tailgating event"),
                meta={"gap_s": gap, "line": line, "t_event_start": ot}))
        return out


# --------------------------------------------------------------------- crowd
class CrowdDetector:
    """Count threshold (per zone or whole frame) OR sudden dispersal."""

    def __init__(self, p: Dict[str, Any], zones: Sequence[Dict[str, Any]]):
        self.p = p
        self.zones = [z for z in zones if z.get("type") == "crowd"]
        self.over_since: Dict[str, Optional[float]] = {}
        self.speed_hist: Deque[Tuple[float, float, int]] = deque(maxlen=2000)

    def update(self, t: float, tracks: Sequence[TrackState], cam: str, wh: Tuple[int, int]) -> List[Trigger]:
        p = self.p
        W, H = wh
        people = [tr for tr in tracks if tr.cls_name == "person" and tr.last_t == t]
        out: List[Trigger] = []
        regions: List[Tuple[str, Optional[List[Point]], float]] = []
        if self.zones:
            for z in self.zones:
                poly = [(x, y) for x, y in z["polygon"]]
                regions.append((z["name"], poly, float(z.get("crowd_count", p["crowd_count"]))))
        else:
            regions.append(("frame", None, float(p["crowd_count"])))
        for name, poly, limit in regions:
            inside = [tr for tr in people
                      if poly is None or point_in_polygon((tr.foot[0] / W, tr.foot[1] / H), poly)]
            n = len(inside)
            if n > limit:
                since = self.over_since.get(name)
                if since is None:
                    self.over_since[name] = since = t
                if t - since >= p["min_duration_s"]:
                    score = _clip01(0.5 + 0.5 * (n - limit) / max(limit, 1.0))
                    out.append(Trigger(
                        trigger_type="crowd_count", category="", camera_id=cam, t=t,
                        track_ids=tuple(sorted(tr.track_id for tr in inside)), dedupe_ids=(),
                        heuristic_score=score, zone=name,
                        bboxes={tr.track_id: tr.bbox for tr in inside[:30]},
                        reason=f"{n} people in '{name}' (limit {int(limit)}) for {t - since:.1f}s",
                        meta={"count": n, "limit": limit, "t_event_start": since}))
            else:
                self.over_since[name] = None
        d = p["dispersal"]
        if d["enabled"]:
            out.extend(self._dispersal(t, people, cam, d))
        return out

    def _dispersal(self, t: float, people: Sequence[TrackState], cam: str, d: Dict[str, Any]) -> List[Trigger]:
        movers = [tr for tr in people if tr.age >= 1.0]
        if len(movers) < d["min_tracks"]:
            return []
        mean_speed = sum(tr.speed_norm(d["velocity_window_s"]) for tr in movers) / len(movers)
        self.speed_hist.append((t, mean_speed, len(movers)))
        recent = [s for ts, s, _ in self.speed_hist if t - ts <= d["window_s"]]
        base = [s for ts, s, _ in self.speed_hist if d["window_s"] < t - ts <= d["baseline_s"] + d["window_s"]]
        base_span = max((ts for ts, _, _ in self.speed_hist if t - ts > d["window_s"]), default=None)
        first = self.speed_hist[0][0]
        if not recent or len(base) < 3 or (t - d["window_s"] - first) < d["baseline_min_s"]:
            return []
        cur = sum(recent) / len(recent)
        baseline = sum(base) / len(base)
        if cur >= d["min_speed"] and cur >= d["speed_ratio"] * max(baseline, 0.05):
            score = _clip01(0.5 + 0.5 * (cur / max(baseline, 0.05) / d["speed_ratio"] - 1.0))
            return [Trigger(
                trigger_type="crowd_dispersal", category="", camera_id=cam, t=t,
                track_ids=tuple(sorted(tr.track_id for tr in movers)), dedupe_ids=(),
                heuristic_score=score, zone="frame",
                bboxes={tr.track_id: tr.bbox for tr in movers[:30]},
                reason=(f"mean speed of {len(movers)} tracks rose to {cur:.2f} body-heights/s "
                        f"vs baseline {baseline:.2f} within {d['window_s']}s"),
                meta={"speed": cur, "baseline": baseline, "t_event_start": t - d["window_s"]})]
        return []


# ---------------------------------------------------------------- suspicious
class SuspiciousDetector:
    """Loitering in restricted polygons, entry into prohibited polygons and
    (stretch) abandoned objects."""

    def __init__(self, p: Dict[str, Any], zones: Sequence[Dict[str, Any]]):
        self.p = p
        self.zones = [z for z in zones if z.get("type") in ("restricted", "prohibited")]
        self.dwell: Dict[Tuple[str, int], Tuple[float, float]] = {}  # (zone, tid) -> (first, last)
        self.obj: Dict[int, Dict[str, Any]] = {}

    def update(self, t: float, tracks: Sequence[TrackState], cam: str, wh: Tuple[int, int]) -> List[Trigger]:
        p = self.p
        W, H = wh
        out: List[Trigger] = []
        people = [tr for tr in tracks if tr.cls_name == "person" and tr.last_t == t]
        for z in self.zones:
            poly = [(x, y) for x, y in z["polygon"]]
            for tr in people:
                key = (z["name"], tr.track_id)
                inside = point_in_polygon((tr.foot[0] / W, tr.foot[1] / H), poly)
                first, last = self.dwell.get(key, (None, None))
                if inside:
                    if first is None or t - last > p["exit_grace_s"]:
                        first = t
                    self.dwell[key] = (first, t)
                    dwell = t - first
                    if z["type"] == "prohibited" and dwell >= p["prohibited_min_s"]:
                        out.append(Trigger(
                            trigger_type="zone_entry", category="", camera_id=cam, t=t,
                            track_ids=(tr.track_id,), heuristic_score=_clip01(0.6 + 0.4 * min(1.0, dwell / 3.0)),
                            bboxes={tr.track_id: tr.bbox}, zone=z["name"],
                            reason=f"track {tr.track_id} entered prohibited zone '{z['name']}'",
                            meta={"dwell_s": dwell, "t_event_start": first}))
                    elif z["type"] == "restricted":
                        loiter = float(z.get("loiter_s", p["loiter_s"]))
                        if dwell >= loiter:
                            out.append(Trigger(
                                trigger_type="loitering", category="", camera_id=cam, t=t,
                                track_ids=(tr.track_id,), heuristic_score=_clip01(0.5 + 0.5 * min(1.0, dwell / (3 * loiter))),
                                bboxes={tr.track_id: tr.bbox}, zone=z["name"],
                                reason=f"track {tr.track_id} stayed {dwell:.1f}s in restricted zone '{z['name']}' (limit {loiter}s)",
                                meta={"dwell_s": dwell, "t_event_start": first}))
        ab = p["abandoned"]
        if ab["enabled"]:
            out.extend(self._abandoned(t, tracks, people, cam, ab))
        return out

    def _abandoned(self, t: float, tracks: Sequence[TrackState], people: Sequence[TrackState],
                   cam: str, ab: Dict[str, Any]) -> List[Trigger]:
        out: List[Trigger] = []
        for tr in tracks:
            if tr.cls_name not in ab["classes"] or tr.last_t != t:
                continue
            st = self.obj.setdefault(tr.track_id, {"anchor": tr.centroid, "since": t, "owner_seen": False})
            h = tr.height
            ax, ay = st["anchor"]
            cx, cy = tr.centroid
            if math.hypot(cx - ax, cy - ay) > ab["stationary_move_ratio"] * h:
                st.update(anchor=tr.centroid, since=t, owner_seen=False)
            near = any(math.hypot(pp.centroid[0] - cx, pp.centroid[1] - cy) < ab["owner_radius_ratio"] * max(tr.width, h)
                       for pp in people)
            if near:
                st["owner_seen"] = True
                st["last_near"] = t
                continue
            if st["owner_seen"] and t - st["since"] >= ab["stationary_s"]:
                out.append(Trigger(
                    trigger_type="abandoned_object", category="", camera_id=cam, t=t,
                    track_ids=(tr.track_id,), heuristic_score=0.5,
                    bboxes={tr.track_id: tr.bbox}, reason=f"{tr.cls_name} {tr.track_id} stationary {t - st['since']:.0f}s with no person nearby",
                    meta={"t_event_start": st["since"]}))
        return out


# -------------------------------------------------------------------- engine
class HeuristicEngine:
    """Per-camera collection of heuristics. Returns triggers with categories."""

    def __init__(self, camera: CameraConfig, params: Dict[str, Any], categories: Dict[str, str]):
        self.cam = camera.id
        self.params = params
        self.categories = categories
        self.emit_interval = float(params.get("emit_interval_s", 1.0))
        self._last_emit: Dict[Tuple, float] = {}
        self.person_down = PersonDownDetector(params["person_down"]) if params["person_down"]["enabled"] else None
        self.fighting = FightingDetector(params["fighting"]) if params["fighting"]["enabled"] else None
        self.tailgating = (TailgatingDetector(params["tailgating"], camera.entry_lines)
                           if params["tailgating"]["enabled"] and camera.entry_lines else None)
        self.crowd = CrowdDetector(params["crowd"], camera.zones) if params["crowd"]["enabled"] else None
        self.suspicious = (SuspiciousDetector(params["suspicious"], camera.zones)
                           if params["suspicious"]["enabled"] else None)

    def update(self, t: float, tracks: Sequence[TrackState], frame_wh: Tuple[int, int]) -> List[Trigger]:
        trig: List[Trigger] = []
        if self.person_down:
            trig += self.person_down.update(t, tracks, self.cam)
        if self.fighting:
            trig += self.fighting.update(t, tracks, self.cam)
        if self.tailgating:
            trig += self.tailgating.update(t, tracks, self.cam, frame_wh)
        if self.crowd:
            trig += self.crowd.update(t, tracks, self.cam, frame_wh)
        if self.suspicious:
            trig += self.suspicious.update(t, tracks, self.cam, frame_wh)
        out = []
        for tg in trig:
            tg.category = self.categories.get(tg.trigger_type, "suspicious activity")
            k = tg.dedupe_key()
            if t - self._last_emit.get(k, -1e9) < self.emit_interval:
                continue  # rate-limit re-emission of a persisting candidate
            self._last_emit[k] = t
            out.append(tg)
        return out


# ------------------------------------------------------------------ cooldown
@dataclass
class CooldownEntry:
    event_id: str
    t_first: float
    t_last: float
    trigger: Trigger
    best_score: float
    extensions: int = 0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.t_last - self.t_first)


class CooldownManager:
    """Suppress near-duplicate events.

    Key: (camera_id, trigger_type, track_id_set[, zone]). A trigger whose key was
    seen within ``cooldown_s`` extends the existing event's duration instead of
    creating a new event. A *different* track id set is deliberately treated as
    a new candidate: we never silently merge unrelated people.
    """

    def __init__(self, cooldown_s: float):
        self.cooldown_s = float(cooldown_s)
        self.entries: Dict[Tuple, CooldownEntry] = {}

    def check(self, trig: Trigger) -> Tuple[str, CooldownEntry]:
        """Returns ("new" | "extend", entry)."""
        key = trig.dedupe_key()
        ent = self.entries.get(key)
        if ent is not None and trig.t - ent.t_last < self.cooldown_s:
            ent.t_last = max(ent.t_last, trig.t)
            ent.extensions += 1
            ent.best_score = max(ent.best_score, trig.heuristic_score)
            return "extend", ent
        ent = CooldownEntry(event_id="", t_first=trig.t, t_last=trig.t, trigger=trig,
                            best_score=trig.heuristic_score)
        self.entries[key] = ent
        self._gc(trig.t)
        return "new", ent

    def _gc(self, now: float) -> None:
        if len(self.entries) < 512:
            return
        for k in [k for k, e in self.entries.items() if now - e.t_last > 4 * self.cooldown_s]:
            del self.entries[k]
