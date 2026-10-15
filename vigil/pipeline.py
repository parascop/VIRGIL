"""Video pipeline orchestration (Process 1).

Threads (all bounded, all stoppable):
  T1  one ingestion thread per camera (vigil.ingest.CameraSource)
  T2  one detection/tracking/heuristics worker (shared YOLO model)
  T3  one VLM verification worker (bounded priority queue, never blocks T1/T2)
  T4  one embedding + persistence worker (bounded queue)
  +   status writer / job watcher and the CLIP text-embedding RPC for the UI

Launch:  python -m vigil.pipeline --config configs/cameras.yaml --profile configs/profile_rtx2050.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .composite import build_composite
from .config import CameraConfig, ConfigError, heuristic_params, load_cameras, load_profile, parse_camera, to_local_dirs, CONFIG_DIR
from .heuristics import CooldownEntry, CooldownManager, HeuristicEngine
from .ingest import CameraSource, DropOldestQueue, RingBuffer
from .metrics import Metrics
from .storage import EventStore, persist_event, reindex_pending
from .tracker import Detector, DetectorError, TrackBook, annotate
from .types import EventRecord, Frame, NONE_CATEGORY, Trigger, VLMRequest, VLMResult, new_event_id
from .util import atomic_write_image, atomic_write_json, setup_logging
from .vlm import VLMVerifier, apply_verdict

log = logging.getLogger("vigil.pipeline")


# ------------------------------------------------------------------- queues
class BoundedPriorityQueue:
    """Bounded queue that pops the highest-priority item first. When full it
    never grows: it returns the evicted item(s) so the caller can persist them
    as ``unverified`` (no silent loss)."""

    def __init__(self, maxsize: int, overflow: str = "lowest_priority"):
        self.maxsize, self.overflow = maxsize, overflow
        self._items: List[Tuple[float, int, Any]] = []
        self._seq = 0
        self._cv = threading.Condition()
        self.dropped = 0
        self.max_depth = 0

    def put(self, item: Any, priority: float) -> List[Any]:
        with self._cv:
            evicted: List[Any] = []
            self._seq += 1
            entry = (float(priority), self._seq, item)
            if len(self._items) >= self.maxsize:
                if self.overflow == "reject_new":
                    evicted.append(item)
                    self.dropped += 1
                    return evicted
                if self.overflow == "drop_oldest":
                    victim = min(self._items, key=lambda e: e[1])
                else:  # lowest_priority: ties evict the older item
                    victim = min(self._items + [entry], key=lambda e: (e[0], e[1]))
                if victim is entry:
                    evicted.append(item)
                    self.dropped += 1
                    return evicted
                self._items.remove(victim)
                evicted.append(victim[2])
                self.dropped += 1
            self._items.append(entry)
            self.max_depth = max(self.max_depth, len(self._items))
            self._cv.notify()
            return evicted

    def get(self, timeout: float = 0.2) -> Optional[Any]:
        with self._cv:
            if not self._items:
                self._cv.wait(timeout)
            if not self._items:
                return None
            best = max(self._items, key=lambda e: (e[0], -e[1]))
            self._items.remove(best)
            return best[2]

    def qsize(self) -> int:
        with self._cv:
            return len(self._items)

    def drain(self) -> List[Any]:
        with self._cv:
            out = [e[2] for e in sorted(self._items, key=lambda e: e[1])]
            self._items.clear()
            return out


@dataclass
class PersistJob:
    record: EventRecord
    composite: Optional[np.ndarray]
    trigger_frame: Optional[np.ndarray]
    entry: Optional[CooldownEntry] = None


class CamRuntime:
    def __init__(self, cfg: CameraConfig, profile: Dict[str, Any], source: CameraSource, queue: DropOldestQueue,
                 ring: RingBuffer):
        self.cfg, self.source, self.queue, self.ring = cfg, source, queue, ring
        self.book = TrackBook(cfg.id, profile["detector"]["track_lost_timeout_s"])
        self.engine = HeuristicEngine(cfg, heuristic_params(profile, cfg), profile["categories"])
        self.cooldown = CooldownManager(profile["cooldown_s"])
        self.last_live = 0.0
        self.banners: Dict[Tuple, Tuple[float, Trigger]] = {}
        self.wh: Tuple[int, int] = (0, 0)
        self.last_t = 0.0
        self.last_wall = 0.0
        self.frames = 0
        self.next_sparse: Optional[float] = None
        self.job_id: Optional[str] = None


class Pipeline:
    def __init__(self, profile: Dict[str, Any], cameras: Sequence[CameraConfig], *, speed: float = 1.0,
                 loop: bool = False, vlm_enabled: Optional[bool] = None, detector: Any = None,
                 verifier: Any = None, embedder: Any = None, store: Optional[EventStore] = None,
                 index: Any = None, watch_jobs: bool = True, write_live: bool = True,
                 hw: Optional[Dict[str, Any]] = None):
        self.p = profile
        self.speed = speed
        self.loop = loop
        self.vlm_enabled = profile["vlm"]["enabled"] if vlm_enabled is None else vlm_enabled
        self.stop_event = threading.Event()
        self.metrics = Metrics()
        self.hw = hw or {}
        self.detector = detector
        self.verifier = verifier
        self.embedder = embedder
        self.store = store
        self.index = index
        self.watch_jobs = watch_jobs
        self.write_live = write_live
        self.cams: Dict[str, CamRuntime] = {}
        self._cams_lock = threading.Lock()
        self.vlm_q = BoundedPriorityQueue(profile["vlm"]["queue_maxsize"], profile["vlm"]["queue_overflow"])
        self.persist_q = DropOldestQueue(profile["embeddings"]["persist_queue_maxsize"], overflow="block")
        self.threads: List[threading.Thread] = []
        self._busy = {"detect": 0, "vlm": 0, "persist": 0}
        self._busy_lock = threading.Lock()
        self._dur: Dict[str, CooldownEntry] = {}
        self._initial_cams = list(cameras)
        self.event_counts: Dict[str, int] = {}
        self.verdict_counts: Dict[str, int] = {"verified": 0, "rejected": 0, "unverified": 0}
        self.vlm_info: Dict[str, Any] = {}
        self.embedder_ok = False
        self.detect_failures = 0
        self.fallback_done = False
        self.started_at = 0.0
        self.live_dir = Path(profile["paths"]["live"])
        self.started = False
        self._jobs_dir = Path(profile["paths"]["jobs"])

    # ---------------------------------------------------------- lifecycle
    def start(self) -> None:
        to_local_dirs(self.p)
        self.started_at = time.time()
        self.store = self.store or EventStore(self.p["paths"]["sqlite"])
        self._init_detector()
        self._init_embedder_and_index()
        self._init_vlm()
        if self.embedder_ok:
            try:
                n = reindex_pending(self.store, self.embedder, self.index, self.p["paths"]["keyframes"])
                if n:
                    log.info("re-indexed %d events left unindexed by a previous run", n)
            except Exception as e:  # noqa: BLE001
                self.metrics.error("reindex", str(e))
        for cam in self._initial_cams:
            self.add_camera(cam)
        self._spawn("detect", self._detect_loop)
        self._spawn("vlm", self._vlm_loop)
        self._spawn("persist", self._persist_loop)
        self._spawn("status", self._status_loop)
        if self.embedder_ok:
            from .embeddings import EmbedRpcServer

            rpc = EmbedRpcServer(self.embedder, Path(self.p["paths"]["jobs"]).parent / "rpc", self.stop_event)
            rpc.start()
            self.threads.append(rpc)
        self.started = True
        log.info("pipeline started (%d cameras, vlm=%s)", len(self.cams), "on" if self.vlm_enabled else "off")

    def _spawn(self, name: str, fn: Callable[[], None]) -> None:
        def runner() -> None:
            while not self.stop_event.is_set():
                try:
                    fn()
                    return
                except Exception as e:  # noqa: BLE001 - worker crash must not kill the app
                    log.exception("worker %s crashed; restarting", name)
                    self.metrics.error(name, f"{type(e).__name__}: {e}")
                    self.stop_event.wait(1.0)

        t = threading.Thread(target=runner, name=name, daemon=True)
        t.start()
        self.threads.append(t)

    def _init_detector(self) -> None:
        if self.detector is None:
            from .device import resolve_device

            dev, why = resolve_device(self.p["detector"].get("device", "auto") if self.p["detector"].get("device") != "auto"
                                      else self.p.get("torch_device", "auto"), self.hw or None)
            self.detector = Detector(dict(self.p["detector"], track_fps=self.p["ingest"]["target_fps"]), dev)
            log.info("detector device: %s (%s)", dev, why)
            try:
                self.detector.load()
            except Exception as e:  # noqa: BLE001
                if dev != "cpu":
                    log.warning("YOLO failed on %s (%s); reinitialising on CPU", dev, e)
                    self.detector.reload_on("cpu")
                else:
                    raise
            self.metrics.warmup["yolo_load_s"] = round(self.detector.load_seconds, 3)
            self.metrics.warmup["yolo_warmup_s"] = round(self.detector.warmup_seconds, 3)
            self.metrics.set("detector_device", self.detector.device)

    def _init_embedder_and_index(self) -> None:
        try:
            if self.embedder is None:
                from .embeddings import make_embedder

                self.embedder = make_embedder(self.p)
            t0 = time.time()
            self.embedder.load()
            self.metrics.warmup["clip_load_s"] = round(time.time() - t0, 3)
            if self.index is None:
                from .embeddings import VectorIndex

                self.index = VectorIndex(self.p["paths"]["chroma"], self.embedder.tag, self.embedder.dim)
            self.embedder_ok = True
            self.metrics.set("clip", self.embedder.tag)
        except Exception as e:  # noqa: BLE001 - search degrades, detection continues
            self.embedder_ok = False
            log.error("CLIP/Chroma unavailable: %s. Events will be stored without embeddings "
                      "(rerun `python -m vigil.embeddings --repair` later).", e)
            self.metrics.error("embeddings", str(e))

    def _init_vlm(self) -> None:
        if not self.vlm_enabled:
            self.vlm_info = {"enabled": False}
            return
        if self.verifier is None:
            self.verifier = VLMVerifier(self.p["vlm"])
        try:
            self.vlm_info = self.verifier.check() if hasattr(self.verifier, "check") else {"reachable": True}
        except Exception as e:  # noqa: BLE001
            self.vlm_info = {"reachable": False, "error": str(e)}
        if self.vlm_info.get("error"):
            log.warning("VLM not ready: %s (events will be stored as unverified until it is)", self.vlm_info["error"])

    def add_camera(self, cfg: CameraConfig, speed: Optional[float] = None, job_id: Optional[str] = None) -> None:
        sp = self.speed if speed is None else speed
        with self._cams_lock:
            old = self.cams.get(cfg.id)
            if old is not None:
                if old.source.is_alive() and not old.source.eof.is_set():
                    raise RuntimeError(f"camera {cfg.id} is still busy")
            ing = self.p["ingest"]
            q = DropOldestQueue(ing["queue_maxsize"], "block" if (cfg.is_file and sp == 0) else "drop_oldest")
            # The ring must also cover the frames still waiting in the detection queue, otherwise
            # a trigger would only find the last few already-processed frames in the ring.
            lag_s = ing["queue_maxsize"] / cfg.target_fps + 1.0
            ring = RingBuffer(ing["ring_seconds"] + lag_s, cfg.target_fps)
            src = CameraSource(cfg, q, ring, self.stop_event, ing, speed=sp, loop=self.loop)
            rt = CamRuntime(cfg, self.p, src, q, ring)
            rt.job_id = job_id
            if hasattr(self.detector, "reset_camera"):
                self.detector.reset_camera(cfg.id)
            self.cams[cfg.id] = rt
        src.start()
        log.info("camera %s started (%s)", cfg.id, "file" if cfg.is_file else "live")

    def stop(self, timeout: float = 15.0) -> None:
        """Graceful shutdown: stop sources, drain queues into the DB, join threads."""
        if self.stop_event.is_set() and not self.threads:
            return
        self.stop_event.set()
        deadline = time.time() + timeout
        for rt in list(self.cams.values()):
            rt.source.join(max(0.1, deadline - time.time()))
        for t in self.threads:
            t.join(max(0.1, deadline - time.time()))
        # nothing queued for the VLM may be lost: persist as unverified
        for req in self.vlm_q.drain():
            self._persist_from_request(req, None, "pipeline shut down before the VLM verified this candidate")
        while True:
            job = self.persist_q.get(0)
            if job is None:
                break
            self._do_persist(job)
        self._write_status(final=True)
        self.threads.clear()
        log.info("pipeline stopped")

    # -------------------------------------------------------------- helpers
    def _busy_inc(self, k: str, d: int) -> None:
        with self._busy_lock:
            self._busy[k] += d

    def all_idle(self) -> bool:
        with self._cams_lock:
            cams = list(self.cams.values())
        if any(not rt.source.eof.is_set() or rt.queue.qsize() for rt in cams):
            return False
        with self._busy_lock:
            if any(v for v in self._busy.values()):
                return False
        return self.vlm_q.qsize() == 0 and self.persist_q.qsize() == 0

    def run_until_done(self, timeout: Optional[float] = None) -> bool:
        """Offline mode: wait until all file sources are fully processed."""
        t0 = time.time()
        quiet = 0
        while not self.stop_event.is_set():
            if self.all_idle():
                quiet += 1
                if quiet >= 3:
                    return True
            else:
                quiet = 0
            if timeout and time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return False

    # ------------------------------------------------------------ detection
    def _detect_loop(self) -> None:
        while not self.stop_event.is_set():
            got = False
            with self._cams_lock:
                cams = list(self.cams.values())
            for rt in cams:
                fr = rt.queue.get(0)
                if fr is None:
                    continue
                got = True
                self._busy_inc("detect", 1)
                try:
                    self._process_frame(rt, fr)
                except Exception as e:  # noqa: BLE001
                    log.exception("[%s] frame processing failed", rt.cfg.id)
                    self.metrics.error("detect", f"{rt.cfg.id}: {type(e).__name__}: {e}")
                finally:
                    self._busy_inc("detect", -1)
            if not got:
                self.stop_event.wait(0.01)

    def _recover_detector(self, err: Exception) -> None:
        """Bounded fallback: after repeated failures reinitialise on CPU once."""
        self.detect_failures += 1
        self.metrics.error("detector", f"{type(err).__name__}: {err}")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # best effort only; does not guarantee OOM recovery
        except Exception:  # noqa: BLE001
            pass
        if self.detect_failures >= 3 and not self.fallback_done and getattr(self.detector, "device", "cpu") != "cpu" \
                and hasattr(self.detector, "reload_on"):
            self.fallback_done = True
            log.warning("detector failing on %s: reinitialising on CPU", self.detector.device)
            try:
                self.detector.reload_on("cpu")
                self.metrics.set("detector_device", "cpu (fallback)")
            except Exception as e:  # noqa: BLE001
                self.metrics.error("detector", f"CPU fallback failed: {e}")

    def _process_frame(self, rt: CamRuntime, fr: Frame) -> None:
        self.metrics.observe("ingest_to_detect", max(0.0, time.time() - fr.t_captured))
        H, W = fr.image.shape[:2]
        rt.wh = (W, H)
        t0 = time.perf_counter()
        try:
            dets = self.detector.track(rt.cfg.id, fr.image)
        except Exception as e:  # noqa: BLE001
            self._recover_detector(e)
            return
        self.metrics.observe("detect", time.perf_counter() - t0)
        rt.last_t, rt.last_wall, rt.frames = fr.t, time.time(), rt.frames + 1
        self.metrics.frame_processed(rt.cfg.id)
        tracks = rt.book.update(fr.t, dets)
        for tg in rt.engine.update(fr.t, tracks, (W, H)):
            self._handle_trigger(rt, tg, fr)
        now = time.time()
        for k in [k for k, (exp, _) in rt.banners.items() if exp < now]:
            del rt.banners[k]
        if self.write_live and now - rt.last_live >= self.p["live_frame_interval_s"]:
            rt.last_live = now
            try:
                img = annotate(fr.image, dets, rt.cfg, [tg for _, tg in rt.banners.values()],
                               f"{rt.cfg.id} {self.metrics.fps(rt.cfg.id):.1f} fps")
                atomic_write_image(self.live_dir / f"{rt.cfg.id}.jpg", img, self.p["live_jpeg_quality"])
            except Exception as e:  # noqa: BLE001
                self.metrics.error("live_frame", str(e))
        sp = self.p["ingest"]["sparse_index"]
        if sp["enabled"]:
            if rt.next_sparse is None:
                rt.next_sparse = fr.t
            if fr.t >= rt.next_sparse:
                rt.next_sparse = fr.t + sp["interval_s"]
                self._persist_sparse(rt, fr)

    # -------------------------------------------------------------- triggers
    def _duration(self, e: CooldownEntry) -> float:
        base = float(e.trigger.meta.get("_base_duration", 0.0))
        return base + e.duration_s

    def _handle_trigger(self, rt: CamRuntime, tg: Trigger, fr: Frame) -> None:
        tg.t_media, tg.frame_index = fr.t_media, fr.index
        action, entry = rt.cooldown.check(tg)
        rt.banners[tg.dedupe_key()] = (time.time() + 3.0, tg)
        if action == "extend":
            self.metrics.inc("triggers_extended")
            self.store.update_duration(entry.event_id, self._duration(entry))
            return
        entry.event_id = new_event_id()
        start = tg.meta.get("t_event_start")
        tg.meta["_base_duration"] = max(0.0, tg.t - float(start)) if start is not None else 0.0
        self._dur[entry.event_id] = entry
        if len(self._dur) > 4000:
            for k in list(self._dur)[:2000]:
                self._dur.pop(k, None)
        self.metrics.inc("triggers")
        frames = rt.ring.pick(fr.t, self.p["vlm"]["composite_span_s"], self.p["vlm"]["composite_frames"])
        if not frames:
            frames = [Frame(fr.camera_id, fr.index, fr.t, fr.image)]
        comp, n, times = build_composite(frames, self.p["vlm"]["composite_max_side"])
        h, w = fr.image.shape[:2]
        s = min(1.0, 960 / max(h, w))
        import cv2

        trig_img = cv2.resize(fr.image, (int(w * s), int(h * s))) if s < 1 else fr.image.copy()
        req = VLMRequest(event_id=entry.event_id, trigger=tg, composite=comp, trigger_frame=trig_img,
                         sample_count=n, t_trigger=time.time(), frame_times=tuple(times),
                         source_path=str(rt.cfg.source) if rt.cfg.is_file else "",
                         priority=tg.heuristic_score, t_enqueued=time.time())
        log.info("[%s] trigger %s score=%.2f: %s", tg.camera_id, tg.trigger_type, tg.heuristic_score, tg.reason)
        if not self.vlm_enabled:
            self._persist_from_request(req, None, "VLM disabled (heuristics-only mode)")
            return
        for ev in self.vlm_q.put(req, tg.heuristic_score):
            self.metrics.inc("dropped_triggers")
            self._persist_from_request(ev, None, "VLM queue saturated: candidate not verified")
        self.metrics.set("vlm_queue_depth", self.vlm_q.qsize())

    # ------------------------------------------------------------------ VLM
    def _vlm_loop(self) -> None:
        if self.vlm_enabled and hasattr(self.verifier, "warm_up"):
            try:
                self.verifier.warm_up()
            except Exception:  # noqa: BLE001
                pass
        while not self.stop_event.is_set():
            req = self.vlm_q.get(0.2)
            if req is None:
                continue
            self._busy_inc("vlm", 1)
            try:
                self.metrics.set("vlm_queue_depth", self.vlm_q.qsize())
                self.metrics.observe("trigger_to_vlm_start", max(0.0, time.time() - req.t_trigger))
                try:
                    res = self.verifier.verify(req)
                except Exception as e:  # noqa: BLE001 - verifier must never break the loop
                    res = VLMResult(ok=False, error=f"verifier crashed: {type(e).__name__}: {e}",
                                    t_start=time.time(), t_end=time.time())
                if res.ok:
                    self.metrics.observe("vlm", res.t_end - res.t_start)
                    self.metrics.set("last_vlm_latency_s", round(res.t_end - res.t_start, 3))
                else:
                    self.metrics.inc("vlm_failures")
                    self.metrics.error("vlm", res.error)
                self._persist_from_request(req, res, "")
            finally:
                self._busy_inc("vlm", -1)

    def _persist_from_request(self, req: VLMRequest, res: Optional[VLMResult], why: str) -> None:
        tg = req.trigger
        f = apply_verdict(req, res, self.p["vlm"], self.p["confidence"], why)
        entry = self._dur.get(req.event_id)
        summary = f["summary"]
        if f["verdict"] == "unverified":
            summary = f"[unverified] {tg.reason}"
        meta = {"heuristic_category": tg.category, "heuristic_reason": tg.reason, "zone": tg.zone,
                "sample_count": req.sample_count, "frame_times": list(req.frame_times),
                "category_changed": f["category_changed"],
                "trigger_meta": {k: v for k, v in tg.meta.items() if not k.startswith("_")},
                "event_start_t": tg.meta.get("t_event_start"), "frame_index": tg.frame_index}
        raw = ""
        if res is not None:
            raw = res.raw if res.raw else json.dumps({"error": res.error})
            meta.update(vlm_reason=res.reason, objects=res.objects, vlm_attempts=res.attempts,
                        summary_truncated=res.summary_truncated)
        else:
            raw = json.dumps({"error": f["error"]})
        rec = EventRecord(
            id=req.event_id, timestamp=tg.t, camera_id=tg.camera_id, category=f["category"],
            confidence_score=f["confidence_score"], summary=summary, keyframe_path="", verdict=f["verdict"],
            trigger_type=tg.trigger_type, heuristic_score=tg.heuristic_score, vlm_confidence=f["vlm_confidence"],
            track_ids=list(tg.track_ids), bboxes={str(k): [round(float(x), 1) for x in v] for k, v in tg.bboxes.items()},
            vlm_raw=raw, vlm_model=(res.model if res else "") or (self.p["vlm"]["model"] if self.vlm_enabled else ""),
            duration_s=self._duration(entry) if entry else 0.0, t_trigger=req.t_trigger,
            t_vlm_start=res.t_start if res else 0.0, t_vlm_end=res.t_end if res else 0.0,
            source_path=req.source_path, media_time_s=tg.t_media, error=f["error"], meta=meta)
        self.persist_q.put(PersistJob(rec, req.composite, req.trigger_frame, entry), self.stop_event)

    def _persist_sparse(self, rt: CamRuntime, fr: Frame) -> None:
        import cv2

        h, w = fr.image.shape[:2]
        s = min(1.0, 640 / max(h, w))
        img = cv2.resize(fr.image, (int(w * s), int(h * s))) if s < 1 else fr.image
        rec = EventRecord(id=new_event_id(), timestamp=fr.t, camera_id=rt.cfg.id, category=NONE_CATEGORY,
                          confidence_score=0.0, summary="", keyframe_path="", verdict="unverified",
                          trigger_type="scene_index", t_trigger=time.time(), source_path=str(rt.cfg.source) if rt.cfg.is_file else "",
                          media_time_s=fr.t_media, error="sparse scene-index frame (CLIP only, no VLM description)",
                          meta={"sparse_index": True})
        self.persist_q.put(PersistJob(rec, img, None), self.stop_event)

    # ---------------------------------------------------------- persistence
    def _persist_loop(self) -> None:
        while not self.stop_event.is_set():
            job = self.persist_q.get(0.2)
            if job is None:
                continue
            self._do_persist(job)

    def _do_persist(self, job: PersistJob) -> None:
        self._busy_inc("persist", 1)
        try:
            rec = job.record
            if job.entry is not None:  # refresh duration accumulated while queued
                rec.duration_s = self._duration(job.entry)
            persist_event(self.store, rec, job.composite, job.trigger_frame, self.p["paths"]["keyframes"],
                          self.embedder if self.embedder_ok else None, self.index if self.embedder_ok else None)
            if job.entry is not None:
                self.store.update_duration(rec.id, self._duration(job.entry))  # close the race
            stored = self.store.get_event(rec.id)
            if stored is not None and rec.trigger_type != "scene_index":
                self.metrics.observe("trigger_to_store", max(0.0, stored.t_stored - stored.t_trigger))
                for k, stage in (("embed_ms", "embed"), ("index_ms", "store")):
                    if k in stored.meta:
                        self.metrics.observe(stage, stored.meta[k] / 1000.0)
                self.event_counts[rec.category] = self.event_counts.get(rec.category, 0) + 1
                self.verdict_counts[rec.verdict] = self.verdict_counts.get(rec.verdict, 0) + 1
        except Exception as e:  # noqa: BLE001
            log.exception("persist failed for %s", job.record.id)
            self.metrics.error("persist", f"{type(e).__name__}: {e}")
        finally:
            self._busy_inc("persist", -1)

    # ----------------------------------------------------------- status/jobs
    def _status_loop(self) -> None:
        iv = self.p["status_write_interval_s"]
        while not self.stop_event.wait(iv):
            if self.watch_jobs:
                try:
                    self._poll_jobs()
                except Exception as e:  # noqa: BLE001
                    self.metrics.error("jobs", str(e))
            try:
                self._write_status()
            except Exception as e:  # noqa: BLE001
                log.error("status write failed: %s", e)

    def build_status(self, final: bool = False) -> Dict[str, Any]:
        snap = self.metrics.snapshot()
        cams = {}
        with self._cams_lock:
            items = list(self.cams.items())
        any_bad = False
        for cid, rt in items:
            st = rt.source.status()
            any_bad |= st["state"] in ("unavailable", "failed", "reconnecting")
            cams[cid] = {**st, "fps": round(self.metrics.fps(cid), 2), "queue_depth": rt.queue.qsize(),
                         "dropped_frames": rt.queue.dropped, "last_processed_t": rt.last_t,
                         "name": rt.cfg.name, "frames_processed": rt.frames, "job_id": rt.job_id}
        vlm_state = ("disabled" if not self.vlm_enabled else
                     ("ok" if snap["counters"].get("vlm_failures", 0) == 0 or snap["stages"].get("vlm") else "degraded"))
        health = "stopped" if final else ("degraded" if any_bad or vlm_state == "degraded" or not self.embedder_ok else "ok")
        return {"updated_at": time.time(), "pid": __import__("os").getpid(), "version": __version__, "health": health,
                "profile": self.p.get("name"), "vlm_model": self.p["vlm"]["model"] if self.vlm_enabled else None,
                "vlm_state": vlm_state, "vlm_info": self.vlm_info, "embedder_ok": self.embedder_ok,
                "detector_device": self.metrics.gauges.get("detector_device"), "cameras": cams,
                "vlm_queue_depth": self.vlm_q.qsize(), "vlm_queue_max_depth": self.vlm_q.max_depth,
                "dropped_triggers": self.vlm_q.dropped, "persist_queue_depth": self.persist_q.qsize(),
                "event_counts": dict(self.event_counts), "verdict_counts": dict(self.verdict_counts),
                "latest_detect_latency_ms": (snap["stages"].get("detect") or {}).get("last_ms"),
                "latest_vlm_latency_s": snap["gauges"].get("last_vlm_latency_s"), "metrics": snap,
                "uptime_s": time.time() - self.started_at}

    def _write_status(self, final: bool = False) -> None:
        if not self.write_live:
            return
        atomic_write_json(self.live_dir / "status.json", self.build_status(final))

    def _poll_jobs(self) -> None:
        if not self._jobs_dir.exists():
            return
        for f in sorted(self._jobs_dir.glob("job_*.json")):
            try:
                job = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if job.get("state") == "running":
                rt = self.cams.get(job.get("camera_id", ""))
                if rt is not None and rt.job_id == job["job_id"] and rt.source.eof.is_set():
                    job["state"] = "done" if rt.source.state != "failed" else "failed"
                    job["message"] = rt.source.error or f"processed {rt.frames} frames"
                    job["finished_at"] = time.time()
                    atomic_write_json(f, job)
                continue
            if job.get("state") != "queued":
                continue
            try:
                template = job.get("camera_id")
                base = {c.id: c for c in self._initial_cams}.get(template)
                raw = {"id": job["camera_id"], "source": job["source"], "target_fps": job.get("target_fps", 10)}
                if base is not None:
                    raw.update(name=base.name, zones=base.zones, entry_lines=base.entry_lines, thresholds=base.thresholds,
                               target_fps=base.target_fps)
                cam = parse_camera(raw, 10.0, resolve_paths=False)
                self.add_camera(cam, speed=float(job.get("speed", 0)), job_id=job["job_id"])
                job.update(state="running", message="processing", started_at=time.time())
            except Exception as e:  # noqa: BLE001
                job.update(state="failed", message=f"{type(e).__name__}: {e}", finished_at=time.time())
            atomic_write_json(f, job)


# ------------------------------------------------------------------- CLI
def build_pipeline(args: argparse.Namespace) -> Pipeline:
    from .device import detect_hardware, select_profile

    hw = detect_hardware()
    prof_path = args.profile
    if not prof_path:
        p, why = select_profile(hw, CONFIG_DIR)
        prof_path = str(p)
        print(f"[vigil] auto-selected profile {p.name}: {why}")
    profile = load_profile(prof_path)
    setup_logging(profile["logging"]["level"], profile["logging"]["debug"])
    cams = load_cameras(args.config, profile["ingest"]["target_fps"])
    for ov in args.source or []:
        cid, _, path = ov.partition("=")
        for c in cams:
            if c.id == cid:
                c.source = path
    if args.only:
        cams = [c for c in cams if c.id in args.only]
    embedder = None
    if getattr(args, "test_stub_embedder", False):
        from .embeddings import HashEmbedder

        embedder = HashEmbedder()
        log.warning("USING THE TEST STUB EMBEDDER: semantic search results are meaningless")
    return Pipeline(profile, cams, speed=args.speed, loop=args.loop, vlm_enabled=False if args.no_vlm else None,
                    watch_jobs=not args.no_jobs, hw=hw, embedder=embedder)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Vigil video pipeline")
    ap.add_argument("--config", default="configs/cameras.yaml")
    ap.add_argument("--profile", default=None, help="profile YAML (default: auto-detect)")
    ap.add_argument("--speed", type=float, default=1.0, help="file playback speed: 1=realtime, N>1 faster, 0=as fast as possible")
    ap.add_argument("--loop", action="store_true", help="loop file sources")
    ap.add_argument("--no-vlm", action="store_true", help="heuristics only (ablation baseline)")
    ap.add_argument("--no-jobs", action="store_true", help="do not watch data/jobs for UI uploads")
    ap.add_argument("--exit-when-done", action="store_true", help="exit when all file sources are processed")
    ap.add_argument("--only", nargs="*", help="only these camera ids")
    ap.add_argument("--test-stub-embedder", action="store_true",
                    help="TEST ONLY: deterministic hash embedder (no CLIP weights needed). Search results are NOT semantic.")
    ap.add_argument("--source", action="append", help="override a camera source: cam1=path/to/video.mp4")
    args = ap.parse_args(argv)
    try:
        pipe = build_pipeline(args)
    except ConfigError as e:
        print(f"[vigil] configuration error: {e}")
        return 2
    stop = threading.Event()

    def _sig(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _sig)
    try:
        signal.signal(signal.SIGTERM, _sig)
    except (AttributeError, ValueError):
        pass
    try:
        pipe.start()
    except DetectorError as e:
        print(f"[vigil] cannot start detector: {e}")
        return 3
    try:
        while not stop.is_set():
            if args.exit_when_done and pipe.run_until_done(timeout=1.0):
                break
            stop.wait(0.5)
    finally:
        pipe.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
