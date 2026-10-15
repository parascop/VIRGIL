"""Pipeline tests with a scripted detector and fake/mock VLMs. They exercise the real
ingestion, tracking-book, heuristics, queues, persistence and shutdown code paths;
YOLO and Ollama are replaced by clearly-labelled fixtures."""
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers import (FakeDetector, FakeVerifier, MockOllama, cam, make_profile, make_video, person_down_script)
from vigil.embeddings import HashEmbedder, VectorIndex
from vigil.ingest import CameraSource, DropOldestQueue, RingBuffer, probe_video
from vigil.pipeline import BoundedPriorityQueue, Pipeline
from vigil.storage import EventStore
from vigil.types import Frame
from vigil.vlm import VLMVerifier


def run_pipeline(tmp_path, script, verifier=None, vlm=True, n_frames=60, speed=0.0, detector=None, profile=None,
                 camera=None, video=None, embedder=HashEmbedder()):
    prof = profile or make_profile(tmp_path)
    video = video or make_video(tmp_path / "v.mp4", n_frames=n_frames)
    c = camera or cam("cam1", source=video, target_fps=10)
    det = detector or FakeDetector({"cam1": script})
    store = EventStore(prof["paths"]["sqlite"])
    idx = VectorIndex(prof["paths"]["chroma"], embedder.tag, embedder.dim)
    p = Pipeline(prof, [c], speed=speed, vlm_enabled=vlm, detector=det, verifier=verifier, embedder=embedder,
                 store=store, index=idx, watch_jobs=False)
    p.start()
    done = p.run_until_done(timeout=60)
    p.stop()
    return p, store, prof, done


# ------------------------------------------------------------------ ingestion
def test_probe_valid_empty_and_missing(tmp_path):
    v = make_video(tmp_path / "ok.mp4", 20)
    info = probe_video(v)
    assert info["ok"] and info["width"] == 160 and info["frame_count"] == 20 and info["fps"] == pytest.approx(10, abs=0.5)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    assert not probe_video(str(empty))["ok"]
    assert not probe_video(str(tmp_path / "nope.mp4"))["ok"]
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    assert not probe_video(str(junk))["ok"]


def _source(tmp_path, video, fps=10, speed=0.0, qmax=100, **kw):
    prof = make_profile(tmp_path)
    q = DropOldestQueue(qmax, "block" if speed == 0 else "drop_oldest")
    ring = RingBuffer(4.0, fps)
    stop = threading.Event()
    s = CameraSource(cam("camX", source=video, target_fps=fps), q, ring, stop, prof["ingest"], speed=speed, **kw)
    return s, q, ring, stop


def test_file_ingest_samples_at_target_fps(tmp_path):
    v = make_video(tmp_path / "v30.mp4", n_frames=90, fps=30)  # 3 s @ 30fps
    s, q, ring, stop = _source(tmp_path, v, fps=10)
    s.start(); s.join(15)
    frames = []
    while (f := q.get(0)) is not None:
        frames.append(f)
    assert s.state == "finished" and 29 <= len(frames) <= 31
    ts = [f.t_media for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:])) and ts[1] - ts[0] == pytest.approx(0.1, abs=0.04)
    assert all(f.camera_id == "camX" for f in frames)
    assert len(ring) <= ring.max_frames


def test_realtime_pacing_and_speedup(tmp_path):
    v = make_video(tmp_path / "v.mp4", n_frames=20, fps=10)  # 2 s
    s, q, ring, stop = _source(tmp_path, v, speed=4.0)
    t0 = time.time(); s.start(); s.join(15)
    assert 0.35 < time.time() - t0 < 1.5  # ~2s / 4


def test_queue_is_bounded_drop_oldest_and_counts(tmp_path):
    v = make_video(tmp_path / "v.mp4", n_frames=60, fps=10)
    s, q, ring, stop = _source(tmp_path, v, speed=50.0, qmax=5)
    s.start(); s.join(15)
    assert q.qsize() <= 5 and q.dropped > 0 and q.max_depth <= 5
    last = None
    while (f := q.get(0)) is not None:
        last = f
    assert last.t_media >= 5.0  # newest frames survive, oldest were dropped


def test_ring_buffer_bounded_and_composite_picks_chronological(tmp_path):
    ring = RingBuffer(4.0, 10)
    for i in range(200):
        ring.add(Frame("c", i, 100 + i / 10, np.zeros((64, 64, 3), np.uint8)))
    assert len(ring) <= ring.max_frames and len(ring) <= 60
    picked = ring.pick(119.9, 3.0, 4)
    assert len(picked) == 4 and [f.t for f in picked] == sorted(f.t for f in picked)
    assert picked[-1].index == 199 and picked[0].t == pytest.approx(116.9, abs=0.1)


def test_invalid_empty_sources_fail_cleanly(tmp_path):
    for name, content in (("missing.mp4", None), ("empty.mp4", b""), ("junk.mp4", b"x" * 500)):
        path = tmp_path / name
        if content is not None:
            path.write_bytes(content)
        s, q, ring, stop = _source(tmp_path, str(path))
        s.start(); s.join(10)
        assert s.state == "failed" and s.error and s.eof.is_set() and not s.is_alive()


def test_live_reconnect_with_backoff_and_clean_stop(tmp_path):
    prof = make_profile(tmp_path)
    prof["ingest"]["reconnect"].update(initial_s=0.05, factor=2.0, max_s=0.2, max_attempts=0)
    q = DropOldestQueue(5); ring = RingBuffer(4.0, 10); stop = threading.Event()
    s = CameraSource(cam("cam9", source="rtsp://127.0.0.1:1/none"), q, ring, stop, prof["ingest"])
    s.start()
    time.sleep(1.5)
    st = s.status()
    assert st["state"] in ("unavailable", "reconnecting", "connecting") and st["reconnects"] >= 2
    assert st["state"] != "connected"  # never marked healthy while down
    stop.set(); s.join(5)
    assert not s.is_alive()


# ----------------------------------------------------------- end-to-end paths
def test_end_to_end_verified_event_with_fake_vlm(tmp_path):
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), FakeVerifier("verify"))
    assert done
    evs = store.query(verdicts=("verified", "rejected", "unverified"))
    assert len(evs) == 1, "cooldown must collapse the persisting candidate into ONE event"
    e = evs[0]
    assert (e.category, e.verdict, e.camera_id, e.trigger_type) == ("person down", "verified", "cam1", "person_down")
    assert e.duration_s > 0 and e.keyframe_path and e.index_status == "indexed" and e.vlm_model == "fake-vlm"
    assert e.t_trigger <= e.t_vlm_start <= e.t_vlm_end <= e.t_stored
    assert e.meta["sample_count"] == 4 and len(e.meta["frame_times"]) == 4  # full chronological 2x2 composite
    assert e.media_time_s is not None and e.source_path.endswith("v.mp4")
    assert (Path(prof["paths"]["keyframes"]) / e.keyframe_path).exists()
    assert p.metrics.counters["triggers_extended"] > 0


def test_vlm_rejection_is_stored_for_evaluation(tmp_path):
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), FakeVerifier("reject"))
    e = store.query(verdicts=("rejected",))[0]
    assert e.category == "none" and e.summary and e.verdict == "rejected" and e.trigger_type == "person_down"
    assert store.query(verdicts=("verified", "unverified")) == []


def test_vlm_timeout_yields_unverified_not_crash(tmp_path):
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), FakeVerifier("timeout"))
    assert done
    e = store.query()[0]
    assert e.verdict == "unverified" and e.category == "person down" and e.vlm_confidence == 0
    assert "timed out" in e.error and e.summary.startswith("[unverified]")


def test_dead_ollama_mid_run_produces_unverified_events(tmp_path):
    """Real VLMVerifier talking to a server that is shut down = 'killing Ollama'."""
    prof = make_profile(tmp_path)
    m = MockOllama(); m.__enter__(); url = m.url; m.__exit__()
    prof["vlm"].update(base_url=url, connect_timeout_s=0.5, timeout_s=1)
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), VLMVerifier(prof["vlm"]), profile=prof)
    e = store.query()[0]
    assert done and e.verdict == "unverified" and "connect" in e.error.lower()
    assert p.metrics.counters["vlm_failures"] >= 1


def test_real_verifier_against_mock_ollama_verifies(tmp_path):
    good = {"category": "person down", "vlm_confidence": 0.9, "summary": "Person on floor.", "objects": ["person"], "reason": "r"}
    with MockOllama([good]) as m:
        prof = make_profile(tmp_path)
        prof["vlm"]["base_url"] = m.url
        p, store, prof, done = run_pipeline(tmp_path, person_down_script(), VLMVerifier(prof["vlm"]), profile=prof)
    e = store.query()[0]
    assert e.verdict == "verified" and e.vlm_confidence == pytest.approx(0.9) and e.summary == "Person on floor."


def test_heuristics_only_mode(tmp_path):
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), None, vlm=False)
    e = store.query()[0]
    assert e.verdict == "unverified" and "heuristics-only" in e.error and e.vlm_model == ""


def test_slow_vlm_never_blocks_ingestion_and_queue_stays_bounded(tmp_path):
    """Many candidates + a slow VLM: all frames are still processed, the VLM
    queue never exceeds its bound and overflow is persisted, not lost."""
    prof = make_profile(tmp_path)
    prof["vlm"]["queue_maxsize"] = 2
    prof["cooldown_s"] = 0.0  # allow many candidates
    prof["heuristics"]["emit_interval_s"] = 0.0
    script = person_down_script(n_still=120)
    p, store, prof, done = run_pipeline(tmp_path, script, FakeVerifier("verify", delay=0.4), profile=prof, n_frames=160)
    assert done
    assert p.detector.calls["cam1"] == 160  # every sampled frame was detected despite the slow VLM
    assert p.vlm_q.max_depth <= 2
    evs = store.query(limit=1000)
    n_unver = sum(e.verdict == "unverified" for e in evs)
    assert p.vlm_q.dropped == n_unver > 0
    assert all("saturated" in e.error for e in evs if e.verdict == "unverified")
    assert p.metrics.counters["triggers"] == len(evs)  # nothing silently lost


def test_inference_failures_are_survived(tmp_path):
    det = FakeDetector({"cam1": person_down_script()}, fail_every=5)
    p, store, prof, done = run_pipeline(tmp_path, [], FakeVerifier(), detector=det)
    assert done and p.metrics.counters["errors"] > 0
    assert p.detect_failures > 0


def test_corrupt_source_does_not_crash_pipeline(tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"\x00" * 64)
    p, store, prof, done = run_pipeline(tmp_path, [], FakeVerifier(), video=str(bad))
    assert done and store.count() == 0
    st = p.build_status()
    assert st["cameras"]["cam1"]["state"] == "failed" and st["cameras"]["cam1"]["error"]


def test_embedder_failure_keeps_events(tmp_path):
    class NoClip(HashEmbedder):
        def load(self):
            raise RuntimeError("weights unavailable offline")

    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), FakeVerifier(), embedder=NoClip())
    e = store.query()[0]
    assert done and e.verdict == "verified" and e.index_status == "failed" and not p.embedder_ok


def test_live_frames_status_and_clean_shutdown(tmp_path):
    p, store, prof, done = run_pipeline(tmp_path, person_down_script(), FakeVerifier())
    live = Path(prof["paths"]["live"])
    jpg = live / "cam1.jpg"
    assert jpg.exists() and cv2.imread(str(jpg)) is not None
    assert not list(live.glob("*.tmp"))  # atomic writes leave no temp files
    st = json.loads((live / "status.json").read_text())
    assert st["health"] == "stopped" and "cam1" in st["cameras"] and "metrics" in st
    assert {"vlm_queue_depth", "dropped_triggers", "event_counts", "latest_vlm_latency_s"} <= set(st)
    assert not any(t.is_alive() for t in p.threads) and not any(rt.source.is_alive() for rt in p.cams.values())
    assert all(rt.source._cap is None for rt in p.cams.values())  # captures released


def test_stop_persists_queued_candidates_as_unverified(tmp_path):
    """Shutting down with candidates still waiting for the VLM must not lose them."""
    prof = make_profile(tmp_path)
    prof["cooldown_s"] = 0.0
    prof["heuristics"]["emit_interval_s"] = 0.0
    store = EventStore(prof["paths"]["sqlite"])
    emb = HashEmbedder()
    idx = VectorIndex(prof["paths"]["chroma"], emb.tag, emb.dim)
    video = make_video(tmp_path / "v.mp4", 140)
    p = Pipeline(prof, [cam("cam1", source=video)], speed=0.0, vlm_enabled=True,
                 detector=FakeDetector({"cam1": person_down_script(n_still=110)}), verifier=FakeVerifier("verify", delay=1.5),
                 embedder=emb, store=store, index=idx, watch_jobs=False)
    p.start()
    t0 = time.time()
    while p.vlm_q.qsize() < 2 and time.time() - t0 < 30:
        time.sleep(0.05)
    assert p.vlm_q.qsize() >= 2, "test setup: expected queued candidates"
    queued = p.vlm_q.qsize()
    p.stop(timeout=15)
    evs = store.query(limit=1000)
    assert p.vlm_q.qsize() == 0 and p.persist_q.qsize() == 0
    shut = [e for e in evs if e.verdict == "unverified" and "shut down" in e.error]
    assert len(shut) >= 1 and len(shut) <= queued
    assert not any(t.is_alive() for t in p.threads)


# --------------------------------------------------------------- queue units
def test_bounded_priority_queue_policies():
    q = BoundedPriorityQueue(2, "lowest_priority")
    assert q.put("a", 0.5) == [] and q.put("b", 0.9) == []
    assert q.put("c", 0.7) == ["a"]  # lowest priority evicted
    assert q.put("d", 0.1) == ["d"]  # new one is the lowest -> rejected
    assert q.get(0) == "b" and q.get(0) == "c" and q.get(0) is None
    q = BoundedPriorityQueue(2, "drop_oldest")
    q.put("a", 0.9); q.put("b", 0.1)
    assert q.put("c", 0.2) == ["a"]
    q = BoundedPriorityQueue(1, "reject_new")
    q.put("a", 0.1)
    assert q.put("b", 0.9) == ["b"] and q.dropped == 1


def test_drop_oldest_queue_blocks_when_configured():
    q = DropOldestQueue(2, "block")
    q.put(1); q.put(2)
    stop = threading.Event()
    t = threading.Thread(target=lambda: q.put(3, stop)); t.start()
    time.sleep(0.2); assert t.is_alive() and q.qsize() == 2
    q.get(0); t.join(2)
    assert not t.is_alive() and q.qsize() == 2 and q.dropped == 0
