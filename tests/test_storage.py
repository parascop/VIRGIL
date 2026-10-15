import json
import sqlite3
import threading

import numpy as np
import pytest

from helpers import make_profile
from vigil.embeddings import HashEmbedder, VectorIndex
from vigil.storage import (EventStore, index_event, persist_event, rebuild_index, reindex_pending, resolve_keyframe)
from vigil.types import EventRecord


def rec(i="e1", **kw):
    d = dict(id=i, timestamp=1000.0 + len(i), camera_id="cam1", category="person down", confidence_score=0.7,
             summary="a person lies on the floor", keyframe_path="", verdict="verified", trigger_type="person_down",
             heuristic_score=0.6, vlm_confidence=0.8, track_ids=[3], bboxes={"3": [1, 2, 3, 4]}, vlm_raw="{}",
             vlm_model="m", duration_s=2.0, t_trigger=999.0, t_vlm_start=999.5, t_vlm_end=1000.0, t_stored=0.0)
    d.update(kw)
    return EventRecord(**d)


@pytest.fixture
def store(tmp_path):
    return EventStore(tmp_path / "db" / "vigil.db")


def img(v=100):
    return np.full((48, 64, 3), v, np.uint8)


def test_creation_wal_and_indexes(store):
    assert store.conn().execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    idx = {r["name"] for r in store.conn().execute("PRAGMA index_list(events)")}
    assert {"idx_events_cam_ts", "idx_events_category"} <= idx
    cols = {r["name"] for r in store.conn().execute("PRAGMA table_info(events)")}
    required = {"id", "timestamp", "camera_id", "category", "confidence_score", "summary", "keyframe_path", "verdict",
                "trigger_type", "heuristic_score", "vlm_confidence", "track_ids", "bboxes", "vlm_raw", "vlm_model",
                "duration_s", "t_trigger", "t_vlm_start", "t_vlm_end", "t_stored"}
    assert required <= cols
    info = {r["name"]: r["type"] for r in store.conn().execute("PRAGMA table_info(events)")}
    assert all(info[c] == "REAL" for c in ("timestamp", "t_trigger", "t_vlm_start", "t_vlm_end", "t_stored"))


def test_insert_get_and_json_fields(store):
    assert store.insert_event(rec())
    got = store.get_event("e1")
    assert got.track_ids == [3] and got.bboxes == {"3": [1, 2, 3, 4]} and got.verdict == "verified"
    assert isinstance(got.meta, dict)


def test_duplicate_id_is_ignored_not_overwritten(store):
    assert store.insert_event(rec(summary="first"))
    assert store.insert_event(rec(summary="second")) is False
    assert store.get_event("e1").summary == "first"
    assert store.count() == 1
    assert store.insert_event(rec(summary="replaced"), replace=True)
    assert store.get_event("e1").summary == "replaced"


def test_transaction_rolls_back_on_error(store):
    with pytest.raises(RuntimeError):
        with store.transaction() as c:
            c.execute("INSERT INTO events (id,timestamp,camera_id,category,verdict) VALUES ('x',1,'c','none','rejected')")
            raise RuntimeError("boom")
    assert store.get_event("x") is None
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_event(rec("bad", verdict="maybe"))  # CHECK constraint
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_event(rec("bad2", category="made up"))
    assert store.count() == 0


def test_rejected_and_unverified_are_kept(store):
    store.insert_event(rec("r1", verdict="rejected", category="none"))
    store.insert_event(rec("u1", verdict="unverified", error="ollama down"))
    assert store.counts_by_verdict() == {"rejected": 1, "unverified": 1}
    assert {e.id for e in store.query(verdicts=("rejected",))} == {"r1"}


def test_update_duration_only_grows(store):
    store.insert_event(rec(duration_s=2.0))
    assert store.update_duration("e1", 9.0) and store.get_event("e1").duration_s == 9.0
    assert store.update_duration("e1", 3.0) is False
    assert store.update_duration("missing", 3.0) is False


def test_concurrent_writers_do_not_corrupt(store):
    errs = []

    def w(k):
        try:
            for j in range(25):
                store.insert_event(rec(f"t{k}_{j}"))
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=w, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and store.count() == 100


def test_time_camera_category_filters(store):
    store.insert_event(rec("a", timestamp=100.0, camera_id="cam1", category="fighting"))
    store.insert_event(rec("b", timestamp=200.0, camera_id="cam2", category="fighting"))
    store.insert_event(rec("c", timestamp=300.0, camera_id="cam2", category="tailgating", confidence_score=0.2))
    assert [e.id for e in store.query(camera_id="cam2")] == ["c", "b"]
    assert [e.id for e in store.query(category="fighting", time_from=150.0)] == ["b"]
    assert [e.id for e in store.query(time_to=250.0, min_confidence=0.5)] == ["b", "a"]


# ------------------------------------------------------- persistence + index
@pytest.fixture
def env(tmp_path):
    prof = make_profile(tmp_path)
    st = EventStore(prof["paths"]["sqlite"])
    emb = HashEmbedder()
    idx = VectorIndex(prof["paths"]["chroma"], emb.tag, emb.dim)
    return prof, st, emb, idx


def test_persist_saves_keyframe_row_and_embeddings(env):
    prof, st, emb, idx = env
    r = persist_event(st, rec(), img(), img(), prof["paths"]["keyframes"], emb, idx)
    got = st.get_event("e1")
    assert resolve_keyframe(prof["paths"]["keyframes"], got.keyframe_path).exists()
    assert got.index_status == "indexed" and got.t_stored > 0
    assert idx.counts() == {"image": 1, "summary": 1}
    assert "embed_ms" in got.meta and "trigger_frame_path" in got.meta
    assert not list(__import__("pathlib").Path(prof["paths"]["keyframes"]).rglob("*.tmp"))  # atomic writes


def test_embedding_failure_does_not_lose_event(env):
    prof, st, emb, idx = env

    class Boom(HashEmbedder):
        def embed_images(self, images):
            raise RuntimeError("OOM in CLIP")

    persist_event(st, rec(), img(), None, prof["paths"]["keyframes"], Boom(), idx)
    got = st.get_event("e1")
    assert got is not None and got.index_status == "failed" and "OOM" in got.index_error
    assert reindex_pending(st, emb, idx, prof["paths"]["keyframes"]) == 1  # repaired later
    assert st.get_event("e1").index_status == "indexed"


def test_interrupted_persistence_leaves_pending_row_that_can_be_repaired(env):
    """Process dies after the SQL insert but before embedding: row must survive."""
    prof, st, emb, idx = env
    r = rec()
    from vigil.storage import keyframe_rel_path
    import cv2, pathlib
    rel = keyframe_rel_path(r.camera_id, r.timestamp, r.id)
    p = pathlib.Path(prof["paths"]["keyframes"]) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img())
    r.keyframe_path, r.index_status = rel, "pending"
    st.insert_event(r)  # ... crash here ...
    assert [e.id for e in st.unindexed()] == ["e1"]
    assert reindex_pending(st, emb, idx, prof["paths"]["keyframes"]) == 1
    assert idx.counts()["image"] == 1


def test_missing_keyframe_behaviour(env):
    prof, st, emb, idx = env
    persist_event(st, rec(), None, None, prof["paths"]["keyframes"], emb, idx)  # no image at all
    got = st.get_event("e1")
    assert got.keyframe_path == "" and got.index_status == "indexed"  # summary-only vector
    assert idx.counts() == {"image": 0, "summary": 1}
    persist_event(st, rec("e2", summary=""), None, None, prof["paths"]["keyframes"], emb, idx)
    assert st.get_event("e2").index_status == "skipped"


def test_rebuild_index_from_sqlite_and_keyframes(env):
    prof, st, emb, idx = env
    for i in range(3):
        persist_event(st, rec(f"e{i}", summary=f"event number {i}"), img(40 * i + 20), None, prof["paths"]["keyframes"], emb, idx)
    before = {e.id: e for e in st.iter_events()}
    # corrupt/lose the vector index entirely, and delete one keyframe
    idx.reset()
    assert idx.counts() == {"image": 0, "summary": 0}
    kp = resolve_keyframe(prof["paths"]["keyframes"], before["e1"].keyframe_path)
    kp.unlink()
    stats = rebuild_index(st, emb, idx, prof["paths"]["keyframes"])
    assert stats["total"] == 3 and stats["missing_keyframe"] == 1
    assert idx.counts() == {"image": 2, "summary": 3}
    after = {e.id: e for e in st.iter_events()}
    assert all(after[k].summary == before[k].summary and after[k].category == before[k].category for k in before)


def test_scores_among_candidates_and_dimension_validation(env):
    prof, st, emb, idx = env
    persist_event(st, rec("e1", summary="red shirt person"), img(), None, prof["paths"]["keyframes"], emb, idx)
    persist_event(st, rec("e2", summary="empty hallway"), img(200), None, prof["paths"]["keyframes"], emb, idx)
    q = emb.embed_texts(["red shirt person"])[0]
    s = idx.scores("summary", q, ["e1", "e2", "ghost"])
    assert set(s) == {"e1", "e2"} and s["e1"] > s["e2"]
    with pytest.raises(ValueError):
        idx.upsert("bad", np.ones(7, np.float32), None, {"camera_id": "c"})
    with pytest.raises(ValueError):
        idx.upsert("nan", np.full(emb.dim, np.nan, np.float32), None, {"camera_id": "c"})


def test_unreadable_keyframe_is_recorded(env):
    prof, st, emb, idx = env
    persist_event(st, rec(), img(), None, prof["paths"]["keyframes"], emb, idx)
    kp = resolve_keyframe(prof["paths"]["keyframes"], st.get_event("e1").keyframe_path)
    kp.write_bytes(b"not a jpeg")
    idx.reset()
    assert index_event(st, st.get_event("e1"), prof["paths"]["keyframes"], emb, idx) == "indexed"
    assert "unreadable" in st.get_event("e1").index_error  # image vector skipped, summary vector kept
