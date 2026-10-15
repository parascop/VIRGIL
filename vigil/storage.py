"""SQLite event storage (authoritative) + the explicit persistence order.

Persistence order for one event (``persist_event``), chosen so a crash at any
step never makes an event silently vanish from search:

1. save keyframe images (temp file + ``os.replace``)
2. INSERT the event row (``index_status='pending'``)            <- event now durable
3. compute CLIP image/summary embeddings and upsert into ChromaDB
4. UPDATE index_status ('indexed' | 'skipped' | 'failed'), t_stored, timings

If step 3 fails or the process dies, the row stays in SQLite with a non-
'indexed' status; ``reindex_pending`` (run at startup) and ``rebuild_index``
repair the vector index from SQLite + keyframes. Search always starts from
SQLite candidates, so an un-indexed event is still returned (ranked without a
semantic score) instead of disappearing.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

from .types import CATEGORIES, NONE_CATEGORY, EventRecord
from .util import atomic_write_image

log = logging.getLogger("vigil.storage")

_CAT_SQL = ",".join(f"'{c}'" for c in CATEGORIES + (NONE_CATEGORY,))

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  timestamp REAL NOT NULL,
  camera_id TEXT NOT NULL,
  category TEXT NOT NULL CHECK(category IN ({_CAT_SQL})),
  confidence_score REAL NOT NULL DEFAULT 0,
  summary TEXT NOT NULL DEFAULT '',
  keyframe_path TEXT NOT NULL DEFAULT '',
  verdict TEXT NOT NULL CHECK(verdict IN ('verified','rejected','unverified')),
  trigger_type TEXT NOT NULL DEFAULT '',
  heuristic_score REAL NOT NULL DEFAULT 0,
  vlm_confidence REAL NOT NULL DEFAULT 0,
  track_ids TEXT NOT NULL DEFAULT '[]',
  bboxes TEXT NOT NULL DEFAULT '{{}}',
  vlm_raw TEXT NOT NULL DEFAULT '',
  vlm_model TEXT NOT NULL DEFAULT '',
  duration_s REAL NOT NULL DEFAULT 0,
  t_trigger REAL NOT NULL DEFAULT 0,
  t_vlm_start REAL NOT NULL DEFAULT 0,
  t_vlm_end REAL NOT NULL DEFAULT 0,
  t_stored REAL NOT NULL DEFAULT 0,
  source_path TEXT NOT NULL DEFAULT '',
  media_time_s REAL,
  error TEXT NOT NULL DEFAULT '',
  index_status TEXT NOT NULL DEFAULT 'pending',
  index_error TEXT NOT NULL DEFAULT '',
  meta TEXT NOT NULL DEFAULT '{{}}'
);
CREATE INDEX IF NOT EXISTS idx_events_cam_ts ON events(camera_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_verdict ON events(verdict);
"""

COLUMNS = ["id", "timestamp", "camera_id", "category", "confidence_score", "summary", "keyframe_path",
           "verdict", "trigger_type", "heuristic_score", "vlm_confidence", "track_ids", "bboxes", "vlm_raw",
           "vlm_model", "duration_s", "t_trigger", "t_vlm_start", "t_vlm_end", "t_stored", "source_path",
           "media_time_s", "error", "index_status", "index_error", "meta"]


def _row_to_record(r: sqlite3.Row) -> EventRecord:
    d = dict(r)
    for k, default in (("track_ids", []), ("bboxes", {}), ("meta", {})):
        try:
            d[k] = json.loads(d[k]) if d[k] else default
        except (TypeError, ValueError):
            d[k] = default
    return EventRecord(**d)


class EventStore:
    """Thread-safe SQLite wrapper: one connection per thread, WAL mode,
    explicit transactions, parameterised statements only."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._init_schema()

    # -- connections ---------------------------------------------------------
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=10000")
            self._local.c = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    def _init_schema(self) -> None:
        c = self.conn()
        c.executescript(SCHEMA)  # executescript commits implicitly; DDL is idempotent

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        c = self.conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            yield c
            c.execute("COMMIT")
        except BaseException:
            c.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except Exception:  # noqa: BLE001
                    pass
            self._conns.clear()
        self._local = threading.local()

    # -- writes ------------------------------------------------------------
    def insert_event(self, rec: EventRecord, replace: bool = False) -> bool:
        """Insert; a duplicate id is ignored (returns False) unless ``replace``."""
        vals = [getattr(rec, k) for k in COLUMNS]
        for i, k in enumerate(COLUMNS):
            if k in ("track_ids", "bboxes", "meta"):
                vals[i] = json.dumps(vals[i], default=str)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        try:
            with self.transaction() as c:
                c.execute(f"{verb} INTO events ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})", vals)
            return True
        except sqlite3.IntegrityError:
            # Only a duplicate primary key is tolerated. ("INSERT OR IGNORE" would
            # also silently swallow CHECK/NOT NULL violations and lose events.)
            if self.get_event(rec.id) is not None:
                return False
            raise

    def update_duration(self, event_id: str, duration_s: float) -> bool:
        with self.transaction() as c:
            return c.execute("UPDATE events SET duration_s=? WHERE id=? AND duration_s<?",
                             (float(duration_s), event_id, float(duration_s))).rowcount == 1

    def finalize(self, event_id: str, t_stored: float, index_status: str, index_error: str = "",
                 meta_updates: Optional[Dict[str, Any]] = None) -> None:
        with self.transaction() as c:
            row = c.execute("SELECT meta FROM events WHERE id=?", (event_id,)).fetchone()
            if row is None:
                return
            try:
                meta = json.loads(row["meta"] or "{}")
            except ValueError:
                meta = {}
            meta.update(meta_updates or {})
            c.execute("UPDATE events SET t_stored=?, index_status=?, index_error=?, meta=? WHERE id=?",
                      (t_stored, index_status, index_error[:500], json.dumps(meta, default=str), event_id))

    def set_keyframe(self, event_id: str, rel_path: str) -> None:
        with self.transaction() as c:
            c.execute("UPDATE events SET keyframe_path=? WHERE id=?", (rel_path, event_id))

    # -- reads -------------------------------------------------------------
    def get_event(self, event_id: str) -> Optional[EventRecord]:
        r = self.conn().execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return _row_to_record(r) if r else None

    def count(self) -> int:
        return int(self.conn().execute("SELECT COUNT(*) FROM events").fetchone()[0])

    def query(self, camera_id: Optional[str] = None, category: Optional[str] = None,
              verdicts: Optional[Sequence[str]] = None, time_from: Optional[float] = None,
              time_to: Optional[float] = None, min_confidence: Optional[float] = None,
              include_scene_index: bool = False, limit: int = 100, newest_first: bool = True,
              index_statuses: Optional[Sequence[str]] = None) -> List[EventRecord]:
        """Structured filtering. Every value is bound as a parameter."""
        where: List[str] = []
        args: List[Any] = []
        if camera_id:
            where.append("camera_id=?")
            args.append(camera_id)
        if category:
            where.append("category=?")
            args.append(category)
        if verdicts:
            where.append(f"verdict IN ({','.join('?' * len(verdicts))})")
            args.extend(verdicts)
        if time_from is not None:
            where.append("timestamp>=?")
            args.append(float(time_from))
        if time_to is not None:
            where.append("timestamp<=?")
            args.append(float(time_to))
        if min_confidence is not None:
            where.append("confidence_score>=?")
            args.append(float(min_confidence))
        if not include_scene_index:
            where.append("trigger_type<>'scene_index'")
        if index_statuses:
            where.append(f"index_status IN ({','.join('?' * len(index_statuses))})")
            args.extend(index_statuses)
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp " + ("DESC" if newest_first else "ASC") + " LIMIT ?"
        args.append(int(max(1, min(limit, 100000))))
        return [_row_to_record(r) for r in self.conn().execute(sql, args).fetchall()]

    def counts_by_category(self, since: Optional[float] = None, verdicts: Sequence[str] = ("verified", "unverified")) -> Dict[str, int]:
        sql = "SELECT category, COUNT(*) n FROM events WHERE trigger_type<>'scene_index'"
        args: List[Any] = []
        if verdicts:
            sql += f" AND verdict IN ({','.join('?' * len(verdicts))})"
            args += list(verdicts)
        if since is not None:
            sql += " AND timestamp>=?"
            args.append(since)
        sql += " GROUP BY category"
        return {r["category"]: r["n"] for r in self.conn().execute(sql, args)}

    def counts_by_verdict(self) -> Dict[str, int]:
        return {r["verdict"]: r["n"] for r in self.conn().execute(
            "SELECT verdict, COUNT(*) n FROM events WHERE trigger_type<>'scene_index' GROUP BY verdict")}

    def timing_rows(self, limit: int = 500) -> List[sqlite3.Row]:
        return self.conn().execute(
            "SELECT id, timestamp, camera_id, verdict, t_trigger, t_vlm_start, t_vlm_end, t_stored, meta "
            "FROM events WHERE t_trigger>0 ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()

    def iter_events(self, batch: int = 200) -> Iterator[EventRecord]:
        last = ""
        while True:
            rows = self.conn().execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (last, batch)).fetchall()
            if not rows:
                return
            for r in rows:
                yield _row_to_record(r)
            last = rows[-1]["id"]

    def unindexed(self, limit: int = 1000) -> List[EventRecord]:
        rows = self.conn().execute("SELECT * FROM events WHERE index_status NOT IN ('indexed','skipped') LIMIT ?", (limit,)).fetchall()
        return [_row_to_record(r) for r in rows]


# --------------------------------------------------------------- keyframes
def keyframe_rel_path(camera_id: str, timestamp: float, event_id: str, suffix: str = "") -> str:
    day = time.strftime("%Y%m%d", time.gmtime(timestamp))
    return f"{camera_id}/{day}/{event_id}{suffix}.jpg"


def resolve_keyframe(root: str | Path, rel: str) -> Optional[Path]:
    """Absolute path of a stored keyframe, or None if missing."""
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = Path(root) / rel
    return p if p.exists() else None


def persist_event(store: EventStore, rec: EventRecord, composite: Optional[np.ndarray],
                  trigger_frame: Optional[np.ndarray], keyframes_root: str | Path,
                  embedder: Any = None, index: Any = None) -> EventRecord:
    """Persist one event following the documented order. Never raises: every
    failure is recorded in ``error`` / ``index_error`` and the row is kept."""
    errors: List[str] = []
    # 1. keyframes (atomic)
    try:
        if composite is not None:
            rel = keyframe_rel_path(rec.camera_id, rec.timestamp, rec.id)
            atomic_write_image(Path(keyframes_root) / rel, composite, 90)
            rec.keyframe_path = rel
        if trigger_frame is not None:
            atomic_write_image(Path(keyframes_root) / keyframe_rel_path(rec.camera_id, rec.timestamp, rec.id, "_trigger"),
                               trigger_frame, 85)
            rec.meta["trigger_frame_path"] = keyframe_rel_path(rec.camera_id, rec.timestamp, rec.id, "_trigger")
    except Exception as e:  # noqa: BLE001
        errors.append(f"keyframe write failed: {e}")
        log.error("[%s] keyframe write failed: %s", rec.id, e)
    if errors:
        rec.error = (rec.error + "; " if rec.error else "") + "; ".join(errors)
    rec.index_status = "pending"
    # 2. durable row
    try:
        if not store.insert_event(rec):
            log.warning("[%s] duplicate event id ignored", rec.id)
            return rec
    except Exception as e:  # noqa: BLE001
        log.exception("[%s] SQLite insert failed", rec.id)
        raise RuntimeError(f"event could not be stored: {e}") from e
    # 3 + 4. embeddings and index
    index_event(store, rec, keyframes_root, embedder, index)
    return rec


def index_event(store: EventStore, rec: EventRecord, keyframes_root: str | Path, embedder: Any, index: Any,
                image_array: Optional[np.ndarray] = None) -> str:
    """Embed + upsert one stored event; update its index_status. Returns status."""
    t0 = time.time()
    meta_up: Dict[str, Any] = {}
    if embedder is None or index is None:
        store.finalize(rec.id, time.time(), "failed", "embedder/index unavailable", meta_up)
        return "failed"
    status, err = "indexed", ""
    try:
        img_vec = txt_vec = None
        img = image_array
        if img is None:
            kp = resolve_keyframe(keyframes_root, rec.keyframe_path)
            if kp is not None:
                import cv2

                img = cv2.imread(str(kp))
                if img is None:
                    err += "keyframe unreadable; "
        elif img is not None:
            pass
        te = time.time()
        if img is not None:
            img_vec = embedder.embed_images([img])[0]
        if rec.summary.strip():
            txt_vec = embedder.embed_texts([rec.summary])[0]
        meta_up["embed_ms"] = round((time.time() - te) * 1000, 2)
        ti = time.time()
        if img_vec is None and txt_vec is None:
            status, err = "skipped", err or "nothing to embed (no keyframe, no summary)"
        else:
            index.upsert(rec.id, img_vec, txt_vec, {"camera_id": rec.camera_id, "category": rec.category,
                                                    "verdict": rec.verdict, "timestamp": float(rec.timestamp)})
            if img_vec is None:
                err += "image embedding missing; "
        meta_up["index_ms"] = round((time.time() - ti) * 1000, 2)
    except Exception as e:  # noqa: BLE001 - vector failure must not lose the event
        status, err = "failed", f"{type(e).__name__}: {e}"
        log.error("[%s] indexing failed: %s", rec.id, err)
    store.finalize(rec.id, time.time(), status, err.strip(), meta_up)
    log.debug("[%s] indexed status=%s in %.1f ms", rec.id, status, (time.time() - t0) * 1000)
    return status


def rebuild_index(store: EventStore, embedder: Any, index: Any, keyframes_root: str | Path,
                  progress: Any = None) -> Dict[str, int]:
    """Drop and rebuild the whole vector index from SQLite + keyframes.
    SQLite metadata is never modified except ``index_status``."""
    index.reset()
    stats = {"total": 0, "indexed": 0, "skipped": 0, "failed": 0, "missing_keyframe": 0}
    for rec in store.iter_events():
        stats["total"] += 1
        if resolve_keyframe(keyframes_root, rec.keyframe_path) is None and rec.keyframe_path:
            stats["missing_keyframe"] += 1
        status = index_event(store, rec, keyframes_root, embedder, index)
        stats[status] = stats.get(status, 0) + 1
        if progress:
            progress(stats)
    return stats


def reindex_pending(store: EventStore, embedder: Any, index: Any, keyframes_root: str | Path) -> int:
    """Repair events whose indexing was interrupted or failed."""
    n = 0
    for rec in store.unindexed():
        if index_event(store, rec, keyframes_root, embedder, index) == "indexed":
            n += 1
    return n
