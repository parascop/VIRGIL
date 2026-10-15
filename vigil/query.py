"""Natural-language query -> validated structured filter -> parameterised SQL
-> optional CLIP ranking over the filtered candidates.

An LLM never writes or executes SQL. It (optionally) proposes a JSON filter that
is validated field by field; if it times out or returns garbage we fall back to
a deterministic regex/keyword parser that works fully offline.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .storage import EventStore
from .types import CATEGORIES, EventRecord

log = logging.getLogger("vigil.query")

CATEGORY_PATTERNS: List[Tuple[str, str]] = [
    ("person down", r"person[\s-]*down|persons?\s+down|man\s+down|woman\s+down|\bfall(?:s|en|ing)?\b|\bfell\b|collaps\w*|passed\s+out|faint\w*|lying\s+on\s+the\s+(?:floor|ground)"),
    ("fighting", r"\bfight\w*|brawl\w*|scuffl\w*|altercation\w*|punch\w*|assault\w*|violen\w*"),
    ("tailgating", r"tailgat\w*|piggy[\s-]*back\w*"),
    ("crowd incident", r"\bcrowd\w*|stampede|\bmob\b|congestion|overcrowd\w*|dispersal|dispersing"),
    ("suspicious activity", r"suspicious|loiter\w*|trespass\w*|intrud\w*|unauthori[sz]ed|prowl\w*|(?:abandoned|unattended)\s+(?:bag|object|luggage|package)"),
]
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])}
_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}
_STOP = set("""show find me all any the a an of in on at from to for with and or that which who what when where did do does
 events event incidents incident possible potential possibly there were was is are happened happen occurred occur
 camera cam get list display give see search look looking frames frame containing contains contain video clip clips
 footage please can you i want need everything anything latest recent newest""".split())


def category_from_text(text: str) -> Optional[str]:
    t = text.lower()
    for cat, pat in CATEGORY_PATTERNS:
        if re.search(pat, t):
            return cat
    return None


@dataclass
class StructuredFilter:
    category: Optional[str] = None
    camera_id: Optional[str] = None
    time_from: Optional[float] = None  # epoch seconds (UTC based)
    time_to: Optional[float] = None
    visual_query: Optional[str] = None
    limit: int = 20
    include_rejected: bool = False
    min_confidence: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    source: str = "fallback"  # llm | fallback | manual

    def to_dict(self, tz: Optional[ZoneInfo] = None) -> Dict[str, Any]:
        d = asdict(self)
        if tz is not None:
            for k in ("time_from", "time_to"):
                if d[k] is not None:
                    d[k] = datetime.fromtimestamp(d[k], tz=tz).isoformat(timespec="seconds")
        return d


# ------------------------------------------------------------- time parsing
_TIME = r"(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?|noon|midnight"


def _parse_clock(s: str) -> Optional[Tuple[int, int, bool]]:
    """-> (hour, minute, had_meridiem) from '2 pm', '14:30', 'noon'."""
    s = s.strip().lower()
    if s == "noon":
        return 12, 0, True
    if s == "midnight":
        return 0, 0, True
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m?\.?", s) or re.fullmatch(r"(\d{1,2})(?::(\d{2}))?()", s)
    if not m:
        return None
    h, mi, mer = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if mer:
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if mer == "p" else 0)
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return None
    if not mer and m.group(2) is None and h > 24:
        return None
    return h, mi, bool(mer)


def _day_start(d: datetime) -> datetime:
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def parse_time_expression(text: str, now: datetime) -> Tuple[Optional[float], Optional[float], List[str], str]:
    """Resolve time words using ``now`` (tz-aware). Returns (from, to, notes, text_without_time)."""
    t = text.lower()
    notes: List[str] = []
    a: Optional[datetime] = None
    b: Optional[datetime] = None
    rest = text

    def cut(m: "re.Match[str]") -> None:
        nonlocal rest
        rest = rest[:m.start()] + " " + rest[m.end():]

    today0 = _day_start(now)
    base = today0
    mday = re.search(r"\byesterday\b", t)
    if mday:
        base = today0 - timedelta(days=1)
    miso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    mmon = re.search(r"\b(?:on\s+)?(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    explicit_day = False
    try:
        if miso:
            base = today0.replace(year=int(miso.group(1)), month=int(miso.group(2)), day=int(miso.group(3)))
            explicit_day = True
        elif mmon:
            base = today0.replace(month=_MONTHS[mmon.group(1)], day=int(mmon.group(2)))
            if base > now:
                base = base.replace(year=base.year - 1)
            explicit_day = True
    except ValueError:
        notes.append("could not interpret the date; ignoring it")
        miso = mmon = None

    # --- clock ranges ------------------------------------------------------
    clock = r"(\d{1,2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?|noon|midnight)"
    mr = re.search(r"\b(?:between|from)\s+" + clock + r"\s+(?:and|to|until|-)\s+" + clock, t) or \
        re.search(r"\b" + clock + r"\s*(?:-|to)\s*" + clock + r"(?=\s|$|[.,?])", t)
    if mr and (re.search(r"[ap]\.?m|:|noon|midnight", mr.group(0)) or "between" in mr.group(0) or "from" in mr.group(0)):
        c1, c2 = _parse_clock(mr.group(1)), _parse_clock(mr.group(2))
        if c1 and c2:
            h1, m1, mer1 = c1
            h2, m2, mer2 = c2
            if mer2 and not mer1 and h1 <= 12 and "pm" in mr.group(2).replace(".", "") and h1 < 12:
                # "2 to 4 PM": the meridiem applies to both ends
                h1 = (h1 % 12) + 12
                mer1 = True
            elif mer2 and not mer1 and "am" in mr.group(2).replace(".", "") and h1 == 12:
                h1 = 0
            if not (mer1 or mer2):
                notes.append("no AM/PM given; times read as a 24-hour clock")
            a, b = base.replace(hour=h1, minute=m1), base.replace(hour=h2, minute=m2)
            if b <= a:
                b += timedelta(days=1)
            cut(mr)
            return a.timestamp(), b.timestamp(), notes, rest
    # --- relative windows ----------------------------------------------------
    mrel = re.search(r"\b(?:in\s+)?(?:the\s+)?(?:last|past|previous)\s+(\d+)?\s*(second|sec|minute|min|hour|hr|day|week)s?\b", t)
    if mrel:
        n = int(mrel.group(1) or 1)
        a, b = now - timedelta(seconds=n * _UNIT_SECONDS[mrel.group(2)]), now
        cut(mrel)
        return a.timestamp(), b.timestamp(), notes, rest
    mago = re.search(r"\b(\d+)\s*(minute|min|hour|hr|day)s?\s+ago\b", t)
    if mago:
        a, b = now - timedelta(seconds=int(mago.group(1)) * _UNIT_SECONDS[mago.group(2)]), now
        cut(mago)
        return a.timestamp(), b.timestamp(), notes, rest
    # --- parts of day ------------------------------------------------------
    parts = {"morning": (5, 12), "afternoon": (12, 17), "evening": (17, 22), "night": (20, 29), "tonight": (18, 29)}
    mp = re.search(r"\b(?:this\s+)?(morning|afternoon|evening|tonight|night)\b", t)
    if mp and not (mr or mrel):
        h1, h2 = parts[mp.group(1)]
        a, b = base + timedelta(hours=h1), base + timedelta(hours=h2)
        cut(mp)
        if mday:
            cut(mday)
        return a.timestamp(), b.timestamp(), notes, rest
    # --- one-sided clock bounds ---------------------------------------------
    ma = re.search(r"\b(after|since|before|until)\s+" + clock, t)
    if ma and _parse_clock(ma.group(2)):
        h, mi, mer = _parse_clock(ma.group(2))
        if not mer:
            notes.append("no AM/PM given; time read as a 24-hour clock")
        pt = base.replace(hour=h, minute=mi)
        if ma.group(1) in ("after", "since"):
            a, b = pt, base + timedelta(days=1)
        else:
            a, b = base, pt
        cut(ma)
        if mday:
            cut(mday)
        return a.timestamp(), b.timestamp(), notes, rest
    # --- whole days ------------------------------------------------------------
    if miso or mmon:
        m_ = miso or mmon
        a, b = base, base + timedelta(days=1)
        cut(m_)
        return a.timestamp(), b.timestamp(), notes, rest
    if mday:
        a, b = base, base + timedelta(days=1)
        cut(mday)
        return a.timestamp(), b.timestamp(), notes, rest
    mt = re.search(r"\btoday\b", t)
    if mt:
        a, b = today0, today0 + timedelta(days=1)
        cut(mt)
        return a.timestamp(), b.timestamp(), notes, rest
    mw = re.search(r"\bthis\s+week\b", t)
    if mw:
        a, b = today0 - timedelta(days=today0.weekday()), now
        cut(mw)
        return a.timestamp(), b.timestamp(), notes, rest
    return None, None, notes, rest


def _resolve_camera(n: int, known: Sequence[str]) -> str:
    for cand in (f"cam{n}", str(n), f"camera{n}", f"camera_{n}", f"cam_{n}"):
        if cand in known:
            return cand
    return f"cam{n}"


def parse_query_fallback(text: str, now: datetime, known_cameras: Sequence[str] = ()) -> StructuredFilter:
    """Deterministic parser: no model, no network."""
    f = StructuredFilter(source="fallback")
    text = (text or "").strip()
    if not text:
        f.notes.append("empty query: showing the most recent events")
        return f
    t = text
    mc = re.search(r"\b(?:camera|cam|cctv)\s*#?\s*(\d+)\b|\bcam(\d+)\b", t, re.I)
    if mc:
        f.camera_id = _resolve_camera(int(mc.group(1) or mc.group(2)), known_cameras)
        t = t[:mc.start()] + " " + t[mc.end():]
    f.time_from, f.time_to, notes, t = parse_time_expression(t, now)
    f.notes += notes
    f.category = category_from_text(t)
    if f.category:
        for cat, pat in CATEGORY_PATTERNS:
            if cat == f.category:
                t = re.sub(pat, " ", t, flags=re.I)
    ml = re.search(r"\b(?:top|first|latest|most\s+recent|newest|limit)\s+(\d+)\b|\b(\d+)\s+(?:results|events|incidents|matches)\b", t, re.I)
    if ml:
        f.limit = max(1, min(100, int(ml.group(1) or ml.group(2))))
        t = t[:ml.start()] + " " + t[ml.end():]
    if re.search(r"\b(?:rejected|false\s+alarms?|false\s+positives?)\b", t, re.I):
        f.include_rejected = True
        t = re.sub(r"\b(?:rejected|false\s+alarms?|false\s+positives?)\b", " ", t, flags=re.I)
    # visual concept = whatever content words remain
    words = [w for w in re.findall(r"[A-Za-z0-9'\-]+", t) if w.lower() not in _STOP]
    cleaned = " ".join(words).strip()
    cleaned = re.sub(r"^(?:possible|events?)\s+", "", cleaned, flags=re.I)
    if cleaned and any(len(w) > 2 for w in words):
        f.visual_query = cleaned[:200]
    return f


# ------------------------------------------------------------------ LLM path
LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": ["string", "null"], "enum": list(CATEGORIES) + [None]},
        "camera_id": {"type": ["string", "null"]},
        "time_from": {"type": ["string", "null"]},
        "time_to": {"type": ["string", "null"]},
        "visual_query": {"type": ["string", "null"]},
        "limit": {"type": "integer"},
    },
    "required": ["category", "camera_id", "time_from", "time_to", "visual_query", "limit"],
}


def validate_filter_dict(d: Any, tz: ZoneInfo, known_cameras: Sequence[str] = ()) -> StructuredFilter:
    """Strict validation of an LLM-proposed filter. Raises ValueError."""
    if not isinstance(d, dict):
        raise ValueError("filter is not an object")
    f = StructuredFilter(source="llm")
    cat = d.get("category")
    if cat is not None:
        if not isinstance(cat, str):
            raise ValueError("category must be string or null")
        c = re.sub(r"[\s_\-]+", " ", cat.strip().lower())
        if c in ("", "null", "none"):
            c = None
        elif c not in CATEGORIES:
            c = category_from_text(c)
            if c is None:
                raise ValueError(f"unknown category {cat!r}")
        f.category = c
    cam = d.get("camera_id")
    if cam not in (None, "", "null"):
        if not isinstance(cam, (str, int)) or not re.fullmatch(r"[A-Za-z0-9_\-]{1,32}", str(cam)):
            raise ValueError(f"invalid camera id {cam!r}")
        s = str(cam)
        f.camera_id = _resolve_camera(int(s), known_cameras) if s.isdigit() else s
    for k in ("time_from", "time_to"):
        v = d.get(k)
        if v in (None, "", "null"):
            continue
        if not isinstance(v, str):
            raise ValueError(f"{k} must be an ISO string")
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        if not 2000 <= dt.year <= 2100:
            raise ValueError(f"{k} out of range")
        setattr(f, k, dt.timestamp())
    if f.time_from is not None and f.time_to is not None and f.time_from > f.time_to:
        raise ValueError("time_from is after time_to")
    vq = d.get("visual_query")
    if vq not in (None, "", "null"):
        if not isinstance(vq, str):
            raise ValueError("visual_query must be string or null")
        vq = re.sub(r"[\x00-\x1f]+", " ", vq).strip()
        f.visual_query = vq[:200] or None
    lim = d.get("limit", 20)
    if isinstance(lim, bool) or not isinstance(lim, (int, float)):
        raise ValueError("limit must be a number")
    f.limit = max(1, min(100, int(lim)))
    return f


class LLMQueryParser:
    """Text-only call to the *already selected* Ollama model (no second VLM)."""

    def __init__(self, client: Any, model: str, timeout_s: float = 6.0, keep_alive: str = "10m"):
        self.client, self.model, self.timeout_s, self.keep_alive = client, model, timeout_s, keep_alive

    def propose(self, text: str, now: datetime, tz_name: str) -> Any:
        sys_p = ("Convert the user's request about security-camera events into a JSON filter. "
                 f"Current local datetime: {now.isoformat(timespec='seconds')} (timezone {tz_name}). "
                 "Resolve words like today/yesterday/last hour to explicit ISO-8601 timestamps with offset. "
                 f"category must be one of {list(CATEGORIES)} or null. camera_id like 'cam2' or null. "
                 "visual_query = only a visual description to search for in images (e.g. 'person wearing a red shirt'), "
                 "else null. limit default 20. Never invent a camera, time or category that the request does not imply. "
                 "Answer with JSON only.")
        resp = self.client.chat(self.model, [{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                                schema=LLM_SCHEMA, options={"temperature": 0, "num_predict": 150, "num_ctx": 1024},
                                keep_alive=self.keep_alive, timeout=self.timeout_s)
        return json.loads((resp.get("message") or {}).get("content", ""))


_CAM_WORDS = re.compile(r"\b(?:camera|cam|cctv)\b|\bcam\d+\b", re.I)
_TIME_WORDS = re.compile(r"today|yesterday|hour|minute|\bday|week|morning|afternoon|evening|night|\b[ap]\.?m\b|\d:\d|between|since|after|before|\bago\b|\d{4}-\d{2}-\d{2}|noon|midnight", re.I)


def reconcile(llm: StructuredFilter, det: StructuredFilter, text: str) -> StructuredFilter:
    """Trust the deterministic extraction where the text is explicit; drop LLM
    values the request does not support (small models hallucinate cameras/times)."""
    out = StructuredFilter(**{**asdict(llm), "notes": list(llm.notes)})
    if det.camera_id:
        if llm.camera_id != det.camera_id:
            out.notes.append(f"camera taken from the text ({det.camera_id}) instead of the model's {llm.camera_id}")
        out.camera_id = det.camera_id
    elif llm.camera_id and not _CAM_WORDS.search(text):
        out.notes.append("ignored a camera the request never mentioned")
        out.camera_id = None
    if det.time_from is not None:
        if (llm.time_from, llm.time_to) != (det.time_from, det.time_to):
            out.notes.append("time window resolved deterministically from the text")
        out.time_from, out.time_to = det.time_from, det.time_to
        out.notes += [n for n in det.notes if n not in out.notes]
    elif (llm.time_from or llm.time_to) and not _TIME_WORDS.search(text):
        out.notes.append("ignored a time window the request never mentioned")
        out.time_from = out.time_to = None
    if det.category:
        out.category = det.category
    out.include_rejected = det.include_rejected
    return out


def parse_query(text: str, now: datetime, tz_name: str, known_cameras: Sequence[str] = (),
                llm: Optional[LLMQueryParser] = None) -> StructuredFilter:
    tz = ZoneInfo(tz_name)
    now = now.astimezone(tz)
    det = parse_query_fallback(text, now, known_cameras)
    if llm is None or not (text or "").strip():
        return det
    try:
        proposed = validate_filter_dict(llm.propose(text, now, tz_name), tz, known_cameras)
        return reconcile(proposed, det, text)
    except Exception as e:  # noqa: BLE001 - timeout, connection, bad JSON, bad values
        det.notes.append(f"LLM parser unavailable ({type(e).__name__}); used the offline deterministic parser")
        log.info("LLM query parse failed: %s", e)
        return det


# ------------------------------------------------------------------- search
@dataclass
class SearchResult:
    event: EventRecord
    rank: int
    image_score: Optional[float] = None  # raw cosine similarity (not a probability)
    summary_score: Optional[float] = None
    hybrid_score: Optional[float] = None  # weighted *normalised* similarity (not a probability)
    lexical_score: Optional[float] = None


@dataclass
class SearchResponse:
    filter: StructuredFilter
    results: List[SearchResult]
    mode: str
    n_candidates: int
    latency_ms: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _fixed(d: Dict[str, float], rng: Sequence[float]) -> Dict[str, float]:
    lo, hi = rng
    return {k: max(0.0, min(1.0, (v - lo) / (hi - lo))) for k, v in d.items()}


class SearchEngine:
    def __init__(self, store: EventStore, index: Any, embedder: Any, retrieval_cfg: Dict[str, Any],
                 tz_name: str = "UTC", llm_parser: Optional[LLMQueryParser] = None,
                 known_cameras: Sequence[str] = ()):
        self.store, self.index, self.embedder = store, index, embedder
        self.cfg = retrieval_cfg
        self.tz_name = tz_name
        self.llm = llm_parser
        self.known_cameras = list(known_cameras)

    def parse(self, text: str, now: Optional[datetime] = None) -> StructuredFilter:
        now = now or datetime.now(ZoneInfo(self.tz_name))
        return parse_query(text, now, self.tz_name, self.known_cameras, self.llm)

    def ask(self, text: str, mode: Optional[str] = None, now: Optional[datetime] = None) -> SearchResponse:
        t0 = time.perf_counter()
        flt = self.parse(text, now)
        resp = self.search(flt, mode)
        resp.latency_ms["parse"] = round((time.perf_counter() - t0) * 1000 - resp.latency_ms.get("total", 0), 2)
        return resp

    def candidates(self, flt: StructuredFilter) -> List[EventRecord]:
        verdicts = ("verified", "unverified", "rejected") if flt.include_rejected else ("verified", "unverified")
        return self.store.query(camera_id=flt.camera_id, category=flt.category, verdicts=verdicts,
                                time_from=flt.time_from, time_to=flt.time_to, min_confidence=flt.min_confidence,
                                include_scene_index=bool(flt.visual_query and not flt.category),
                                limit=int(self.cfg.get("max_candidates", 2000)))

    def search(self, flt: StructuredFilter, mode: Optional[str] = None) -> SearchResponse:
        mode = mode or self.cfg["mode"]
        warnings: List[str] = []
        lat: Dict[str, float] = {}
        t0 = time.perf_counter()
        cands = self.candidates(flt)
        lat["sql"] = round((time.perf_counter() - t0) * 1000, 2)
        if not flt.visual_query:  # purely structured request: deterministic, newest first
            res = [SearchResult(e, i + 1) for i, e in enumerate(cands[:flt.limit])]
            lat["total"] = round((time.perf_counter() - t0) * 1000, 2)
            return SearchResponse(flt, res, "structured", len(cands), lat, warnings)
        ids = [e.id for e in cands]
        img_raw: Dict[str, float] = {}
        txt_raw: Dict[str, float] = {}
        t1 = time.perf_counter()
        semantic_ok = True
        try:
            qv = self.embedder.embed_texts([flt.visual_query])[0]
            lat["embed_query"] = round((time.perf_counter() - t1) * 1000, 2)
            t2 = time.perf_counter()
            if mode in ("image", "hybrid"):
                img_raw = self.index.scores("image", qv, ids)
            if mode in ("summary", "hybrid"):
                txt_raw = self.index.scores("summary", qv, ids)
            lat["vector"] = round((time.perf_counter() - t2) * 1000, 2)
            if not img_raw and not txt_raw and ids:
                warnings.append("no embeddings found for the candidate events (index empty or being rebuilt)")
        except Exception as e:  # noqa: BLE001 - CLIP/Chroma unavailable => lexical fallback
            semantic_ok = False
            warnings.append(f"semantic search unavailable ({type(e).__name__}: {e}); used keyword matching on summaries")
        t3 = time.perf_counter()
        results: List[SearchResult] = []
        if semantic_ok and (img_raw or txt_raw):
            if self.cfg.get("normalization", "minmax") == "fixed" or len(ids) < 2:
                rng = self.cfg["fixed_ranges"]
                ni, nt = _fixed(img_raw, rng["image"]), _fixed(txt_raw, rng["summary"])
            else:
                ni, nt = _minmax(img_raw), _minmax(txt_raw)
            wi, wt = float(self.cfg["image_weight"]), float(self.cfg["summary_weight"])
            for e in cands:
                if mode == "image":
                    h = ni.get(e.id)
                elif mode == "summary":
                    h = nt.get(e.id)
                else:
                    h = wi * ni.get(e.id, 0.0) + wt * nt.get(e.id, 0.0) if (e.id in ni or e.id in nt) else None
                results.append(SearchResult(e, 0, img_raw.get(e.id), txt_raw.get(e.id), h))
            results.sort(key=lambda r: (r.hybrid_score is None, -(r.hybrid_score or 0.0), -r.event.timestamp))
        else:
            toks = {w for w in re.findall(r"[a-z0-9]+", flt.visual_query.lower()) if w not in _STOP}
            for e in cands:
                words = set(re.findall(r"[a-z0-9]+", (e.summary + " " + e.category).lower()))
                sc = (len(toks & words) / len(toks)) if toks else 0.0
                results.append(SearchResult(e, 0, lexical_score=sc))
            results.sort(key=lambda r: (-(r.lexical_score or 0.0), -r.event.timestamp))
        results = results[:flt.limit]
        for i, r in enumerate(results):
            r.rank = i + 1
        lat["rank"] = round((time.perf_counter() - t3) * 1000, 2)
        lat["total"] = round((time.perf_counter() - t0) * 1000, 2)
        return SearchResponse(flt, results, mode if semantic_ok else "keyword", len(cands), lat, warnings)
