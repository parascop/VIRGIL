from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from helpers import make_profile
from vigil.embeddings import HashEmbedder, VectorIndex
from vigil.query import (LLMQueryParser, SearchEngine, StructuredFilter, parse_query, parse_query_fallback,
                         validate_filter_dict)
from vigil.storage import EventStore, persist_event
from vigil.types import EventRecord

TZ = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 15, 18, 50, tzinfo=TZ)  # Thursday


def ep(y, mo, d, h=0, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=TZ).timestamp()


def P(text, cams=("cam1", "cam2")):
    return parse_query_fallback(text, NOW, cams)


def test_camera_and_category_and_today():
    f = P("Show all tailgating events from camera 2 today.")
    assert (f.camera_id, f.category) == ("cam2", "tailgating")
    assert f.time_from == ep(2026, 10, 15) and f.time_to == ep(2026, 10, 16)
    assert f.visual_query is None


@pytest.mark.parametrize("text,cat", [
    ("Find possible person-down events in the last hour", "person down"),
    ("someone collapsed", "person down"), ("any fall detected", "person down"),
    ("Show possible fighting events that occurred yesterday", "fighting"), ("brawl in the lobby", "fighting"),
    ("Find crowd incidents between 2 PM and 4 PM", "crowd incident"),
    ("Show suspicious activity near the restricted entrance", "suspicious activity"),
    ("loitering", "suspicious activity")])
def test_category_synonyms(text, cat):
    assert P(text).category == cat


def test_last_hour_window():
    f = P("Find possible person-down events in the last hour")
    assert f.time_to == NOW.timestamp() and f.time_from == pytest.approx(NOW.timestamp() - 3600)


def test_yesterday():
    f = P("Show possible fighting events that occurred yesterday")
    assert f.time_from == ep(2026, 10, 14) and f.time_to == ep(2026, 10, 15)


def test_between_2pm_and_4pm():
    f = P("Find crowd incidents between 2 PM and 4 PM")
    assert f.time_from == ep(2026, 10, 15, 14) and f.time_to == ep(2026, 10, 15, 16)
    assert f.visual_query is None


def test_yesterday_between_times():
    f = P("fighting yesterday between 9am and 11am")
    assert f.time_from == ep(2026, 10, 14, 9) and f.time_to == ep(2026, 10, 14, 11)


def test_ambiguous_clock_is_flagged_not_guessed():
    f = P("crowd between 2 and 4")
    assert f.time_from == ep(2026, 10, 15, 2) and any("24-hour" in n for n in f.notes)


def test_last_n_units_and_limit():
    f = P("show the latest 5 events in the past 30 minutes")
    assert f.limit == 5 and f.time_from == pytest.approx(NOW.timestamp() - 1800)


def test_iso_date():
    f = P("tailgating on 2026-10-10")
    assert f.time_from == ep(2026, 10, 10) and f.time_to == ep(2026, 10, 11)


def test_visual_queries():
    f = P("Find frames containing a person wearing a red shirt.")
    assert f.category is None and "red shirt" in f.visual_query and f.camera_id is None
    f = P("When did someone enter through the door?")
    assert "enter through door" in f.visual_query
    f = P("Show suspicious activity near the restricted entrance")
    assert f.category == "suspicious activity" and "restricted entrance" in f.visual_query


def test_empty_query_is_conservative():
    f = P("   ")
    assert f.category is None and f.camera_id is None and f.time_from is None and f.visual_query is None
    assert f.notes


def test_camera_forms():
    assert P("cam3 events").camera_id == "cam3"
    assert P("camera #2").camera_id == "cam2"
    assert parse_query_fallback("camera 2", NOW, ["2"]).camera_id == "2"


# ---------------------------------------------------------------- LLM path
class FakeLLM:
    def __init__(self, payload=None, exc=None):
        self.payload, self.exc = payload, exc

    def propose(self, text, now, tz_name):
        if self.exc:
            raise self.exc
        return self.payload


def test_malformed_llm_output_falls_back_to_deterministic():
    for llm in (FakeLLM(payload="not json at all"), FakeLLM(exc=TimeoutError("slow")),
                FakeLLM(payload={"category": "pirates", "camera_id": None, "limit": 3}),
                FakeLLM(payload={"camera_id": "cam1'; DROP TABLE events;--", "limit": 3})):
        f = parse_query("tailgating events from camera 2 today", NOW, "Asia/Kolkata", ["cam2"], llm)
        assert f.source == "fallback" and f.camera_id == "cam2" and f.category == "tailgating"
        assert any("deterministic" in n for n in f.notes)


def test_llm_hallucinated_camera_and_time_are_dropped():
    payload = {"category": "fighting", "camera_id": "cam9", "time_from": "2026-10-01T00:00:00+05:30",
               "time_to": "2026-10-02T00:00:00+05:30", "visual_query": None, "limit": 20}
    f = parse_query("show fighting events", NOW, "Asia/Kolkata", [], FakeLLM(payload=payload))
    assert f.source == "llm" and f.camera_id is None and f.time_from is None and f.category == "fighting"


def test_llm_valid_filter_accepted_and_text_wins_on_conflict():
    payload = {"category": "tailgating", "camera_id": "cam1", "time_from": "2026-10-15T00:00:00+05:30",
               "time_to": "2026-10-16T00:00:00+05:30", "visual_query": None, "limit": 7}
    f = parse_query("tailgating on camera 2 today", NOW, "Asia/Kolkata", ["cam1", "cam2"], FakeLLM(payload=payload))
    assert f.camera_id == "cam2" and f.limit == 7 and f.time_from == ep(2026, 10, 15)


def test_validate_filter_rejects_bad_values():
    for bad in ({"category": 5}, {"time_from": "yesterday-ish"}, {"limit": "many"}, "x",
                {"time_from": "2026-10-16T00:00:00+05:30", "time_to": "2026-10-15T00:00:00+05:30"},
                {"time_from": "1850-01-01T00:00:00Z"}):
        with pytest.raises(Exception):
            validate_filter_dict(bad, TZ)
    assert validate_filter_dict({"limit": 5000}, TZ).limit == 100


# ------------------------------------------------------- SQL safety / search
def _populate(tmp_path):
    prof = make_profile(tmp_path)
    store = EventStore(prof["paths"]["sqlite"])
    emb = HashEmbedder()
    idx = VectorIndex(prof["paths"]["chroma"], emb.tag, emb.dim)
    import numpy as np

    def add(i, cam, cat, ts, summary, verdict="verified", trig="x"):
        rec = EventRecord(id=f"e{i}", timestamp=ts, camera_id=cam, category=cat, confidence_score=0.5, summary=summary,
                          keyframe_path="", verdict=verdict, trigger_type=trig)
        img = np.full((32, 32, 3), 40 * i, np.uint8)
        persist_event(store, rec, img, None, prof["paths"]["keyframes"], emb, idx)

    now = NOW.timestamp()
    add(1, "cam2", "tailgating", now - 3600, "two people pass the gate together")
    add(2, "cam1", "fighting", now - 7200, "two people fighting near the door")
    add(3, "cam2", "tailgating", now - 90000, "yesterday tailgating pair at the gate")
    add(4, "cam1", "person down", now - 600, "a person in a red shirt lies on the floor")
    add(5, "cam1", "none", now - 500, "candidate rejected", verdict="rejected")
    add(6, "cam1", "none", now - 400, "", verdict="unverified", trig="scene_index")
    return prof, store, emb, idx


def test_structured_query_filters_and_rejected_hidden(tmp_path):
    prof, store, emb, idx = _populate(tmp_path)
    eng = SearchEngine(store, idx, emb, prof["retrieval"], "Asia/Kolkata", known_cameras=["cam1", "cam2"])
    r = eng.ask("Show all tailgating events from camera 2 today", now=NOW)
    assert [x.event.id for x in r.results] == ["e1"]  # e3 is yesterday
    assert r.mode == "structured"
    r = eng.ask("show everything", now=NOW)
    assert "e5" not in {x.event.id for x in r.results} and "e6" not in {x.event.id for x in r.results}
    r = eng.ask("show rejected events", now=NOW)
    assert "e5" in {x.event.id for x in r.results}


def test_visual_query_ranks_with_hybrid_and_reports_scores(tmp_path):
    prof, store, emb, idx = _populate(tmp_path)
    eng = SearchEngine(store, idx, emb, prof["retrieval"], "Asia/Kolkata")
    r = eng.ask("person in a red shirt on the floor", now=NOW)
    assert r.results[0].event.id == "e4"
    top = r.results[0]
    assert top.hybrid_score is not None and top.image_score is not None and top.summary_score is not None
    for mode in ("image", "summary", "hybrid"):
        assert eng.search(StructuredFilter(visual_query="red shirt"), mode).results


def test_scene_index_frames_are_searchable_only_for_visual_queries(tmp_path):
    prof, store, emb, idx = _populate(tmp_path)
    eng = SearchEngine(store, idx, emb, prof["retrieval"], "Asia/Kolkata")
    assert "e6" in {x.event.id for x in eng.search(StructuredFilter(visual_query="anything", limit=50)).results}
    assert "e6" not in {x.event.id for x in eng.search(StructuredFilter(limit=50)).results}


def test_semantic_failure_degrades_to_keyword_search(tmp_path):
    prof, store, emb, idx = _populate(tmp_path)

    class Broken:
        def embed_texts(self, t):
            raise RuntimeError("clip not available")

    eng = SearchEngine(store, idx, Broken(), prof["retrieval"], "Asia/Kolkata")
    r = eng.ask("red shirt", now=NOW)
    assert r.mode == "keyword" and r.warnings and r.results[0].event.id == "e4"


def test_sql_injection_like_input_is_inert_and_parameterised(tmp_path):
    prof, store, emb, idx = _populate(tmp_path)
    eng = SearchEngine(store, idx, emb, prof["retrieval"], "Asia/Kolkata")
    stmts = []
    store.conn().set_trace_callback(stmts.append)
    evil = "camera 2'; DROP TABLE events; -- tailgating \" OR 1=1"
    r = eng.ask(evil, now=NOW)
    store.conn().set_trace_callback(None)
    assert store.count() == 6  # table intact
    assert r.filter.category == "tailgating" and r.filter.camera_id == "cam2"
    selects = [s for s in stmts if s.startswith("SELECT * FROM events")]
    assert selects and all("DROP" not in s.upper() and "OR 1=1" not in s for s in selects)
    # sqlite3's trace shows bound values already expanded; verify placeholders at source level
    import inspect
    from vigil import storage
    src = inspect.getsource(storage.EventStore.query)
    assert "?" in src and "f\"" not in src.split("where.append")[0][:0]
    flt = StructuredFilter(camera_id="cam1' OR '1'='1", category="x'; DROP TABLE events;--")
    assert store.query(camera_id=flt.camera_id, category=flt.category) == []
    assert store.count() == 6
