"""Dashboard smoke test with Streamlit's AppTest: the app must render without exceptions
or error boxes (this catches broken dependency combinations such as pyarrow vs numpy,
which only show up when st.table / st.dataframe render)."""
import json
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from vigil.storage import EventStore  # noqa: E402
from vigil.types import EventRecord  # noqa: E402


def _seed(tmp: Path) -> Path:
    paths = {k: str(tmp / k) for k in ("chroma", "keyframes", "live", "clips", "jobs", "cache")}
    paths["sqlite"] = str(tmp / "vigil.db")
    prof = tmp / "profile.yaml"
    prof.write_text(yaml.safe_dump({"name": "pytest", "paths": paths, "vlm": {"enabled": False}}), encoding="utf-8")
    store = EventStore(paths["sqlite"])
    now = time.time()
    for i, (cat, verdict) in enumerate([("person down", "verified"), ("fighting", "rejected"), ("tailgating", "unverified")]):
        store.insert_event(EventRecord(id=f"ev_test{i}", timestamp=now - i * 60, camera_id="cam1", category=cat,
                                       confidence_score=0.6, summary=f"test event {i}", keyframe_path="",
                                       verdict=verdict, trigger_type="person_down", t_stored=now))
    store.close()
    live = Path(paths["live"])
    live.mkdir(parents=True, exist_ok=True)
    (live / "status.json").write_text(json.dumps({
        "updated_at": now, "health": "ok", "profile": "pytest", "vlm_model": "none", "vlm_state": "disabled",
        "vlm_info": {}, "embedder_ok": True, "detector_device": "cpu", "cameras": {}, "vlm_queue_depth": 0,
        "dropped_triggers": 0, "event_counts": {}, "verdict_counts": {}, "metrics": {"stages": {}, "counters": {}, "gauges": {}},
        "uptime_s": 1.0}), encoding="utf-8")
    return prof


def test_dashboard_renders_without_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("VIGIL_PROFILE", str(_seed(tmp_path)))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    monkeypatch.chdir(ROOT)
    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert len(at.tabs) >= 4
    assert len(at.table) >= 1  # the per-category table is the element that needs pyarrow
