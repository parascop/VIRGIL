"""VLM client/verifier against a mock Ollama HTTP server (TEST FIXTURE, not a real model)."""
import json
import socket
import time

import numpy as np
import pytest

from helpers import MockOllama
from vigil.types import Trigger, VLMRequest, VLMResult
from vigil.vlm import (OllamaClient, VLMVerifier, apply_verdict, normalize_category, validate_vlm_output)

GOOD = {"category": "person down", "vlm_confidence": 0.82, "summary": "A person lies on the floor.",
        "objects": ["person", "floor"], "reason": "Seen lying across tiles 3-4."}


def cfg(url, **kw):
    c = {"base_url": url, "model": "moondream", "num_ctx": 2048, "temperature": 0, "num_predict": 120,
         "keep_alive": "5m", "num_gpu": None, "timeout_s": 3, "connect_timeout_s": 1, "retries": 1,
         "unavailable_cooldown_s": 5, "allow_reclassify": True}
    c.update(kw)
    return c


def req(category="person down"):
    tg = Trigger(trigger_type="person_down", category=category, camera_id="cam1", t=1.0, track_ids=(1,),
                 heuristic_score=0.7, reason="aspect ratio flipped")
    img = np.zeros((64, 64, 3), np.uint8)
    return VLMRequest(event_id="ev1", trigger=tg, composite=img, trigger_frame=img, sample_count=4, t_trigger=time.time())


W = {"heuristic_weight": 0.4, "vlm_weight": 0.6}


def test_validate_accepts_good_and_normalises():
    d, err, trunc = validate_vlm_output(json.dumps({**GOOD, "category": "Person_Down"}))
    assert d["category"] == "person down" and not err and not trunc
    d, _, trunc = validate_vlm_output(json.dumps({**GOOD, "summary": " ".join(["word"] * 40)}))
    assert len(d["summary"].split()) == 25 and trunc


@pytest.mark.parametrize("bad", [
    "", "not json", "[1,2]", json.dumps({**GOOD, "category": "robbery"}),
    json.dumps({**GOOD, "vlm_confidence": 1.5}), json.dumps({**GOOD, "vlm_confidence": -0.1}),
    json.dumps({**GOOD, "vlm_confidence": "high"}), '{"category":"none","vlm_confidence":NaN,"summary":"","objects":[],"reason":""}',
    '{"category":"none","vlm_confidence":Infinity,"summary":"","objects":[],"reason":""}',
    json.dumps({**GOOD, "objects": "person"}), json.dumps({**GOOD, "summary": 5}), json.dumps({**GOOD, "vlm_confidence": True})])
def test_validate_rejects_bad(bad):
    d, err, _ = validate_vlm_output(bad)
    assert d is None and err


def test_fenced_json_tolerated():
    d, _, _ = validate_vlm_output("```json\n" + json.dumps(GOOD) + "\n```")
    assert d and d["category"] == "person down"


def test_service_checks_and_schema_request():
    with MockOllama([GOOD]) as m:
        v = VLMVerifier(cfg(m.url))
        info = v.check()
        assert info["reachable"] and info["model_available"] and info["schema_supported"]
        res = v.verify(req())
        assert res.ok and res.category == "person down" and res.attempts == 1 and res.t_end >= res.t_start
        body = m.requests[0]
        assert body["options"]["temperature"] == 0 and body["options"]["num_ctx"] == 2048
        assert body["options"]["num_predict"] == 120 and body["format"]["type"] == "object"
        assert body["messages"][1]["images"] and "chronological" in body["messages"][0]["content"].lower()
        assert "aspect ratio flipped" in body["messages"][1]["content"]


def test_missing_model_message():
    with MockOllama(models=["other:latest"]) as m:
        info = VLMVerifier(cfg(m.url)).check()
        assert info["reachable"] and not info["model_available"] and "ollama pull moondream" in info["error"]


def test_invalid_output_retried_exactly_once():
    with MockOllama(["garbage", "still garbage", GOOD]) as m:
        res = VLMVerifier(cfg(m.url)).verify(req())
        assert not res.ok and res.attempts == 2 and len(m.requests) == 2 and "invalid structured output" in res.error
        assert "garbage" in res.raw


def test_invalid_then_valid_succeeds_on_retry():
    with MockOllama(["oops", GOOD]) as m:
        res = VLMVerifier(cfg(m.url)).verify(req())
        assert res.ok and res.attempts == 2


def test_timeout_does_not_raise():
    with MockOllama([GOOD], delay=2.0) as m:
        t0 = time.time()
        res = VLMVerifier(cfg(m.url, timeout_s=0.5)).verify(req())
        assert not res.ok and "timeout" in res.error and time.time() - t0 < 3


def test_connection_refused_then_backoff():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()  # closed port
    v = VLMVerifier(cfg(f"http://127.0.0.1:{port}"))
    r1 = v.verify(req())
    assert not r1.ok and "connect" in r1.error
    t0 = time.time()
    r2 = v.verify(req())  # in back-off: immediate, no network attempt
    assert not r2.ok and "back-off" in r2.error and time.time() - t0 < 0.2
    info = v.check()
    assert not info["reachable"] and "ollama pull" in info["error"]


def test_http_error_reported_as_unverified_reason():
    with MockOllama() as m:
        m.post_status = 500
        res = VLMVerifier(cfg(m.url)).verify(req())
        assert not res.ok and "http" in res.error.lower() and "crashed" in res.error


# ---------------------------------------------------------------- verdicts
def test_verdict_verified_rejected_unverified():
    ok = VLMResult(ok=True, category="person down", vlm_confidence=0.8, summary="s", model="m")
    f = apply_verdict(req(), ok, cfg("http://x"), W)
    assert f["verdict"] == "verified" and f["category"] == "person down"
    assert f["confidence_score"] == pytest.approx(0.4 * 0.7 + 0.6 * 0.8)
    none = VLMResult(ok=True, category="none", vlm_confidence=0.9, summary="nothing", model="m")
    f = apply_verdict(req(), none, cfg("http://x"), W)
    assert f["verdict"] == "rejected" and f["category"] == "none"
    assert f["confidence_score"] == pytest.approx(0.4 * 0.7 + 0.6 * 0.1)
    f = apply_verdict(req(), VLMResult(ok=False, error="timeout"), cfg("http://x"), W)
    assert f["verdict"] == "unverified" and f["category"] == "person down" and f["vlm_confidence"] == 0.0
    assert "timeout" in f["error"]
    f = apply_verdict(req(), None, cfg("http://x"), W, "queue saturated")
    assert f["verdict"] == "unverified" and f["error"] == "queue saturated"


def test_reclassification_policy():
    other = VLMResult(ok=True, category="suspicious activity", vlm_confidence=0.7, summary="s", model="m")
    assert apply_verdict(req(), other, cfg("http://x"), W)["category"] == "suspicious activity"
    f = apply_verdict(req(), other, cfg("http://x", allow_reclassify=False), W)
    assert f["verdict"] == "rejected"


def test_normalize_category():
    assert normalize_category("Crowd-Incident") == "crowd incident" and normalize_category("x") is None
