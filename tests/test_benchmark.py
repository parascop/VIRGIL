"""Benchmark script logic against a mock Ollama (no real model involved)."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from helpers import MockOllama

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("benchmark_models", ROOT / "scripts" / "benchmark_models.py")
bm = importlib.util.module_from_spec(spec)
sys.modules["benchmark_models"] = bm
spec.loader.exec_module(bm)

GOOD = {"category": "person down", "vlm_confidence": 0.8, "summary": "s", "objects": [], "reason": "r"}


def row(model, valid, p95, mem=1000, size=1e9, side=512):
    return {"model": model, "status": "ok", "options": {}, "per_size": {str(side): {
        "valid_rate": valid, "p95_s": p95, "p50_s": p95 / 2, "gpu_peak_used_mb": mem, "ollama_loaded_mb": mem, "n": 10,
        "valid": int(valid * 10), "placement": "GPU"}}}


def test_selection_prefers_smaller_model_when_comparable():
    r = [row("big", 1.0, 5, size=4e9), row("small", 0.95, 6, size=1e9)]
    s = bm.select_best(r, {"big": 4e9, "small": 1e9}, 3600)
    assert s["passed"] and s["choice"]["model"] == "small"


def test_selection_valid_rate_is_primary_and_gates():
    r = [row("fast", 0.5, 2, size=1e9), row("good", 1.0, 8, size=3e9)]
    assert bm.select_best(r, {"fast": 1e9, "good": 3e9}, 3600)["choice"]["model"] == "good"


def test_selection_latency_and_memory_gates_and_fallback():
    r = [row("slow", 1.0, 30), row("hungry", 1.0, 5, mem=5000)]
    s = bm.select_best(r, {}, 3600)
    assert not s["passed"] and s["choice"] is not None and s["unmet"]  # best safe fallback + reported unmet targets
    assert bm.select_best([row("junk", 0.0, 5)], {}, 3600)["choice"] is None
    assert bm.select_best([], {}, 3600)["choice"] is None


def test_profile_update_preserves_comments_and_is_idempotent():
    text = (ROOT / "configs" / "profile_rtx2050.yaml").read_text()
    block = {"selected_model": "qwen3-vl:2b", "p95_s": 7.1}
    t1 = bm.update_profile_text(text, "qwen3-vl:2b", 768, 0, block)
    t2 = bm.update_profile_text(t1, "qwen3-vl:2b", 768, 0, block)
    d = yaml.safe_load(t2)
    assert d["vlm"]["model"] == "qwen3-vl:2b" and d["vlm"]["composite_max_side"] == 768 and d["vlm"]["num_gpu"] == 0
    assert d["benchmark"]["selected_model"] == "qwen3-vl:2b" and t2.count("BEGIN BENCHMARK") == 1
    assert "# Windows 10/11" in t2  # comments survive


def test_models_7b_and_up_are_never_benchmarked():
    for tag in ("llava:7b", "qwen2.5vl:7b", "llama3.2-vision:11b", "gemma3:27b"):
        assert bm.TOO_BIG.search(tag)
    for tag in ("qwen3-vl:2b", "gemma3:4b", "moondream", "qwen2.5vl:3b", "granite3.2-vision"):
        assert not bm.TOO_BIG.search(tag)


def test_end_to_end_with_mock_missing_models_skipped(tmp_path, capsys):
    prof = tmp_path / "profile.yaml"
    shutil.copy(ROOT / "configs" / "profile_gtx1650.yaml", prof)
    out = tmp_path / "bench.json"
    with MockOllama([GOOD] * 40, models=["moondream:latest"]) as m:
        rc = bm.main(["--profile", str(prof), "--base-url", m.url, "--models", "moondream", "qwen3-vl:2b", "gemma3:27b",
                      "--frames", "3", "--sizes", "512", "--write-profile", "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert [r["model"] for r in rep["results"]] == ["moondream"]
    assert set(rep["skipped"]) == {"qwen3-vl:2b", "gemma3:27b"}  # unavailable / too large: skipped, no crash
    s = rep["results"][0]["per_size"]["512"]
    assert s["valid_rate"] == 1.0 and s["p50_s"] is not None and s["gpu_peak_used_mb"] is None  # no GPU here -> null, not invented
    d = yaml.safe_load(prof.read_text())
    assert d["vlm"]["model"] == "moondream" and "benchmark" in d


def test_unreachable_ollama_exits_cleanly(tmp_path):
    assert bm.main(["--profile", str(ROOT / "configs" / "profile_gtx1650.yaml"), "--base-url", "http://127.0.0.1:9", "--out", str(tmp_path / "x.json")]) == 2
