"""Unit tests for the pure scoring logic of eval/run_eval.py (no pipeline run)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
ev = importlib.util.module_from_spec(spec)
sys.modules["run_eval"] = ev
spec.loader.exec_module(ev)

GT, Pred = ev.GT, ev.Pred


def test_true_positive_and_miss():
    gts = [GT("a.mp4", "person down", 4, 12), GT("b.mp4", "fighting", 1, 10)]
    preds = [Pred("a.mp4", 6.5, "person down", 0.8)]
    r = ev.evaluate_variant(gts, preds, tol=2.0)
    assert r["per_category"]["person down"]["tp"] == 1
    assert r["per_category"]["fighting"]["fn"] == 1
    assert r["confusion"]["fighting"]["none"] == 1
    assert r["micro"]["precision"] == 1.0 and r["micro"]["recall"] == 0.5


def test_tolerance_boundaries():
    gts = [GT("a.mp4", "tailgating", 2.0, 3.0)]
    assert ev.evaluate_variant(gts, [Pred("a.mp4", 5.0, "tailgating")], 2.0)["micro"]["tp"] == 1
    assert ev.evaluate_variant(gts, [Pred("a.mp4", 5.01, "tailgating")], 2.0)["micro"]["tp"] == 0


def test_duplicates_not_false_alarms_and_no_double_credit():
    gts = [GT("a.mp4", "crowd incident", 2, 12)]
    preds = [Pred("a.mp4", 3, "crowd incident"), Pred("a.mp4", 5, "crowd incident"), Pred("a.mp4", 9, "crowd incident")]
    r = ev.evaluate_variant(gts, preds, 1.0)
    assert r["micro"]["tp"] == 1 and r["micro"]["fp"] == 0 and r["duplicates"] == 2


def test_wrong_category_is_fp_and_confusion_and_miss():
    gts = [GT("a.mp4", "fighting", 1, 10)]
    r = ev.evaluate_variant(gts, [Pred("a.mp4", 4, "suspicious activity", 0.6)], 1.0)
    assert r["per_category"]["fighting"]["fn"] == 1
    assert r["per_category"]["suspicious activity"]["fp"] == 1
    assert r["confusion"]["fighting"]["suspicious activity"] == 1
    assert r["confusion"]["none"]["suspicious activity"] == 0  # counted once, in the GT row


def test_event_in_other_clip_does_not_match():
    gts = [GT("a.mp4", "person down", 4, 12)]
    r = ev.evaluate_variant(gts, [Pred("b.mp4", 6, "person down")], 2.0)
    assert r["micro"]["tp"] == 0 and r["micro"]["fp"] == 1 and r["micro"]["fn"] == 1
    assert r["confusion"]["none"]["person down"] == 1


def test_negative_clip_false_alarm_rate():
    assert ev.fp_per_hour(3, 0.5) == 6.0
    assert ev.fp_per_hour(3, 0.0) is None


def test_reliability_buckets_count_precision():
    gts = [GT("a.mp4", "person down", 0, 10)]
    preds = [Pred("a.mp4", 5, "person down", 0.9), Pred("z.mp4", 5, "person down", 0.3)]
    r = ev.evaluate_variant(gts, preds, 1.0)
    top = [b for b in r["reliability"] if b["range"][0] == 0.85][0]
    low = [b for b in r["reliability"] if b["range"][0] == 0.0][0]
    assert top["n"] == 1 and top["precision"] == 1.0
    assert low["n"] == 1 and low["precision"] == 0.0


def _row(i, trig, cat, verdict, heur=0.5, conf=0.7, t=3.0, src="/x/a.mp4"):
    return {"id": i, "trigger_type": trig, "category": cat, "verdict": verdict, "heuristic_score": heur,
            "confidence_score": conf, "media_time_s": t, "source_path": src}


def test_split_variants_rejected_dropped_and_vlm_flag():
    cat_map = {"person_down": "person down", "fighting": "fighting"}
    rows = [_row("1", "person_down", "person down", "verified"),
            _row("2", "fighting", "fighting", "rejected"),
            _row("3", "fighting", "suspicious activity", "verified"),
            _row("4", "scene_index", "suspicious activity", "unverified")]
    heur, vlm, executed = ev.split_variants(rows, cat_map)
    assert [p.event_id for p in heur] == ["1", "2", "3"]
    assert [p.event_id for p in vlm] == ["1", "3"]
    assert [p.category for p in heur] == ["person down", "fighting", "fighting"]  # heuristic category
    assert vlm[1].category == "suspicious activity"  # VLM reclassification honoured
    assert executed is True


def test_vlm_not_executed_when_all_unverified():
    rows = [_row("1", "person_down", "person down", "unverified")]
    _, _, executed = ev.split_variants(rows, {"person_down": "person down"})
    assert executed is False


def test_manifest_is_valid_and_complete():
    rows = ev.load_manifest(ROOT / "eval" / "manifest.csv")
    gts = ev.ground_truth(rows)
    assert {g.category for g in gts} == set(ev.CATEGORIES)  # every category has a positive clip
    assert any(r.category == "none" and r.status == "runnable" for r in rows)  # and negatives
    assert all(r.status == "not_run" for r in rows if r.clip.startswith("external/"))
    assert all(g.end > g.start for g in gts)
