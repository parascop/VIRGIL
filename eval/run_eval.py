#!/usr/bin/env python
"""Vigil evaluation harness.

Runs the real pipeline offline (``--speed 0``) once per clip listed in
``eval/manifest.csv`` into ONE shared SQLite/Chroma store, then scores the stored
events against the manifest's ground-truth intervals.

Everything reported comes from the events actually stored by those runs. Sections
that need components that are unavailable (VLM, real CLIP weights) are written as
``{"executed": false, "reason": ...}``; no number is ever invented or estimated.

Variants (both derived from the *same* stored events, so the ablation is paired):
  heuristics_only  every candidate counts as a detection, category = heuristic category
  vlm_assisted     rejected candidates are dropped; verified ones use the VLM category;
                   unverified ones (VLM unavailable) keep the heuristic category

Matching (see ``evaluate_variant``): a prediction matches a ground-truth interval
of the same clip when the categories are equal and its media time lies within
``[start - tol, end + tol]``. One prediction earns credit for at most one interval and
extra same-category predictions inside an already-credited interval are reported as
``duplicates`` (not false alarms).

Usage:
  python eval/run_eval.py [--profile configs/profile_x.yaml] [--vlm auto|off|on]
                          [--tolerance 2.0] [--skip-run] [--out eval/results/eval.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vigil.types import CATEGORIES  # noqa: E402
from vigil.util import percentile  # noqa: E402

NONE = "none"
LABELS: Tuple[str, ...] = tuple(CATEGORIES) + (NONE,)
RELIABILITY_EDGES = (0.0, 0.5, 0.7, 0.85, 1.0001)


# ----------------------------------------------------------------- data
@dataclass
class GT:
    clip: str  # basename
    category: str
    start: float
    end: float


@dataclass
class Pred:
    clip: str
    t: float  # media time (s)
    category: str
    conf: float = 0.0
    verdict: str = "unverified"
    event_id: str = ""


@dataclass
class ManifestRow:
    clip: str
    camera_id: str
    category: str
    start: Optional[float]
    end: Optional[float]
    status: str
    provenance: str
    notes: str = ""


def _f(v: str) -> Optional[float]:
    v = (v or "").strip()
    return float(v) if v else None


def load_manifest(path: Path) -> List[ManifestRow]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [ManifestRow(r["clip"].strip(), r["camera_id"].strip(), r["category"].strip(), _f(r["start_s"]),
                            _f(r["end_s"]), r["status"].strip(), r["provenance"].strip(), r.get("notes", ""))
                for r in csv.DictReader(fh)]


def ground_truth(rows: Iterable[ManifestRow]) -> List[GT]:
    out = []
    for r in rows:
        if r.status == "runnable" and r.category != NONE:
            if r.start is None or r.end is None:
                raise ValueError(f"{r.clip}: category {r.category!r} needs start_s and end_s")
            if r.category not in CATEGORIES:
                raise ValueError(f"{r.clip}: unknown category {r.category!r}")
            out.append(GT(Path(r.clip).name, r.category, r.start, r.end))
    return out


# -------------------------------------------------------------- scoring (pure)
def _prf(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_variant(gts: Sequence[GT], preds: Sequence[Pred], tol: float) -> Dict[str, Any]:
    """Score one set of predictions. Deterministic and side-effect free."""
    preds = sorted(preds, key=lambda p: (p.clip, p.t))
    used: set = set()  # indices of preds that earned TP credit or were duplicates
    status: Dict[int, str] = {}  # pred index -> tp | dup | fp
    gt_matched: List[Optional[int]] = []
    for g in gts:
        hit = None
        for i, p in enumerate(preds):
            if i in used or p.clip != g.clip or p.category != g.category:
                continue
            if g.start - tol <= p.t <= g.end + tol:
                hit = i
                break
        gt_matched.append(hit)
        if hit is not None:
            used.add(hit)
            status[hit] = "tp"
    for i, p in enumerate(preds):  # duplicates inside an already-credited interval
        if i in used:
            continue
        for g, m in zip(gts, gt_matched):
            if m is not None and p.clip == g.clip and p.category == g.category and g.start - tol <= p.t <= g.end + tol:
                status[i] = "dup"
                used.add(i)
                break
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    confused_events: set = set()
    for g, m in zip(gts, gt_matched):
        if m is not None:
            confusion[g.category][g.category] += 1
            continue
        wrong = [i for i, p in enumerate(preds) if i not in used and p.clip == g.clip and p.category != g.category
                 and g.start - tol <= p.t <= g.end + tol]
        if wrong:
            confusion[g.category][preds[wrong[0]].category] += 1
            confused_events.add(wrong[0])
        else:
            confusion[g.category][NONE] += 1
    fps = []
    for i, p in enumerate(preds):
        if status.get(i) in ("tp", "dup"):
            continue
        status[i] = "fp"
        fps.append(p)
        if i not in confused_events:
            confusion[NONE][p.category] += 1
    per_cat: Dict[str, Any] = {}
    for c in CATEGORIES:
        tp = sum(1 for i, p in enumerate(preds) if status[i] == "tp" and p.category == c)
        fp = sum(1 for p in fps if p.category == c)
        fn = sum(1 for g, m in zip(gts, gt_matched) if g.category == c and m is None)
        per_cat[c] = {"tp": tp, "fp": fp, "fn": fn, "n_gt": sum(1 for g in gts if g.category == c),
                      **_prf(tp, fp, fn)}
    TP = sum(v["tp"] for v in per_cat.values())
    FP = sum(v["fp"] for v in per_cat.values())
    FN = sum(v["fn"] for v in per_cat.values())
    buckets = []
    for lo, hi in zip(RELIABILITY_EDGES[:-1], RELIABILITY_EDGES[1:]):
        sel = [i for i, p in enumerate(preds) if lo <= p.conf < hi and status[i] != "dup"]
        ntp = sum(1 for i in sel if status[i] == "tp")
        buckets.append({"range": [lo, min(hi, 1.0)], "n": len(sel), "n_tp": ntp,
                        "precision": (ntp / len(sel)) if sel else None})
    return {"per_category": per_cat, "micro": {"tp": TP, "fp": FP, "fn": FN, **_prf(TP, FP, FN)},
            "duplicates": sum(1 for s in status.values() if s == "dup"), "n_predictions": len(preds),
            "n_gt": len(gts), "confusion": confusion, "labels": list(LABELS),
            "false_positives": [{"clip": p.clip, "t": p.t, "category": p.category, "conf": p.conf, "id": p.event_id}
                                for p in fps],
            "reliability": buckets,
            "gt_matches": [{"clip": g.clip, "category": g.category, "start": g.start, "end": g.end,
                            "matched_event": preds[m].event_id if m is not None else None,
                            "event_t": preds[m].t if m is not None else None}
                           for g, m in zip(gts, gt_matched)]}


def fp_per_hour(n_fp: int, hours: float) -> Optional[float]:
    return n_fp / hours if hours > 0 else None


def split_variants(events: Sequence[Dict[str, Any]], cat_map: Dict[str, str]) -> Tuple[List[Pred], List[Pred], bool]:
    """Return (heuristics_only, vlm_assisted, vlm_executed) from stored event rows."""
    heur: List[Pred] = []
    vlm: List[Pred] = []
    for e in events:
        if e["trigger_type"] == "scene_index" or e.get("media_time_s") is None:
            continue
        clip = Path(e["source_path"]).name
        hcat = cat_map.get(e["trigger_type"], e["category"])
        conf_h = float(e["heuristic_score"])
        heur.append(Pred(clip, float(e["media_time_s"]), hcat, conf_h, e["verdict"], e["id"]))
        if e["verdict"] != "rejected":
            vlm.append(Pred(clip, float(e["media_time_s"]), e["category"], float(e["confidence_score"]), e["verdict"], e["id"]))
    executed = any(e["verdict"] in ("verified", "rejected") for e in events)
    return heur, vlm, executed


# --------------------------------------------------------------- running
def clip_duration(path: Path) -> Optional[float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        return float(n / fps) if n > 0 and fps > 0 else None
    finally:
        cap.release()


def build_profile(base: Optional[str], out_dir: Path, live_dir: Path, device: Optional[str], vlm: str) -> Path:
    import yaml

    from vigil.config import deep_merge

    raw: Dict[str, Any] = {}
    if base:
        p = Path(base)
        raw = yaml.safe_load(((p if p.is_absolute() or p.exists() else ROOT / p)).read_text(encoding="utf-8")) or {}
    over: Dict[str, Any] = {"name": "eval", "paths": {
        "sqlite": str(out_dir / "vigil.db"), "chroma": str(out_dir / "chroma"), "keyframes": str(out_dir / "keyframes"),
        "live": str(live_dir), "clips": str(out_dir / "clips"), "jobs": str(out_dir / "jobs"),
        "cache": str(ROOT / "data" / "cache")}}
    if device:
        over["detector"] = {"device": device}
    if vlm == "off":
        over["vlm"] = {"enabled": False}
    prof = deep_merge(raw, over)
    live_dir.mkdir(parents=True, exist_ok=True)
    pp = live_dir / "profile.yaml"
    pp.write_text(yaml.safe_dump(prof, sort_keys=False), encoding="utf-8")
    return pp


def run_clip(row: ManifestRow, profile: Path, args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "vigil.pipeline", "--profile", str(profile), "--speed", "0", "--exit-when-done",
           "--no-jobs", "--only", row.camera_id, "--source", f"{row.camera_id}={row.clip}"]
    if args.vlm == "off":
        cmd.append("--no-vlm")
    if args.stub_embedder:
        cmd.append("--test-stub-embedder")
    env = dict(os.environ, ANONYMIZED_TELEMETRY="False", PYTHONUNBUFFERED="1")
    t0 = time.time()
    try:
        cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
        rc, tail = cp.returncode, (cp.stdout + cp.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        rc, tail = -9, f"timeout after {args.timeout}s"
    return {"clip": row.clip, "returncode": rc, "wall_s": round(time.time() - t0, 1), "log_tail": tail}


def read_events(db: Path) -> List[Dict[str, Any]]:
    if not db.exists():
        return []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute("SELECT * FROM events ORDER BY timestamp")]
    finally:
        con.close()


def latency_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vlm = [e["t_vlm_end"] - e["t_vlm_start"] for e in events
           if e["verdict"] in ("verified", "rejected") and e["t_vlm_end"] > e["t_vlm_start"] > 0]
    e2e = [e["t_stored"] - e["t_trigger"] for e in events if e["t_stored"] > e["t_trigger"] > 0]
    r = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"vlm_s": {"n": len(vlm), "p50": r(percentile(vlm, 50)), "p95": r(percentile(vlm, 95))},
            "trigger_to_stored_s": {"n": len(e2e), "p50": r(percentile(e2e, 50)), "p95": r(percentile(e2e, 95)),
                                    "note": "offline replay at --speed 0: queueing delay, not live latency"}}


# ------------------------------------------------------------- retrieval
def run_retrieval(profile_path: Path, queries: Path, events: List[Dict[str, Any]], stub: bool) -> Dict[str, Any]:
    """Recall@3 / MRR per retrieval mode. Only with REAL CLIP weights."""
    if stub:
        return {"executed": False, "reason": "stub embedder in use: similarities are not semantic"}
    try:
        from vigil.config import load_profile
        from vigil.embeddings import VectorIndex, make_embedder
        from vigil.query import SearchEngine, StructuredFilter
        from vigil.storage import EventStore, reindex_pending

        prof = load_profile(str(profile_path))
        emb = make_embedder(prof)
        emb.load()
    except Exception as e:  # noqa: BLE001
        return {"executed": False, "reason": f"CLIP unavailable: {type(e).__name__}: {str(e)[:200]}"}
    store = EventStore(prof["paths"]["sqlite"])
    index = VectorIndex(prof["paths"]["chroma"], emb.tag, emb.dim)
    try:
        reindex_pending(store, emb, index, prof["paths"]["keyframes"])
    except Exception as e:  # noqa: BLE001
        return {"executed": False, "reason": f"indexing failed: {type(e).__name__}: {str(e)[:200]}"}
    engine = SearchEngine(store, index, emb, prof["retrieval"], prof["timezone"])
    with open(queries, newline="", encoding="utf-8") as fh:
        qs = [(r["query"], {c.strip() for c in r["relevant_clips"].split(";") if c.strip()}) for r in csv.DictReader(fh)]
    vlm_summaries = any(e["verdict"] in ("verified", "rejected") for e in events)
    out: Dict[str, Any] = {"executed": True, "n_queries": len(qs), "embedder": emb.tag,
                           "summary_text_source": "VLM summaries" if vlm_summaries else
                           "heuristic text only (VLM not executed): summary/hybrid modes are NOT representative",
                           "modes": {}}
    for mode in ("image", "summary", "hybrid"):
        rr, hit3, per_q = [], 0, []
        for q, rel in qs:
            res = engine.search(StructuredFilter(visual_query=q, limit=10, source="manual"), mode=mode).results
            rank = next((r.rank for r in res if Path(r.event.source_path).name.replace(".mp4", "") in
                         {Path(c).name.replace(".mp4", "") for c in rel}), None)
            rr.append(1.0 / rank if rank else 0.0)
            hit3 += 1 if rank and rank <= 3 else 0
            per_q.append({"query": q, "first_relevant_rank": rank})
        out["modes"][mode] = {"recall_at_3": hit3 / len(qs) if qs else None, "mrr": sum(rr) / len(rr) if rr else None,
                              "per_query": per_q}
    out["caveat"] = ("Clip-level relevance on a few SYNTHETIC clips and 8 hand-written queries: indicative only, "
                     "not a retrieval benchmark.")
    return out


# ------------------------------------------------------------------ main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "eval" / "manifest.csv"))
    ap.add_argument("--queries", default=str(ROOT / "eval" / "queries.csv"))
    ap.add_argument("--profile", default=None, help="base profile (default: defaults.yaml only)")
    ap.add_argument("--device", default=None, help="override detector.device (e.g. cpu)")
    ap.add_argument("--vlm", choices=("auto", "off"), default="auto",
                    help="auto = use the VLM from the profile if reachable (else events stay unverified); off = heuristics only")
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds of slack around each GT interval")
    ap.add_argument("--skip-run", action="store_true", help="re-score an existing run directory")
    ap.add_argument("--run-dir", default=str(ROOT / "eval" / "results" / "run"))
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "eval.json"))
    ap.add_argument("--timeout", type=int, default=3600, help="per-clip pipeline timeout (s)")
    ap.add_argument("--stub-embedder", action="store_true", help="TEST ONLY: no CLIP; retrieval reported as not executed")
    ap.add_argument("--clips", nargs="*", help="only these clip basenames (debug)")
    args = ap.parse_args(argv)

    from vigil.config import load_profile

    rows = load_manifest(Path(args.manifest))
    runnable = [r for r in rows if r.status == "runnable" and (not args.clips or Path(r.clip).name in args.clips)]
    not_run = [r for r in rows if r.status != "runnable"]
    missing = [r.clip for r in runnable if not (ROOT / r.clip).exists()]
    if missing:
        print("missing clips (run scripts/make_demo_clips.py):", *missing, sep="\n  ")
        return 2
    run_dir = Path(args.run_dir)
    if not args.skip_run:
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    runtime: Dict[str, Any] = {}
    profile_for_store = build_profile(args.profile, run_dir, run_dir / "live" / "_store", args.device, args.vlm)
    for r in runnable:
        name = Path(r.clip).stem
        prof = build_profile(args.profile, run_dir, run_dir / "live" / name, args.device, args.vlm)
        if not args.skip_run:
            print(f"[eval] running {r.clip} on {r.camera_id} ...", flush=True)
            runs.append(run_clip(r, prof, args))
            print(f"[eval]   rc={runs[-1]['returncode']} wall={runs[-1]['wall_s']}s", flush=True)
        st = run_dir / "live" / name / "status.json"
        if st.exists():
            try:
                runtime[name] = json.loads(st.read_text(encoding="utf-8")).get("metrics", {})
            except ValueError:
                pass
    failed = [x for x in runs if x["returncode"] != 0]

    profile = load_profile(str(profile_for_store))
    events = read_events(Path(profile["paths"]["sqlite"]))
    gts = ground_truth(runnable)
    heur, vlm_preds, vlm_executed = split_variants(events, profile["categories"])
    durations = {Path(r.clip).name: clip_duration(ROOT / r.clip) for r in runnable}
    total_h = sum(d for d in durations.values() if d) / 3600.0
    neg_clips = {Path(r.clip).name for r in runnable if r.category == NONE}
    neg_h = sum(durations[c] or 0 for c in neg_clips) / 3600.0

    def variant(preds: List[Pred]) -> Dict[str, Any]:
        v = evaluate_variant(gts, preds, args.tolerance)
        n_fp_neg = sum(1 for f in v["false_positives"] if f["clip"] in neg_clips)
        v["false_alarms_per_hour_all_footage"] = fp_per_hour(v["micro"]["fp"], total_h)
        v["false_alarms_per_hour_event_free_footage"] = fp_per_hour(n_fp_neg, neg_h)
        return v

    unver = sum(1 for e in events if e["verdict"] == "unverified" and e["trigger_type"] != "scene_index")
    n_cand = sum(1 for e in events if e["trigger_type"] != "scene_index")
    result: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "setup": {"profile": args.profile, "device_override": args.device, "vlm_mode": args.vlm,
                  "tolerance_s": args.tolerance, "python": sys.version.split()[0], "platform": sys.platform,
                  "vlm_model": profile["vlm"]["model"] if vlm_executed else None},
        "clips": [{"clip": r.clip, "camera_id": r.camera_id, "category": r.category, "duration_s": durations[Path(r.clip).name],
                   "provenance": r.provenance} for r in runnable],
        "not_run_datasets": [{"clip": r.clip, "category": r.category, "provenance": r.provenance, "notes": r.notes}
                             for r in not_run],
        "pipeline_runs": runs, "pipeline_failures": [x["clip"] for x in failed],
        "footage_hours": total_h, "event_free_footage_hours": neg_h,
        "n_candidates": n_cand, "unverified_fraction": (unver / n_cand) if n_cand else None,
        "vlm_executed": vlm_executed,
        "heuristics_only": variant(heur),
        "vlm_assisted": variant(vlm_preds) if vlm_executed else {
            "executed": False, "reason": "no event carries a VLM verdict (Ollama/VLM not reachable during this run); "
                                         "vlm_assisted would equal heuristics_only"},
        "latency": latency_summary(events),
        "runtime_metrics": runtime,
        "retrieval": run_retrieval(profile_for_store, Path(args.queries), events, args.stub_embedder),
        "caveats": [
            "Clips are synthetic (pasted person cut-outs on a plain floor); results show the pipeline works end to end, "
            "not real-world accuracy.",
            "The fighting clip's choreography was tuned until the heuristic fired.",
            "confidence_score is an uncalibrated ranking heuristic, not a probability.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    h = result["heuristics_only"]["micro"]
    print(f"[eval] heuristics-only micro: TP={h['tp']} FP={h['fp']} FN={h['fn']} -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
