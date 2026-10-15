#!/usr/bin/env python
"""Generate REPORT.md and plots from eval/results/*.json.

Only numbers present in the JSON files are used. Anything that was not executed
(VLM, real CLIP, GPU, macOS, Windows, external datasets) is printed as
NOT EXECUTED together with the command that would produce it.

Usage: python eval/make_report.py [--eval eval/results/eval.json] [--out REPORT.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"

NE = "**NOT EXECUTED**"


def pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def num(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def plots(ev: Dict[str, Any], out: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made: List[str] = []
    h = ev["heuristics_only"]
    cats = list(h["per_category"])
    # 1. precision / recall per category (heuristics vs VLM-assisted when executed)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    series = [("heuristics only", h)]
    if ev.get("vlm_executed") and "per_category" in ev["vlm_assisted"]:
        series.append(("VLM assisted", ev["vlm_assisted"]))
    w = 0.8 / (2 * len(series))
    for si, (name, v) in enumerate(series):
        for mi, m in enumerate(("precision", "recall")):
            vals = [(v["per_category"][c][m] or 0.0) for c in cats]
            xs = [i + (si * 2 + mi) * w for i in range(len(cats))]
            ax.bar(xs, vals, w, label=f"{name} {m}")
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(cats))])
    ax.set_xticklabels(cats, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-category precision / recall (synthetic clips)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "per_category.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    made.append(p.name)
    # 2. confusion matrix
    labels = h["labels"]
    mat = [[h["confusion"][a][b] for b in labels] for a in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, mat[i][j], ha="center", va="center", color="black")
    ax.set_xlabel("predicted (heuristic category)")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix, heuristics only")
    fig.tight_layout()
    p = out / "confusion.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    made.append(p.name)
    # 3. reliability
    rel = [b for b in h["reliability"] if b["n"]]
    if rel:
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        ax.bar([f"{b['range'][0]:.2f}-{b['range'][1]:.2f}\n(n={b['n']})" for b in rel], [b["precision"] for b in rel])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of events that were true detections")
        ax.set_xlabel("heuristic score bucket (uncalibrated)")
        ax.set_title("Reliability of the heuristic score")
        fig.tight_layout()
        p = out / "reliability.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p.name)
    return made


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def variant_table(v: Dict[str, Any]) -> str:
    rows = []
    for c, m in v["per_category"].items():
        rows.append([c, str(m["n_gt"]), str(m["tp"]), str(m["fp"]), str(m["fn"]), pct(m["precision"]), pct(m["recall"]),
                     pct(m["f1"])])
    mi = v["micro"]
    rows.append(["**all (micro)**", str(v["n_gt"]), str(mi["tp"]), str(mi["fp"]), str(mi["fn"]), pct(mi["precision"]),
                 pct(mi["recall"]), pct(mi["f1"])])
    return md_table(["category", "GT events", "TP", "FP", "FN", "precision", "recall", "F1"], rows)


def build(ev: Dict[str, Any], diag: Optional[Dict[str, Any]], bench: Optional[Dict[str, Any]], tests: str,
          plot_files: List[str]) -> str:
    s = ev["setup"]
    h = ev["heuristics_only"]
    L: List[str] = []
    a = L.append
    a("# Vigil evaluation report\n")
    a(f"_Generated {ev['generated_at']} by `eval/make_report.py` from `eval/results/eval.json`. Every number below comes from "
      "that file; nothing is estimated or copied from elsewhere._\n")
    a("## 1. Read this first\n")
    a(f"- **What was run:** the real pipeline (YOLOv8n + ByteTrack + heuristics + storage) on `{s['platform']}`, Python {s['python']}, "
      f"detector device override `{s['device_override']}`, base profile `{s['profile']}`.")
    a(f"- **VLM verification:** {'executed with ' + str(s['vlm_model']) if ev['vlm_executed'] else NE + ' (no Ollama/VLM was reachable). Every candidate was stored as `unverified`, so the VLM columns of the ablation do not exist.'}")
    r = ev["retrieval"]
    a(f"- **Retrieval (CLIP + Chroma):** {'executed' if r.get('executed') else NE + ' - ' + str(r.get('reason'))}.")
    a("- **GPU / macOS / Windows:** " + NE + " in this report (the sandbox is a 2-core x86-64 Linux CPU machine). See section 8 for the exact commands.")
    a("- **Data:** all clips are **synthetic** (person cut-outs pasted on a plain floor, `scripts/make_demo_clips.py`). They show that every "
      "stage works end to end and are deterministic, but they say **very little about accuracy on real CCTV footage**. "
      "The fighting clip's choreography was tuned until the heuristic fired, so its result is not independent evidence.")
    a("- `confidence_score` is an uncalibrated ranking heuristic, not a probability.\n")

    a("## 2. Clips\n")
    rows = [[f"`{c['clip']}`", c["camera_id"], c["category"], num(c["duration_s"], 0) + " s"] for c in ev["clips"]]
    a(md_table(["clip", "camera", "ground truth", "duration"], rows))
    a(f"\nTotal footage {num(ev['footage_hours'] * 60, 1)} min, of which {num(ev['event_free_footage_hours'] * 60, 1)} min is event-free.")
    a("\nNot run (datasets listed in `eval/manifest.csv` as `not_run`; download them yourself and add rows):\n")
    for d in ev["not_run_datasets"]:
        a(f"- `{d['clip']}` ({d['category']}): {d['provenance']}")
    a("")

    a(f"## 3. Detection results: heuristics only (tolerance +/-{s['tolerance_s']} s)\n")
    a(variant_table(h))
    a(f"\n- Duplicate same-category events inside an already-detected interval: {h['duplicates']} (not counted as false alarms).")
    a(f"- **False alarms per hour, event-free footage only:** {num(h['false_alarms_per_hour_event_free_footage'])} "
      f"(from {num(ev['event_free_footage_hours'] * 60, 1)} min of footage; a small denominator - treat as indicative).")
    a(f"- False alarms per hour, all footage: {num(h['false_alarms_per_hour_all_footage'])}.")
    a(f"- Candidates stored: {ev['n_candidates']}; unverified fraction: {pct(ev['unverified_fraction'])}.")
    a("\nWhen the alert fired (media time inside the clip):\n")
    a(md_table(["clip", "category", "GT interval", "alert at", "delay after GT start"],
               [[f"`{g['clip']}`", g["category"], f"{g['start']}-{g['end']} s",
                 "missed" if g["event_t"] is None else f"{g['event_t']:.1f} s",
                 "-" if g["event_t"] is None else f"{g['event_t'] - g['start']:.1f} s"] for g in h["gt_matches"]]))
    a("\nDelays are mostly by design (hold times such as `person_down.hold_s`, `suspicious.loiter_s`, `fighting.min_duration_s`).")
    if h["false_positives"]:
        a("\nFalse positives:\n")
        for f in h["false_positives"]:
            a(f"- `{f['clip']}` t={f['t']:.1f}s `{f['category']}` score={f['conf']:.2f}")
    missed = [g for g in h["gt_matches"] if g["matched_event"] is None]
    if missed:
        a("\nMissed ground-truth intervals:\n")
        for g in missed:
            a(f"- `{g['clip']}` {g['category']} [{g['start']}-{g['end']} s]")
    a("")
    for f in plot_files:
        if f in ("confusion.png", "per_category.png"):
            a(f"![{f}](eval/results/{f})")
    a("")

    a("## 4. Ablation: heuristics only vs VLM-assisted\n")
    v = ev["vlm_assisted"]
    if not ev["vlm_executed"]:
        a(f"{NE}. {v.get('reason', '')}")
        a("\nWhat this means: the system's behaviour *without* a VLM is exactly the heuristics-only table above, with every event "
          "labelled `unverified`. Whether the VLM removes false alarms (or wrongly rejects true events) is **not measured here**.")
        a("\nTo produce it on a machine with Ollama: `python scripts/download_models.sh`, then "
          "`python eval/run_eval.py --profile configs/profile_<yours>.yaml && python eval/make_report.py`.")
    else:
        a(variant_table(v))
        rej_tp = h["micro"]["tp"] - v["micro"]["tp"]
        a(f"\n- False alarms removed by the VLM: {h['micro']['fp'] - v['micro']['fp']}; true detections lost to VLM rejection: {max(rej_tp, 0)}.")
        a(f"- False alarms/hour (event-free footage): {num(h['false_alarms_per_hour_event_free_footage'])} -> "
          f"{num(v['false_alarms_per_hour_event_free_footage'])}.")
    a("")

    a("## 5. Reliability of the score\n")
    rel = [b for b in h["reliability"] if b["n"]]
    if rel:
        a(md_table(["heuristic score", "events", "true detections", "fraction"],
                   [[f"{b['range'][0]:.2f}-{b['range'][1]:.2f}", str(b["n"]), str(b["n_tp"]), pct(b["precision"])] for b in rel]))
        a("\nFew events per bucket: this is a sanity check that higher scores are not worse, not a calibration curve.")
    else:
        a("No events to bucket.")
    a("")

    a("## 6. Retrieval\n")
    if r.get("executed"):
        a(f"Embedder `{r['embedder']}`, {r['n_queries']} queries. Summary text source: {r['summary_text_source']}.\n")
        a(md_table(["mode", "Recall@3", "MRR"], [[m, pct(x["recall_at_3"]), num(x["mrr"], 3)] for m, x in r["modes"].items()]))
        a(f"\n{r['caveat']}")
    else:
        a(f"{NE}: {r.get('reason')}.\n")
        a("Query parsing, structured filtering and the hybrid-scoring maths are covered by unit tests (`tests/test_query.py`), "
          "which use a hash stub embedder and therefore prove the *plumbing*, not semantic quality. "
          "Run `python eval/run_eval.py ...` with CLIP weights cached (`bash scripts/download_models.sh`) to fill this in.")
    a("")

    a("## 7. Latency and resources (this CPU sandbox only)\n")
    lat = ev["latency"]
    a(f"- VLM latency: {NE if not lat['vlm_s']['n'] else 'p50 ' + num(lat['vlm_s']['p50'], 2) + ' s, p95 ' + num(lat['vlm_s']['p95'], 2) + ' s (n=' + str(lat['vlm_s']['n']) + ')'}")
    rt = ev.get("runtime_metrics", {}).get("normal_01") or {}
    st = rt.get("stages", {})
    if st:
        rows = [[k, str(m["n"]), num(m["p50_ms"]), num(m["p95_ms"])] for k, m in sorted(st.items())]
        a("\nPer-stage latency while replaying `normal_01.mp4` (offline, as fast as possible, CPU):\n")
        a(md_table(["stage", "n", "p50 ms", "p95 ms"], rows))
    a("\n`n` is the number of samples kept by the bounded metrics window, not the frame count. `ingest_to_detect` is queue "
      "wait: offline replay runs the reader faster than the detector, so it is back-pressure, not real-time latency.")
    mem = rt.get("memory", {})
    if mem:
        a(f"\nProcess RSS at the end of the run: {num(mem.get('rss_mb'), 0)} MB (peak {num(mem.get('peak_rss_mb'), 0)} MB). "
          "No GPU memory figure: no GPU present.")
    a("\nThese are **CPU** numbers on 2 cores. They say nothing about the RTX 2050 / GTX 1650 / M4 targets.")
    a("")
    if bench:
        a("### VLM model benchmark\n")
        a("`eval/results/" + bench["_file"] + "` exists; see that file. " + str(bench.get("note", "")))
    else:
        a("### VLM model benchmark\n")
        a(f"{NE}. `scripts/benchmark_models.py` needs Ollama and the candidate models on the target machine. "
          "Until it has run, the profiles keep the *default* `moondream` and that choice is **unverified**. "
          "Run `python scripts/benchmark_models.py --pull --write-profile` on each machine.")
    a("")

    a("## 8. Verification status by platform\n")
    rows = []
    if diag:
        for c in diag["results"]:
            if c["status"] in ("PASS", "FAIL", "NOT_INSTALLED", "UNSUPPORTED", "MODEL_UNAVAILABLE", "NOT_RUN"):
                rows.append([c["check"], c["status"], str(c["detail"]).replace("|", "/")[:110]])
        a("`scripts/diagnose.py` on the sandbox (`eval/results/diagnose_sandbox.json`):\n")
        a(md_table(["check", "status", "detail"], rows))
        a("")
    a(md_table(["target", "status", "command that verifies it"], [
        ["Linux x86-64 CPU (sandbox)", "tests + eval executed here", "`python -m pytest -q`"],
        ["Windows + NVIDIA RTX 2050", "**unverified**", "`python scripts/diagnose.py --profile configs/profile_rtx2050.yaml` then `python scripts/benchmark_models.py --pull --write-profile`"],
        ["Windows + NVIDIA GTX 1650", "**unverified**", "`python scripts/diagnose.py --profile configs/profile_gtx1650.yaml`"],
        ["macOS M4 (MPS)", "**unverified**", "`python scripts/diagnose.py --profile configs/profile_mac_m4.yaml`"],
        ["Ollama VLM path", "unit-tested against a mock HTTP server only", "`python scripts/diagnose.py` (checks the real server)"],
        ["CLIP weights", "code path tested with random weights only", "`bash scripts/download_models.sh` then `python scripts/diagnose.py`"],
    ]))
    a("")
    a("## 9. Test suite\n")
    a(tests.strip() if tests else f"{NE} when this report was generated (run `python -m pytest -q`).")
    a("\nTests use fakes (detector, verifier), a mock Ollama HTTP server and a hash stub embedder. They check logic, "
      "failure handling and shutdown, not model quality.\n")
    a("## 10. Known limitations\n")
    for c in ev["caveats"]:
        a(f"- {c}")
    a("- **History of this result:** a first eval run (before the fix below) *missed* the fighting clip although the heuristic fired "
      "on it in isolation. Cause: velocity-window boundaries depended on float rounding of the timestamps (epoch-sized timestamps "
      "flipped a 4-frame window into 5 frames on 15-40% of frames), so the trigger was knife-edge. `TrackState._obs_at_or_before` now "
      "has a 1 ms tolerance and `tests/test_heuristics.py::test_velocity_window_independent_of_timestamp_offset` guards it; "
      "all numbers above are from the run after the fix. The fighting rule is still sensitive to its `motion_energy_min` threshold.")
    a("- Heuristic thresholds are untuned defaults chosen for the synthetic clips; real cameras need per-camera tuning in `configs/cameras.yaml`.")
    a("- 2-D boxes cannot separate a fight from hugging, or a fall from sitting down; the VLM stage exists for that, but its effect is not measured here.")
    a("- One manifest row per ground-truth interval and a +/-tolerance match rule: small sample sizes mean percentages jump by large steps.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(RES / "eval.json"))
    ap.add_argument("--out", default=str(ROOT / "REPORT.md"))
    a = ap.parse_args(argv)
    ev = load(Path(a.eval))
    if ev is None:
        print(f"cannot read {a.eval}; run eval/run_eval.py first")
        return 2
    diag = load(RES / "diagnose_sandbox.json")
    bench = None
    for f in sorted(RES.glob("benchmark*.json")):
        bench = load(f)
        if bench:
            bench["_file"] = f.name
    tp = RES / "pytest_summary.txt"
    tests = tp.read_text(encoding="utf-8") if tp.exists() else ""
    files = plots(ev, RES)
    Path(a.out).write_text(build(ev, diag, bench, tests, files), encoding="utf-8")
    print(f"wrote {a.out} and plots {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
