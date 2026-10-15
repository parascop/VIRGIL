#!/usr/bin/env python
"""Benchmark the candidate Ollama vision models on THIS machine and write the
winner into the profile YAML.

For each eligible model it: verifies availability (optionally `ollama pull`s it),
does a load + warm-up, runs 10 fixed frame composites at each composite size,
and records JSON-valid rate, p50/p95 latency, peak memory and GPU placement.
Models are benchmarked one at a time and unloaded in between.

Selection rule: JSON-valid rate is the primary criterion; p95 <= 12 s; peak memory
within the profile budget; among passing models prefer the smaller one.
If nothing passes every target the best safe fallback is chosen and the unmet
targets are reported. Nothing here is estimated: unmeasurable values are null.

  python scripts/benchmark_models.py --profile configs/profile_rtx2050.yaml --pull --write-profile
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from vigil.util import percentile  # noqa: E402

CANDIDATES = ["moondream", "qwen3-vl:2b", "granite3.2-vision", "qwen2.5vl:3b", "gemma3:4b"]
TOO_BIG = re.compile(r"(?<![\d.])(7|8|9|11|12|13|14|20|27|30|32|34|70|72)b\b", re.I)  # never benchmark >= 7B


# ------------------------------------------------------------------ samples
def sample_composites(n: int, max_side: int) -> List[Tuple[np.ndarray, int, str, str]]:
    """n fixed composites: (image, sample_count, claimed_category, reason). Uses the
    synthetic demo clips when present, else crops of the Ultralytics sample photo."""
    import cv2

    from vigil.composite import build_composite
    from vigil.types import Frame

    out: List[Tuple[np.ndarray, int, str, str]] = []
    claims = [("person down", "box aspect ratio flipped and stayed horizontal"), ("tailgating", "two tracks crossed the entry line 0.8 s apart"),
              ("suspicious activity", "a person stayed 8 s in a restricted zone"), ("crowd incident", "8 people in the frame"),
              ("fighting", "two close people with high relative motion")]
    clips = sorted((ROOT / "data" / "clips").glob("*_01.mp4"))
    k = 0
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        for start in (0.35, 0.6):
            if len(out) >= n or total < 40:
                break
            frames = []
            for i in range(4):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (start + 0.08 * i)))
                ok, img = cap.read()
                if ok:
                    frames.append(Frame("bench", i, i * 1.0, img))
            if len(frames) == 4:
                comp, cnt, _ = build_composite(frames, max_side)
                cat, why = claims[k % len(claims)]
                out.append((comp, cnt, cat, why))
                k += 1
        cap.release()
    if len(out) < n:  # fallback: crops of the sample photo
        import ultralytics

        base = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
        h, w = base.shape[:2]
        rng = np.random.RandomState(0)
        while len(out) < n:
            frames = []
            for i in range(4):
                x = rng.randint(0, w // 4)
                y = rng.randint(0, h // 4)
                frames.append(Frame("bench", i, i * 1.0, base[y:y + h * 3 // 4, x:x + w * 3 // 4].copy()))
            comp, cnt, _ = build_composite(frames, max_side)
            cat, why = claims[len(out) % len(claims)]
            out.append((comp, cnt, cat, why))
    return out[:n]


# --------------------------------------------------------------------- probes
class GpuSampler(threading.Thread):
    """Samples nvidia-smi used-memory while a benchmark runs (whole GPU, includes Ollama)."""

    def __init__(self, period: float = 0.4):
        super().__init__(daemon=True)
        self.period, self.peak, self.base, self.total = period, None, None, None
        self._halt = threading.Event()

    def run(self) -> None:
        from vigil.metrics import nvidia_smi_memory

        while not self._halt.is_set():
            used, total = nvidia_smi_memory()
            if used is not None:
                self.total = total
                self.base = used if self.base is None else self.base
                self.peak = used if self.peak is None else max(self.peak, used)
            self._halt.wait(self.period)

    def stop(self) -> Dict[str, Optional[float]]:
        self._halt.set()
        self.join(2)
        return {"gpu_base_used_mb": self.base, "gpu_peak_used_mb": self.peak, "gpu_total_mb": self.total}


def placement(client: Any, model: str) -> Dict[str, Any]:
    for m in client.ps():
        if m.get("name", "").startswith(model.split(":")[0]):
            size, vram = m.get("size") or 0, m.get("size_vram") or 0
            frac = (vram / size) if size else None
            return {"ollama_loaded_mb": round(size / 1e6, 1) if size else None,
                    "gpu_fraction": None if frac is None else round(frac, 3),
                    "placement": ("unknown" if frac is None else "GPU" if frac > 0.98 else "CPU" if frac < 0.02 else f"partial offload ({frac:.0%} on GPU)")}
    return {"ollama_loaded_mb": None, "gpu_fraction": None, "placement": "unknown (model not listed by /api/ps)"}


def bench_model(client: Any, vlm_cfg: Dict[str, Any], model: str, sizes: Sequence[int], n: int,
                extra_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Benchmark one model across composite sizes. Returns a result dict."""
    from vigil.types import Trigger, VLMRequest
    from vigil.vlm import VLMVerifier

    cfg = dict(vlm_cfg, model=model, retries=0)  # raw (first-attempt) JSON validity
    if extra_options and "num_gpu" in extra_options:
        cfg["num_gpu"] = extra_options["num_gpu"]
    ver = VLMVerifier(cfg, client)
    res: Dict[str, Any] = {"model": model, "status": "ok", "per_size": {}, "options": extra_options or {}}
    client.unload(model)
    time.sleep(0.5)
    # ---- load + warm-up (reported separately from steady state) ----
    t0 = time.time()
    comps = sample_composites(n, sizes[0])
    c, cnt, cat, why = comps[0]
    tg = Trigger("probe", cat, "bench", 0.0, (1,), 0.5, reason=why)
    ver.verify(VLMRequest("warm", tg, c, c, cnt, time.time()))
    res["load_warmup_s"] = round(time.time() - t0, 2)
    for side in sizes:
        comps = sample_composites(n, side)
        sampler = GpuSampler()
        sampler.start()
        lat_ok: List[float] = []
        lat_all: List[float] = []
        valid = 0
        errors: List[str] = []
        for i, (comp, cnt, cat, why) in enumerate(comps):
            tg = Trigger("probe", cat, "bench", 0.0, (1,), 0.5, reason=why)
            r = ver.verify(VLMRequest(f"b{i}", tg, comp, comp, cnt, time.time()))
            d = r.t_end - r.t_start
            lat_all.append(d)
            if r.ok:
                valid += 1
                lat_ok.append(d)
            else:
                errors.append(r.error[:120])
        mem = sampler.stop()
        pl = placement(client, model)
        gpu_delta = (mem["gpu_peak_used_mb"] - mem["gpu_base_used_mb"]) if mem["gpu_peak_used_mb"] is not None else None
        res["per_size"][str(side)] = {
            "n": len(comps), "valid": valid, "valid_rate": round(valid / len(comps), 3),
            "p50_s": None if not lat_ok else round(percentile(lat_ok, 50), 3),
            "p95_s": None if not lat_ok else round(percentile(lat_ok, 95), 3),
            "p50_all_s": round(percentile(lat_all, 50), 3), "p95_all_s": round(percentile(lat_all, 95), 3),
            "gpu_peak_used_mb": mem["gpu_peak_used_mb"], "gpu_total_mb": mem["gpu_total_mb"], "gpu_delta_mb": gpu_delta,
            **pl, "errors": sorted(set(errors))[:3]}
    client.unload(model)
    return res


# ------------------------------------------------------------------ selection
def select_best(results: Sequence[Dict[str, Any]], sizes_bytes: Dict[str, int], budget_mb: Optional[float],
                p95_target_s: float = 12.0, min_valid: float = 0.9, tolerance: float = 0.05) -> Dict[str, Any]:
    """Pure function (unit-tested). Returns {"choice": {...}|None, "unmet": [...], "passed": bool}."""
    rows = []
    for r in results:
        if r.get("status") != "ok":
            continue
        for side, s in r["per_size"].items():
            mem = s.get("gpu_peak_used_mb") if s.get("gpu_peak_used_mb") is not None else s.get("ollama_loaded_mb")
            rows.append({"model": r["model"], "side": int(side), "valid_rate": s["valid_rate"], "p95_s": s["p95_s"],
                         "p50_s": s["p50_s"], "mem_mb": mem, "size_bytes": sizes_bytes.get(r["model"]) or sizes_bytes.get(r["model"] + ":latest") or 0,
                         "stats": s, "options": r.get("options", {})})

    def unmet(row: Dict[str, Any]) -> List[str]:
        u = []
        if row["valid_rate"] < min_valid:
            u.append(f"valid JSON rate {row['valid_rate']:.0%} < {min_valid:.0%}")
        if row["p95_s"] is None or row["p95_s"] > p95_target_s:
            u.append(f"p95 latency {row['p95_s']} s > {p95_target_s} s")
        if budget_mb is not None and row["mem_mb"] is not None and row["mem_mb"] > budget_mb:
            u.append(f"memory {row['mem_mb']:.0f} MB > budget {budget_mb:.0f} MB")
        return u

    passing = [r for r in rows if not unmet(r)]
    if passing:
        best_valid = max(r["valid_rate"] for r in passing)
        near = [r for r in passing if r["valid_rate"] >= best_valid - tolerance]
        choice = min(near, key=lambda r: (r["size_bytes"] or 1e18, r["p95_s"], r["side"]))
        return {"choice": choice, "unmet": [], "passed": True}
    usable = [r for r in rows if r["valid_rate"] > 0]
    if not usable:
        return {"choice": None, "unmet": ["no model produced any valid structured output"], "passed": False}
    choice = max(usable, key=lambda r: (r["valid_rate"], -len(unmet(r)), -(r["p95_s"] or 1e9)))
    return {"choice": choice, "unmet": unmet(choice), "passed": False}


# ------------------------------------------------------------- profile update
BEGIN, END = "# BEGIN BENCHMARK (written by scripts/benchmark_models.py)", "# END BENCHMARK"


def update_profile_text(text: str, model: str, side: int, num_gpu: Optional[int], bench_block: Dict[str, Any]) -> str:
    import yaml

    lines = text.splitlines()
    out, in_vlm, seen_gpu = [], False, False
    for ln in lines:
        if re.match(r"^vlm:\s*(#.*)?$", ln):
            in_vlm = True
        elif re.match(r"^\S", ln):
            in_vlm = False
        if in_vlm and re.match(r"^\s+model:", ln):
            ln = re.sub(r"model:\s*\S+", f"model: {model}", ln)
        if in_vlm and re.match(r"^\s+composite_max_side:", ln):
            ln = re.sub(r"composite_max_side:\s*\d+", f"composite_max_side: {side}", ln)
        if in_vlm and re.match(r"^\s+num_gpu:", ln):
            seen_gpu = True
            ln = re.sub(r"num_gpu:\s*\S+", f"num_gpu: {num_gpu if num_gpu is not None else 'null'}", ln)
        out.append(ln)
    text2 = "\n".join(out)
    if num_gpu is not None and not seen_gpu:
        text2 = re.sub(r"(^vlm:[^\n]*\n)", lambda m: m.group(1) + f"  num_gpu: {num_gpu}\n", text2, count=1, flags=re.M)
    text2 = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", "", text2, flags=re.S).rstrip("\n") + "\n"
    block = yaml.safe_dump({"benchmark": bench_block}, sort_keys=False, default_flow_style=False)
    return text2 + f"\n{BEGIN}\n{block}{END}\n"


# ------------------------------------------------------------------------ main
def ollama_pull(model: str, timeout: int = 1800) -> Tuple[bool, str]:
    if shutil.which("ollama") is None:
        return False, "`ollama` command not found"
    try:
        r = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr or r.stdout).strip()[-200:]
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vigil.config import CONFIG_DIR, load_profile
    from vigil.device import detect_hardware, select_profile
    from vigil.vlm import OllamaClient

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    ap.add_argument("--sizes", nargs="*", type=int, default=None, help="composite max sides (default: 512 768)")
    ap.add_argument("--frames", type=int, default=10, help="composites per model/size (spec: 10)")
    ap.add_argument("--pull", action="store_true", help="try `ollama pull` for models that are not present")
    ap.add_argument("--compare-cpu-vlm", action="store_true", help="also benchmark the winner with num_gpu=0 (CPU VLM)")
    ap.add_argument("--write-profile", action="store_true", help="write the winner into the profile YAML")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--p95-target", type=float, default=12.0)
    ap.add_argument("--min-valid", type=float, default=0.9)
    a = ap.parse_args(argv)

    hw = detect_hardware()
    prof_path = Path(a.profile) if a.profile else select_profile(hw, CONFIG_DIR)[0]
    prof = load_profile(str(prof_path))
    vcfg = dict(prof["vlm"])
    if a.base_url:
        vcfg["base_url"] = a.base_url
    client = OllamaClient(vcfg["base_url"], vcfg["connect_timeout_s"], max(vcfg["timeout_s"], 120))
    ver = client.version()
    if ver is None:
        print(f"Ollama is not reachable at {vcfg['base_url']}. Install/start Ollama first (see README). Nothing benchmarked.")
        return 2
    sizes = a.sizes or ([512, 768] if prof["name"] != "defaults" else [512])
    print(f"Ollama {ver}; profile {prof_path.name}; GPU: {hw.get('gpu_name')}; composite sizes {sizes}")
    results: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    sizes_bytes: Dict[str, int] = {}
    for model in a.models:
        if TOO_BIG.search(model):
            skipped[model] = "7B-or-larger models are not benchmarked on the target hardware"
            continue
        try:
            have = client.has_model(model)
        except Exception as e:  # noqa: BLE001
            skipped[model] = f"could not query Ollama: {e}"
            continue
        if not have and a.pull:
            ok, msg = ollama_pull(model)
            have = ok and client.has_model(model)
            if not have:
                skipped[model] = f"pull failed: {msg}"
                continue
        if not have:
            skipped[model] = "tag not pulled/unavailable (run with --pull or `ollama pull`)"
            continue
        sizes_bytes.update(client.tag_sizes())
        print(f"== {model}")
        try:
            r = bench_model(client, vcfg, model, sizes, a.frames)
        except Exception as e:  # noqa: BLE001
            skipped[model] = f"benchmark crashed: {type(e).__name__}: {e}"
            continue
        results.append(r)
        for side, s in r["per_size"].items():
            print(f"   {side}px: valid {s['valid']}/{s['n']}  p50 {s['p50_s']}s  p95 {s['p95_s']}s  placement {s['placement']}  "
                  f"gpu_peak_used {s['gpu_peak_used_mb']} MB  loaded {s['ollama_loaded_mb']} MB")
    budget = prof.get("hardware", {}).get("vram_budget_mb")
    sel = select_best(results, sizes_bytes, budget, a.p95_target, a.min_valid)
    cpu_cmp = None
    if a.compare_cpu_vlm and sel["choice"]:
        m = sel["choice"]["model"]
        print(f"== {m} with CPU VLM (num_gpu=0) for comparison")
        cpu_cmp = bench_model(client, vcfg, m, [sel["choice"]["side"]], a.frames, {"num_gpu": 0})
        print("   ", cpu_cmp["per_size"])
    report = {"time": dt.datetime.now().isoformat(timespec="seconds"), "profile": str(prof_path.name), "ollama_version": ver,
              "hardware": {k: hw.get(k) for k in ("os", "os_release", "machine", "ram_mb", "gpu_name", "gpu_vram_mb", "torch", "cuda_runtime", "nvidia_smi")},
              "vram_budget_mb": budget, "targets": {"p95_s": a.p95_target, "min_valid_rate": a.min_valid},
              "results": results, "skipped": skipped, "selection": {k: v for k, v in sel.items() if k != "choice"},
              "choice": None if sel["choice"] is None else {k: v for k, v in sel["choice"].items() if k != "stats"} | {"stats": sel["choice"]["stats"]},
              "cpu_vlm_comparison": cpu_cmp}
    out = Path(a.out) if a.out else ROOT / "eval" / "results" / f"model_benchmark_{prof['name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nskipped: {skipped or 'none'}")
    if sel["choice"] is None:
        print("NO usable model: " + "; ".join(sel["unmet"]))
        return 1
    c = sel["choice"]
    print(f"SELECTED {c['model']} @ {c['side']}px: valid {c['valid_rate']:.0%}, p50 {c['p50_s']} s, p95 {c['p95_s']} s, "
          f"mem {c['mem_mb']} MB. {'All targets met.' if sel['passed'] else 'TARGETS NOT MET: ' + '; '.join(sel['unmet'])}")
    if a.write_profile:
        block = {"measured_at": report["time"], "ollama_version": ver, "gpu": hw.get("gpu_name"), "os": f"{hw['os']} {hw['os_release']}",
                 "selected_model": c["model"], "composite_max_side": c["side"], "valid_json_rate": c["valid_rate"],
                 "p50_s": c["p50_s"], "p95_s": c["p95_s"], "peak_mem_mb": c["mem_mb"], "placement": c["stats"]["placement"],
                 "all_targets_met": sel["passed"], "unmet_targets": sel["unmet"], "runtime_options": c["options"] or {"num_gpu": "ollama default"}}
        text = update_profile_text(prof_path.read_text(), c["model"], c["side"], (c["options"] or {}).get("num_gpu"), block)
        prof_path.write_text(text)
        print(f"profile updated: {prof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
