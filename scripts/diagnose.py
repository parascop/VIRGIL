#!/usr/bin/env python
"""Cross-platform smoke test / hardware diagnostic.

Every check ends in exactly one status, so results are never over-claimed:
  PASS               the test actually ran and succeeded
  FAIL               the test ran and failed
  NOT_INSTALLED      a required Python package / binary is missing
  UNSUPPORTED        installed but not usable on the active device / platform
  MODEL_UNAVAILABLE  the model weights/tag are not present (or cannot be fetched)
  NOT_RUN            the test was skipped (reason given); NEVER counted as passed
  INFO               a fact, not a test

Usage: python scripts/diagnose.py [--profile configs/profile_x.yaml] [--json out.json]
                                  [--skip-vlm] [--skip-clip] [--skip-streamlit]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

RESULTS: List[Dict[str, Any]] = []


def rec(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"check": name, "status": status, "detail": detail})
    icon = {"PASS": "[ PASS ]", "FAIL": "[ FAIL ]", "NOT_INSTALLED": "[NOT INST]", "UNSUPPORTED": "[UNSUPP ]",
            "MODEL_UNAVAILABLE": "[NO MODEL]", "NOT_RUN": "[NOT RUN]", "INFO": "[ info ]"}[status]
    print(f"{icon} {name}: {detail}")


def guarded(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        rec(name, "FAIL", f"{type(e).__name__}: {e}")


def sample_image():
    import cv2
    import ultralytics

    p = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--skip-clip", action="store_true")
    ap.add_argument("--skip-streamlit", action="store_true")
    a = ap.parse_args()

    # ---- system ------------------------------------------------------------
    rec("python", "INFO", f"{platform.python_version()} ({platform.architecture()[0]}) {sys.executable}")
    rec("os", "INFO", f"{platform.system()} {platform.release()} / cpu arch {platform.machine()}")
    try:
        from vigil.device import detect_hardware, resolve_device, select_profile
        from vigil.config import CONFIG_DIR, load_profile
    except Exception as e:  # noqa: BLE001
        rec("vigil package", "FAIL", f"{type(e).__name__}: {e}")
        return 1
    hw = detect_hardware()
    rec("system RAM", "INFO", f"{(hw['ram_mb'] or 0) / 1024:.1f} GB")
    if hw["torch"] is None:
        rec("pytorch", "NOT_INSTALLED", hw["torch_error"] or "import failed")
    else:
        rec("pytorch", "INFO", f"{hw['torch']} (CUDA runtime {hw['cuda_runtime']})")
        if hw["cuda_available"]:
            rec("cuda", "PASS", f"{hw['gpu_name']}, {hw['gpu_vram_mb']:.0f} MB VRAM")
        elif hw["nvidia_smi"]:
            rec("cuda", "UNSUPPORTED", f"NVIDIA GPU '{hw['nvidia_smi']['name']}' present (driver {hw['nvidia_smi']['driver']}) "
                "but this PyTorch build cannot use it: install the CUDA wheel (see README)")
        else:
            rec("cuda", "NOT_RUN", "no NVIDIA GPU detected")
        if hw["apple_silicon"]:
            rec("mps", "PASS" if hw["mps_available"] else "UNSUPPORTED",
                f"built={hw['mps_built']} available={hw['mps_available']}; unified memory {(hw['ram_mb'] or 0) / 1024:.0f} GB")
        else:
            rec("mps", "NOT_RUN", "not an Apple Silicon machine")
    if hw["nvidia_smi"]:
        rec("gpu (nvidia-smi)", "INFO", f"{hw['nvidia_smi']['name']} / {hw['nvidia_smi']['vram_mb']:.0f} MB / driver {hw['nvidia_smi']['driver']}")

    prof_path, why = (Path(a.profile), "explicit --profile") if a.profile else select_profile(hw, CONFIG_DIR)
    rec("profile", "INFO", f"{prof_path.name} ({why})")
    try:
        prof = load_profile(str(prof_path))
        rec("profile validation", "PASS", f"{prof['name']}: valid")
    except Exception as e:  # noqa: BLE001
        rec("profile validation", "FAIL", str(e))
        return 1

    # ---- OpenCV ------------------------------------------------------------
    def t_opencv() -> None:
        try:
            import cv2
        except ImportError as e:
            rec("opencv", "NOT_INSTALLED", str(e))
            return
        import numpy as np

        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "t.mp4")
            w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            if not w.isOpened():
                rec("opencv video decode", "UNSUPPORTED", f"cv2 {cv2.__version__} cannot create mp4v test video")
                return
            for i in range(10):
                w.write(np.full((48, 64, 3), i * 20, np.uint8))
            w.release()
            cap = cv2.VideoCapture(p)
            n = 0
            while cap.read()[0]:
                n += 1
            cap.release()
            rec("opencv video decode", "PASS" if n == 10 else "FAIL", f"cv2 {cv2.__version__}: decoded {n}/10 frames")

    guarded("opencv", t_opencv)

    # ---- YOLO ----------------------------------------------------------------
    det_device = "cpu"

    def t_yolo() -> None:
        nonlocal det_device
        try:
            import ultralytics
        except ImportError as e:
            rec("ultralytics", "NOT_INSTALLED", str(e))
            return
        rec("ultralytics", "INFO", ultralytics.__version__)
        from vigil.tracker import Detector

        pref = prof["detector"].get("device", "auto")
        dev, why2 = resolve_device(pref if pref != "auto" else prof.get("torch_device", "auto"), hw)
        det_device = dev
        rec("detector device", "INFO", f"{dev} ({why2})")
        img = sample_image()
        if img is None:
            rec("yolo one-image detection", "NOT_RUN", "bundled sample image not found")
            return
        d = Detector(dict(prof["detector"]), dev)
        try:
            t0 = time.time()
            d.load()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            rec("yolo one-image detection", "MODEL_UNAVAILABLE" if "weights" in msg.lower() or "download" in msg.lower() else
                ("UNSUPPORTED" if dev != "cpu" else "FAIL"), msg[:300])
            return
        t1 = time.time()
        res = d.detect_only(img)
        people = [r for r in res if r[1] == "person"]
        rec("yolo one-image detection", "PASS" if people else "FAIL",
            f"{len(res)} objects, {len(people)} people on {dev}; load {d.load_seconds:.2f}s, 1 inference {time.time() - t1:.3f}s (bus.jpg)")
        ok = True
        try:  # ByteTrack per-camera tracker must be constructible
            d.track("diag", img)
        except Exception as e:  # noqa: BLE001
            ok = False
            rec("bytetrack", "NOT_INSTALLED" if "lap" in str(e) else "FAIL", f"{e} (pip install lapx)")
        if ok:
            rec("bytetrack", "PASS", "BYTETracker constructed and updated")

    guarded("yolo", t_yolo)

    # ---- Ollama + VLM --------------------------------------------------------
    def t_vlm() -> None:
        from vigil.vlm import VLMVerifier

        v = VLMVerifier(prof["vlm"])
        info = v.check()
        if not info["reachable"]:
            import shutil

            rec("ollama connectivity", "FAIL" if shutil.which("ollama") else "NOT_INSTALLED",
                ("installed but not running. " if shutil.which("ollama") else "the `ollama` command was not found. ") + info["error"])
            rec("vlm one-image inference", "NOT_RUN", "Ollama not reachable")
            return
        rec("ollama connectivity", "PASS", f"version {info['version']} at {prof['vlm']['base_url']}; "
            f"structured-output schema {'supported' if info['schema_supported'] else 'NOT supported (falls back to format=json)'}")
        if a.skip_vlm:
            rec("vlm one-image inference", "NOT_RUN", "--skip-vlm")
            return
        if not info["model_available"]:
            rec("vlm one-image inference", "MODEL_UNAVAILABLE", info["error"])
            return
        import numpy as np

        from vigil.composite import build_composite
        from vigil.types import Frame, Trigger, VLMRequest

        img = sample_image()
        if img is None:
            rec("vlm one-image inference", "NOT_RUN", "bundled sample image not found")
            return
        frames = [Frame("diag", i, float(i), img) for i in range(4)]
        comp, n, times = build_composite(frames, prof["vlm"]["composite_max_side"])
        tg = Trigger("tailgating", "tailgating", "diag", 3.0, (1, 2), 0.5, reason="diagnostic probe (sample photo, not a real event)")
        req = VLMRequest("diag", tg, comp, img, n, time.time())
        res = v.verify(req)
        if res.ok:
            rec("vlm one-image inference", "PASS", f"{prof['vlm']['model']}: valid JSON, category='{res.category}', "
                f"conf={res.vlm_confidence:.2f}, {res.t_end - res.t_start:.1f}s (first call includes model load)")
            rec("vlm json validity", "PASS", "response passed schema validation")
            ps = v.client.ps()
            for m in ps:
                size, vram = m.get("size", 0), m.get("size_vram", 0)
                rec("ollama placement", "INFO", f"{m.get('name')}: {size / 1e9:.2f} GB loaded, {100 * vram / size if size else 0:.0f}% on GPU")
        else:
            rec("vlm one-image inference", "FAIL", res.error)
            rec("vlm json validity", "FAIL", res.error)

    guarded("ollama", t_vlm)

    # ---- CLIP ---------------------------------------------------------------
    emb_holder: Dict[str, Any] = {}

    def t_clip() -> None:
        if a.skip_clip:
            rec("clip", "NOT_RUN", "--skip-clip")
            return
        try:
            import open_clip  # noqa: F401
        except ImportError as e:
            rec("clip", "NOT_INSTALLED", str(e))
            return
        from vigil.embeddings import EmbedderUnavailable, make_embedder

        e = make_embedder(prof)
        try:
            t0 = time.time()
            e.load()
        except EmbedderUnavailable as ex:
            rec("clip checkpoint load", "MODEL_UNAVAILABLE", str(ex)[:300])
            rec("clip image embedding", "NOT_RUN", "checkpoint not loaded")
            rec("clip text embedding", "NOT_RUN", "checkpoint not loaded")
            return
        rec("clip checkpoint load", "PASS", f"{e.tag} on {e.device} in {time.time() - t0:.1f}s (dim {e.dim})")
        img = sample_image()
        import numpy as np

        iv = e.embed_images([img if img is not None else np.zeros((64, 64, 3), np.uint8)])
        rec("clip image embedding", "PASS" if iv.shape[1] == e.dim else "FAIL", f"shape {iv.shape}, norm {float(np.linalg.norm(iv[0])):.3f}")
        tv = e.embed_texts(["a person standing next to a bus", "an empty street"])
        sims = (iv @ tv.T)[0]
        rec("clip text embedding", "PASS" if tv.shape[1] == e.dim else "FAIL",
            f"shape {tv.shape}; image-text cosine: bus={sims[0]:.3f}, empty street={sims[1]:.3f}")
        emb_holder["e"] = e

    guarded("clip", t_clip)

    # ---- SQLite / Chroma -------------------------------------------------------
    def t_sqlite() -> None:
        from vigil.storage import EventStore
        from vigil.types import EventRecord

        with tempfile.TemporaryDirectory() as d:
            s = EventStore(Path(d) / "t.db")
            ok = s.insert_event(EventRecord("diag", time.time(), "cam1", "person down", 0.5, "x", "", "verified"))
            got = s.get_event("diag")
            mode = s.conn().execute("PRAGMA journal_mode").fetchone()[0]
            s.close()
            rec("sqlite read/write", "PASS" if ok and got else "FAIL", f"sqlite {__import__('sqlite3').sqlite_version}, journal_mode={mode}")

    guarded("sqlite", t_sqlite)

    def t_chroma() -> None:
        try:
            import chromadb
        except ImportError as e:
            rec("chromadb", "NOT_INSTALLED", str(e))
            return
        import numpy as np

        from vigil.embeddings import VectorIndex

        with tempfile.TemporaryDirectory() as d:
            idx = VectorIndex(d, "diag", 8)
            idx.upsert("e1", np.arange(8, dtype=np.float32) + 1, None, {"camera_id": "c"})
            del idx
            idx2 = VectorIndex(d, "diag", 8)  # reopen: proves persistence
            s = idx2.scores("image", np.arange(8, dtype=np.float32) + 1, ["e1"])
            rec("chromadb persistence", "PASS" if abs(s.get("e1", 0) - 1.0) < 1e-4 else "FAIL", f"chromadb {chromadb.__version__}: vector survived reopen")

    guarded("chromadb", t_chroma)

    # ---- Streamlit -----------------------------------------------------------
    def t_streamlit() -> None:
        try:
            import streamlit
        except ImportError as e:
            rec("streamlit", "NOT_INSTALLED", str(e))
            return
        rec("streamlit import", "PASS", streamlit.__version__)
        try:  # st.table / st.dataframe need a pyarrow that matches the installed NumPy
            import pyarrow

            rec("pyarrow (dashboard tables)", "PASS", pyarrow.__version__)
        except Exception as e:  # noqa: BLE001
            rec("pyarrow (dashboard tables)", "FAIL", f"{type(e).__name__}: {e}. Fix: pip install pyarrow==17.0.0")
        if a.skip_streamlit:
            rec("streamlit startup", "NOT_RUN", "--skip-streamlit")
            return
        import socket
        import urllib.request

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "dashboard.py"), "--server.port", str(port),
                                 "--server.headless", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=str(ROOT))
        ok = False
        try:
            for _ in range(40):
                time.sleep(0.5)
                try:
                    if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).read() == b"ok":
                        ok = True
                        break
                except Exception:  # noqa: BLE001
                    pass
        finally:
            proc.terminate()
            try:
                proc.wait(5)
            except Exception:  # noqa: BLE001
                proc.kill()
        rec("streamlit startup", "PASS" if ok else "FAIL", f"health endpoint on :{port} {'ok' if ok else 'did not answer'}")

    guarded("streamlit", t_streamlit)

    counts: Dict[str, int] = {}
    for r in RESULTS:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nSUMMARY:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("NOT_RUN / MODEL_UNAVAILABLE / UNSUPPORTED items are NOT passes; fix or run them on the target machine.")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"hardware": hw, "profile": str(prof_path), "results": RESULTS,
                                            "time": time.time()}, indent=1, default=str))
    return 1 if counts.get("FAIL") or counts.get("NOT_INSTALLED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
