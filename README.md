# Vigil

Local, open-source system that watches video, flags candidate incidents, has a vision-language model (VLM) look at each one, stores everything, and lets you search it in plain English from a dashboard. Nothing leaves your machine unless you point it at a remote VLM yourself.

Categories: **person down, fighting, tailgating, crowd incident, suspicious activity.**

> **Status, honestly.** The pipeline, storage, query parser, dashboard and 140+ unit/integration tests run and pass on a Linux CPU machine. The Ollama VLM path, real CLIP weights, NVIDIA GPUs, Apple MPS and Windows were **not available where this was built**, so those are covered only by mocks or are *unverified* (details at the end). Evaluation uses synthetic clips; see `REPORT.md` for exactly what was and was not measured.

## How it works

```
 camera / file / RTSP                                     ┌─────────────────────────┐
        │                                                 │ Streamlit dashboard     │
  ┌─────▼──────┐  frames   ┌──────────────┐ tracks        │ live · chat search ·    │
  │ ingestion  ├──────────►│ YOLOv8n +    ├────────┐      │ metrics · upload        │
  │ (1 thread/ │ bounded   │ ByteTrack    │        │      └──────▲──────────▲───────┘
  │  camera)   │ queue     │ (1 worker)   │   ┌────▼─────┐       │          │ SQLite (events)
  └─────┬──────┘           └──────────────┘   │heuristics│       │          │ Chroma (CLIP vectors)
        │ ring buffer                         └────┬─────┘       │ status.json, live JPEGs
        │ (last ~4 s)             candidate        │             │
        └──────► 2x2 composite ◄───────────────────┘       ┌─────┴───────┐
                       │                                   │   storage   │
                 ┌─────▼──────┐  verified / rejected /     │ + CLIP index│◄── persist worker
                 │ Ollama VLM ├──── unverified ───────────►│ (SQLite is  │
                 │ (1 worker, │  (VLM down => unverified)  │  the source │
                 │ bounded    │                            │  of truth)  │
                 │ priority q)│                            └─────────────┘
                 └────────────┘
```

1. **Ingestion** reads each camera at `target_fps` (10 by default). Queues are bounded: live sources drop the oldest frame, files block (nothing is lost offline). RTSP/webcam sources reconnect with back-off.
2. **Detection and tracking:** one shared YOLOv8n, one ByteTrack instance per camera.
3. **Heuristics** turn tracks into *candidates* (never final events): person down (aspect-ratio flip + drop + stillness), fighting (proximity + sustained relative motion), tailgating (two crossings of an entry line within `window_s`), crowd (count threshold or sudden dispersal), suspicious (loitering or entering a restricted zone). All thresholds are in `configs/defaults.yaml` / `cameras.yaml`, none are hard-coded. A per-(camera, trigger, tracks, zone) cooldown stops repeated alerts.
4. **VLM verification:** a 2x2 chronological composite of the last frames goes to a local Ollama VLM with a JSON-schema constrained answer (temperature 0, one retry on invalid output). Result: `verified` (VLM agrees; it may *re-classify*), `rejected` (VLM says nothing happened), or `unverified` (VLM unreachable or output unusable). **Rejected and unverified events are stored too** and can be shown with a filter.
5. **Storage:** SQLite (WAL) holds every event. ChromaDB holds CLIP embeddings of the keyframe and of the summary text. If Chroma is lost or corrupted, rebuild it from SQLite (below).
6. **Search:** your sentence is parsed into a structured filter (category, camera, time range, visual phrase) by a local LLM *constrained to a JSON schema and validated*, with a deterministic rule-based fallback. The LLM never writes SQL; filters run as parameterised queries. The visual phrase is ranked with `0.6 x image + 0.4 x summary` similarity (min-max normalised over the candidates). The dashboard shows "How I interpreted this" for every query.

`confidence_score` blends the heuristic score and the VLM confidence. It is a **ranking heuristic, not a calibrated probability.**

## Requirements

- Python **3.11**, 64-bit (3.12 untested).
- About 3 GB free disk (PyTorch, CLIP, YOLO) plus the VLM (moondream ~1.7 GB; others 2-4 GB).
- [Ollama](https://ollama.com/download) for the VLM (optional: without it you get heuristics-only events marked `unverified`).
- `ffmpeg` on PATH is optional: it lets the dashboard re-encode clips for browser playback; OpenCV is the fallback.

## Setup: macOS (Apple Silicon M4)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1          # default macOS wheels include MPS
pip install -r requirements-macos.txt
# install Ollama from https://ollama.com/download (menu-bar app), then:
bash scripts/download_models.sh                        # YOLO (bundled), CLIP, VLM; needs internet once
python scripts/diagnose.py --profile configs/profile_mac_m4.yaml
python scripts/benchmark_models.py --pull --write-profile   # picks the VLM for THIS machine
bash scripts/run_mac.sh                                # pipeline + dashboard at http://localhost:8501
```

## Setup: Windows 10/11 + NVIDIA (RTX 2050 / GTX 1650)

```powershell
py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-windows.txt
# install Ollama for Windows from https://ollama.com/download, then (Git Bash / WSL, or run the commands by hand):
bash scripts/download_models.sh
python scripts\diagnose.py --profile configs\profile_rtx2050.yaml   # or profile_gtx1650.yaml
python scripts\benchmark_models.py --pull --write-profile
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1
```

Use the CUDA wheel that matches your driver (<https://pytorch.org/get-started/locally/>). With 4 GB of VRAM, YOLO and the VLM compete: the profiles budget ~3.4-3.6 GB, keep CLIP on the CPU, and `benchmark_models.py` measures peak VRAM before choosing a VLM. If VRAM is tight set `detector.device: cpu` in the profile.

`scripts/benchmark_models.py` is also what removes the "unverified model choice" caveat: until you run it the profiles use the default `moondream`.

## Run

| What | Command |
|---|---|
| Everything (Mac) | `bash scripts/run_mac.sh` |
| Everything (Windows) | `powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1` |
| 1-command demo with synthetic clips | `bash scripts/demo.sh` / `scripts\demo.ps1` |
| Pipeline only | `python -m vigil.pipeline [--profile P] [--speed 1] [--loop] [--no-vlm]` |
| Dashboard only | `streamlit run app/dashboard.py` |
| Tests | `python -m pytest -q` |
| Diagnostics | `python scripts/diagnose.py [--json out.json]` |
| Evaluation | `python eval/run_eval.py --profile configs/profile_<x>.yaml && python eval/make_report.py` |

Pipeline flags: `--speed N` file playback speed (1 = real time, 0 = as fast as possible), `--loop`, `--no-vlm` (ablation baseline), `--only cam1 cam2`, `--source cam1=path.mp4`, `--exit-when-done`. Ctrl-C shuts down cleanly (queues drained or discarded, SQLite closed, status marked stopped).

**Cameras** live in `configs/cameras.yaml`: a file path, webcam index or `rtsp://` URL. Keep credentials out of the file: use `source: "${CAM3_RTSP_URL}"` and set the environment variable. Zones and entry lines use **normalised 0-1 coordinates**, so they survive resolution changes. Per-camera `thresholds:` override the defaults.

**Dashboard.** *Live*: sidebar feed with camera health and latest events. *Chat search*: ask things like "person down on cam1 yesterday evening" or "people close together near the door". *Metrics*: latency percentiles, queue depths, memory, verdict counts. *Upload video*: queues a file for the running pipeline (use the same camera template to inherit its zones).

Retrieval modes: hybrid (default), image-only, summary-only. Without a running pipeline the dashboard loads CLIP itself; if CLIP weights are missing it says so and falls back to filters plus keyword matching.

A screenshot of the chat tab is in `eval/results/dashboard_chat_MOCK_VLM_STUB.png`; it was taken against **mock VLM output** (summaries start with "[MOCK VLM]") and is only a layout check.

## Offline use

After `download_models.sh` nothing needs the internet. Set `HF_HUB_OFFLINE=1` to be strict. Ollama also runs offline once the model is pulled. Streamlit's usage telemetry is off in `.streamlit/config.toml`; Chroma telemetry is disabled in code.

## Rebuild or repair the vector index

SQLite is the source of truth. If `data/chroma` is deleted or corrupted:

```bash
python -m vigil.embeddings --rebuild      # drop and re-embed every event keyframe + summary
python -m vigil.embeddings --repair       # only events whose index_status is not 'indexed'
```

Events whose keyframe file is missing are marked `index_status = failed` with the reason, not silently skipped.

## Optional: VLM on another machine in your LAN

Set `vlm.base_url` in the profile (or `VIGIL_OLLAMA_URL`). On the GPU host start Ollama with `OLLAMA_HOST=0.0.0.0:11434`.
**Security:** Ollama has **no authentication**. Exposing it lets anyone on that network run models on your hardware and read prompts and images. Only do this on a trusted, firewalled LAN, restrict the port to the Vigil machine's IP, never port-forward it to the internet, and put a VPN or authenticating reverse proxy in front if you need more. Video frames are sent over the network in clear text; plain `http` is used. If the remote VLM disappears, Vigil keeps running and stores events as `unverified`.

## Troubleshooting

- **Everything is `unverified`:** Ollama is not running or the model is not pulled. `ollama list`, then `python scripts/diagnose.py`.
- **`MODEL_UNAVAILABLE` / `NOT_INSTALLED` in diagnose:** the named model or binary is missing; the line says what to install.
- **`No module named lap`:** install `lapx` (it is in requirements).
- **Chroma `capture() takes 1 positional argument`:** wrong `posthog` version; `pip install posthog==3.7.4`.
- **Dashboard shows `pyarrow requires NumPy 2.0 or newer`:** `pip install pyarrow==17.0.0 pandas==2.2.3` (pinned in requirements).
- **Windows `ZoneInfoNotFoundError`:** `pip install tzdata`.
- **CUDA out of memory:** set `detector.device: cpu`, lower `detector.imgsz`, or choose a smaller VLM (`benchmark_models.py`).
- **macOS op not implemented for MPS:** `run_mac.sh` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`; the detector also falls back to CPU if MPS inference fails.
- **Video does not play in the dashboard:** install `ffmpeg`; the dashboard re-encodes to H.264.
- **No events on your own footage:** thresholds are untuned defaults. Start with `--no-vlm` and `logging.debug: true` and tune `heuristics:` per camera.

## Layout

```
configs/    defaults.yaml, cameras.yaml, profile_{rtx2050,gtx1650,mac_m4}.yaml
vigil/      types config device ingest tracker heuristics composite vlm embeddings storage query pipeline media metrics
app/        dashboard.py            scripts/   diagnose, benchmark_models, make_demo_clips, mock_ollama, run_*, demo.*
eval/       manifest.csv queries.csv run_eval.py make_report.py results/        tests/   pytest suite
```

## Licences and data

Ultralytics YOLOv8 is **AGPL-3.0**. If you distribute Vigil or offer it over a network, AGPL obligations apply to the combined work; for closed-source or commercial use obtain an Ultralytics enterprise licence or swap the detector. CLIP (open_clip, OpenAI weights), ChromaDB, Streamlit and Ollama have their own licences; check each VLM's licence before use. Faces and people on video are personal data: check your local law and signage requirements before deploying on real cameras.

## Known limitations and unverified items

- **Verified here (Linux x86-64 CPU, Python 3.11):** unit and integration tests, the pipeline on synthetic clips with real YOLOv8n + ByteTrack, SQLite/Chroma storage, query parsing, the dashboard under Streamlit's AppTest.
- **Not verified here, so treat as unverified:** a real Ollama VLM (tested against a mock HTTP server), real CLIP weights (code path tested with random weights), NVIDIA GPUs, Apple MPS, Windows, FP16, the model choices in the profiles, and live RTSP cameras. The commands to verify each are in `REPORT.md` section 8.
- All bundled clips are synthetic. Heuristics are 2-D box rules: they confuse fights with hugging and falls with sitting; the VLM stage helps but its benefit is unmeasured.
- Heuristic thresholds are starting values, not tuned for any real scene.
- Small VLMs (1-4 B) hallucinate and mis-describe; read `verified` as "a small model agreed", not as proof.
- Not a safety-critical system: do not rely on it as the only means of detecting emergencies.
