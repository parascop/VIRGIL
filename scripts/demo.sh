#!/usr/bin/env bash
# One-command demo (macOS / Linux): make synthetic clips if missing, run the
# pipeline once over data/clips at 4x speed, then open the dashboard.
#   bash scripts/demo.sh [--no-vlm]
# Clips are SYNTHETIC (see scripts/make_demo_clips.py).  Without Ollama running,
# events are stored as 'unverified'.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.11}"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1 ANONYMIZED_TELEMETRY=False
[ -f data/clips/demo_cam1.mp4 ] || "$PY" scripts/make_demo_clips.py
"$PY" scripts/diagnose.py --skip-streamlit || true
"$PY" -m vigil.pipeline --speed 4 --loop "$@" &
PIPE=$!
trap 'kill -INT $PIPE 2>/dev/null || true; wait 2>/dev/null || true' INT TERM EXIT
sleep 5
"$PY" -m streamlit run app/dashboard.py --server.headless false
