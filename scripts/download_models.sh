#!/usr/bin/env bash
# Fetch every model Vigil needs while you still have internet.  After this the
# system runs fully offline.  Safe to re-run (everything is cached / skipped).
#
#   bash scripts/download_models.sh [vlm_model ...]
#
# VLM models default to the one in the active profile (scripts/benchmark_models.py
# may have rewritten it).  Pass names to pull specific ones, e.g.:
#   bash scripts/download_models.sh moondream qwen2.5vl:3b
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
export HF_HOME="${HF_HOME:-$PWD/data/cache/hf}"

echo "== 1/3 YOLOv8n weights =="
if [ -f models/yolov8n.pt ]; then
  echo "models/yolov8n.pt present"
else
  mkdir -p models
  "$PY" - <<'PYEOF'
from pathlib import Path
import shutil
from ultralytics import YOLO
YOLO("yolov8n.pt")                     # downloads into the CWD / Ultralytics cache
src = Path("yolov8n.pt")
if src.exists():
    shutil.move(str(src), "models/yolov8n.pt")
print("saved models/yolov8n.pt")
PYEOF
fi

echo "== 2/3 CLIP ViT-B/32 (open_clip, openai weights) =="
"$PY" - <<'PYEOF'
import open_clip
m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
open_clip.get_tokenizer("ViT-B-32")
print("CLIP cached")
PYEOF

echo "== 3/3 Ollama VLM =="
if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama is not installed. Install it from https://ollama.com/download , start it, then re-run this script."
  echo "(Vigil still runs without it: events are stored as 'unverified'.)"
  exit 0
fi
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("$("$PY" - <<'PYEOF'
from vigil.config import load_profile
print(load_profile(None)["vlm"]["model"])
PYEOF
)")
fi
for m in "${MODELS[@]}"; do
  echo "pulling $m"
  ollama pull "$m"
done
echo "done.  Next: python scripts/diagnose.py"
