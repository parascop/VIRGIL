#!/usr/bin/env bash
# Start Vigil (pipeline + dashboard) on macOS.  Ctrl-C stops both cleanly.
#   bash scripts/run_mac.sh [--speed N] [--loop] [--no-vlm] [--profile configs/profile_mac_m4.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.11}"
[ -x .venv/bin/python ] && PY=".venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1        # unsupported MPS ops fall back to CPU instead of crashing
export ANONYMIZED_TELEMETRY=False
PORT="${VIGIL_PORT:-8501}"
ARGS=("$@")
[[ " ${ARGS[*]:-} " == *" --profile "* ]] || ARGS+=(--profile configs/profile_mac_m4.yaml)

for i in "${!ARGS[@]}"; do [ "${ARGS[$i]}" = "--profile" ] && export VIGIL_PROFILE="${ARGS[$((i+1))]}"; done  # dashboard uses the same profile
"$PY" -m vigil.pipeline "${ARGS[@]}" &
PIPE=$!
cleanup() {
  trap - INT TERM EXIT
  kill -INT "$PIPE" 2>/dev/null || true
  [ -n "${UI:-}" ] && kill "$UI" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT
"$PY" -m streamlit run app/dashboard.py --server.port "$PORT" --server.headless true &
UI=$!
echo "Dashboard: http://localhost:$PORT   (pipeline pid $PIPE)"
wait "$PIPE" "$UI"
