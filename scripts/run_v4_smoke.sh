#!/bin/bash
# v4 축소 실행만 (코드 점검용). setsid nohup bash scripts/run_v4_smoke.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/ssd/jylee/gpu-reclaim
cd "$ROOT/notebooks"
export TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$ROOT/outputs/v4"
LOG="$ROOT/outputs/v4/run.log"
echo "$(date '+%F %T') v4 축소 실행 시작 (GPU_IDS=${GPU_IDS:-자동})" >> "$LOG"
V4_SMOKE=1 /usr/bin/time -f "elapsed %E peakRSS %MKB" "$ROOT/env-gpu/bin/jupyter" nbconvert --to notebook --execute \
    --ExecutePreprocessor.kernel_name=gpu-reclaim-gpu --ExecutePreprocessor.timeout=-1 \
    --output "$ROOT/.cache/tmp/v4_smoke.ipynb" 2026_maas_analysis_v4_robust.ipynb > "$ROOT/outputs/v4/smoke.log" 2>&1
echo "$(date '+%F %T') v4 축소 실행 종료 exit=$?" >> "$LOG"
echo "SMOKE_DONE" >> "$LOG"
