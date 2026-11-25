#!/bin/bash
# v3 GPU 노트북: 축소 실행으로 점검 -> 전체 실행. 세션이 끊겨도 돌도록 setsid nohup으로 띄운다.
#   setsid nohup bash scripts/run_v3.sh > /dev/null 2>&1 < /dev/null &
# 진행 상황: outputs/v3/run.log (단계), outputs/v3/smoke.log, outputs/v3/full.log (nbconvert 출력)
set -u
ROOT=/ssd/jylee/gpu-reclaim
cd "$ROOT/notebooks"
export TMPDIR="$ROOT/.cache/tmp"
mkdir -p "$ROOT/outputs/v3"
LOG="$ROOT/outputs/v3/run.log"
NB=2026_maas_analysis_v3_gpu.ipynb
NBC=("$ROOT/env-gpu/bin/jupyter" nbconvert --to notebook --execute
     --ExecutePreprocessor.kernel_name=gpu-reclaim-gpu --ExecutePreprocessor.timeout=-1)

echo "$(date '+%F %T') 축소 실행 시작 (pid $$)" >> "$LOG"
V3_SMOKE=1 "${NBC[@]}" --output "$ROOT/.cache/tmp/v3_smoke.ipynb" "$NB" > "$ROOT/outputs/v3/smoke.log" 2>&1
s=$?
echo "$(date '+%F %T') 축소 실행 종료 exit=$s" >> "$LOG"

# 축소 실행이 실패하면 전체 실행은 하지 않는다 (smoke.log에서 원인 확인)
if [ "$s" -ne 0 ]; then echo "$(date '+%F %T') 축소 실행 실패 -> 전체 실행 생략" >> "$LOG"; echo "DONE" >> "$LOG"; exit 1; fi
echo "$(date '+%F %T') 전체 실행 시작" >> "$LOG"
/usr/bin/time -f "elapsed %E peakRSS %MKB" "${NBC[@]}" --allow-errors \
    --output 2026_maas_analysis_v3_gpu.executed.ipynb "$NB" > "$ROOT/outputs/v3/full.log" 2>&1
f=$?
n_err=$(grep -c '"output_type": "error"' 2026_maas_analysis_v3_gpu.executed.ipynb 2>/dev/null || echo "?")
echo "$(date '+%F %T') 전체 실행 종료 exit=$f, 오류 셀 수=$n_err" >> "$LOG"
echo "DONE" >> "$LOG"
