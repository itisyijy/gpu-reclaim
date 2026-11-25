#!/bin/bash
# 최종 노트북: 축소 실행 -> 성공하면 전체 실행. 로그: outputs/final/run.log
set -u
ROOT=/ssd/jylee/gpu-reclaim
cd "$ROOT/notebooks"
export TMPDIR="$ROOT/.cache/tmp" GPU_IDS=1
LOG="$ROOT/outputs/final/run.log"
mkdir -p "$ROOT/outputs/final"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
NB=2026_maas_analysis_final
run() {  # $1 = 출력 이름
    "$ROOT/env-gpu/bin/jupyter" nbconvert --to notebook --execute $2 \
        --ExecutePreprocessor.kernel_name=gpu-reclaim-gpu --ExecutePreprocessor.timeout=-1 \
        --output "$1" "$NB.ipynb" >> "$ROOT/outputs/final/nbconvert.log" 2>&1
}
if [ "${SKIP_SMOKE:-0}" = "1" ]; then s=0; say "축소 실행 생략 (이미 통과)"; else
say "축소 실행 시작"
FINAL_SMOKE=1 run "$NB.smoke.ipynb" ""; s=$?
fi
say "축소 실행 종료 exit=$s"
if [ "$s" -ne 0 ]; then say "축소 실행 실패 -> 전체 실행 생략"; echo DONE >> "$LOG"; exit 1; fi
echo SMOKE_DONE >> "$LOG"
[ "${SMOKE_ONLY:-0}" = "1" ] && { echo DONE >> "$LOG"; exit 0; }
say "전체 실행 시작"
run "$NB.executed.ipynb" "--allow-errors"; f=$?
n=$(grep -c '"output_type": "error"' "$NB.executed.ipynb" 2>/dev/null); say "전체 실행 종료 exit=$f, 오류 셀 수=${n:-0}"
echo DONE >> "$LOG"
