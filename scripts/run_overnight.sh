#!/bin/bash
# 야간 연쇄 실행: v3 전체 실행이 끝나기를 기다림 -> v4 전체 실행 -> 보고서 생성.
#   setsid nohup bash scripts/run_overnight.sh > /dev/null 2>&1 < /dev/null &
# 진행 상황: outputs/overnight.log, 보고서: reports/overnight_report.md
set -u
ROOT=/ssd/jylee/gpu-reclaim
cd "$ROOT/notebooks"
export TMPDIR="$ROOT/.cache/tmp"
LOG="$ROOT/outputs/overnight.log"
mkdir -p "$ROOT/outputs/v4" "$ROOT/reports"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "연쇄 실행 시작 (pid $$) — v3 끝나기 대기"
until grep -q '^DONE' "$ROOT/outputs/v3/run.log" 2>/dev/null; do sleep 60; done
say "v3 끝: $(grep -E '종료|생략' "$ROOT/outputs/v3/run.log" | tail -1)"

if [ ! -f "$ROOT/outputs/v3/best_params.json" ]; then
    say "v3 best_params.json 없음 -> v4 생략"
else
    sleep 30   # v3 커널이 GPU 메모리를 내려놓을 시간
    say "v4 전체 실행 시작"
    /usr/bin/time -f "elapsed %E peakRSS %MKB" "$ROOT/env-gpu/bin/jupyter" nbconvert --to notebook --execute --allow-errors \
        --ExecutePreprocessor.kernel_name=gpu-reclaim-gpu --ExecutePreprocessor.timeout=-1 \
        --output 2026_maas_analysis_v4_robust.executed.ipynb 2026_maas_analysis_v4_robust.ipynb \
        > "$ROOT/outputs/v4/full.log" 2>&1
    f=$?
    n_err=$(grep -c '"output_type": "error"' 2026_maas_analysis_v4_robust.executed.ipynb 2>/dev/null || echo "?")
    say "v4 전체 실행 종료 exit=$f, 오류 셀 수=$n_err"
fi

"$ROOT/env-gpu/bin/python" "$ROOT/scripts/make_report.py" >> "$LOG" 2>&1
say "보고서 생성 exit=$?"
say "DONE"
