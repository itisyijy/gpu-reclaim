#!/bin/bash
# 5분 단위 시뮬레이션(같은 개수 정책, v4 점수) 독립 실행. 로그: outputs/final/sim.log
ROOT=/ssd/jylee/gpu-reclaim
LOG=$ROOT/outputs/final/sim.log
mkdir -p $ROOT/outputs/final
export TMPDIR=$ROOT/.cache/tmp OMP_NUM_THREADS=8
echo "$(date '+%F %T') 시작" > $LOG
$ROOT/env-gpu/bin/python $ROOT/scripts/sim_5min_samecount.py >> $LOG 2>&1
echo "$(date '+%F %T') 종료 exit=$?" >> $LOG
echo DONE >> $LOG
