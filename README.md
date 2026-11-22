# gpu-reclaim

Predicting Idle LLM Inference Instances for GPU Reclamation — Alibaba MaaS trace (cluster-trace-v2026-maas).

목표: (instance_id, day) 단위로 **내일(t+1) 유휴 여부**를 예측해 cold start를 늘리지 않고 회수 가능한 GPU-hours를 최대화.
라벨: `Σzero_samples / Σsamples ≥ τ` (instance-day의 세그먼트 합, τ는 val에서 조정).
분할: Train 0–132 · Val 133–139 · Test 140–146 · Hold-out 147–181 (`src/gpu_reclaim/paths.py`의 `SPLITS`).

## 구성

```
gpu-reclaim/
  upstream/                 alibaba/clusterdata 얕은 클론 (cluster-trace-v2026-maas만 sparse checkout)
    cluster-trace-v2026-maas/data/   원본 데이터 (공식 스크립트가 쓰는 경로 그대로)
    cluster-trace-v2026-maas/output/ 공식 재현 결과 (Table 2–4, Fig 2–18)
  downloads/                원본 zip 3개 (검증 완료, 필요 없으면 삭제 가능 15GB)
  data/raw  ->  upstream/cluster-trace-v2026-maas/data
  data/interim, data/processed      가공 산출물
  env/                      conda prefix 환경 (Python 3.12, conda-forge)
  .cache/                   conda/pip 캐시, TMPDIR, DuckDB spill
  src/gpu_reclaim/          paths.py(경로·분할 상수), db.py(DuckDB view)
  scripts/verify_data.py    manifest 대조 무결성 검사
  notebooks/  outputs/
```

루트 디스크(`/`)가 거의 차 있으므로 **캐시·임시파일도 전부 SSD로** 보낸다.

## 환경

```bash
conda activate /ssd/jylee/gpu-reclaim/env
export CONDA_PKGS_DIRS=/ssd/jylee/gpu-reclaim/.cache/conda-pkgs \
       PIP_CACHE_DIR=/ssd/jylee/gpu-reclaim/.cache/pip \
       TMPDIR=/ssd/jylee/gpu-reclaim/.cache/tmp
```

- 패키지 추가는 `conda install -c conda-forge <pkg>` (서버가 glibc 2.27이라 최신 pip 바이너리 휠은 설치 실패하는 경우가 많음)
- `environment.yml`: 요청한 패키지 목록 / `environment.lock.yml`: 설치된 전체 버전
- 공식 재현 고정 버전: duckdb 1.5.5, numpy 2.2.6, matplotlib 3.10.9
- Jupyter 커널: `Python (gpu-reclaim)`
- 프로젝트 패키지는 editable 설치됨 → 어디서든 `import gpu_reclaim`

## 데이터

| 표 | 범위 | 행 수 | 형식 |
| --- | --- | ---: | --- |
| `instance_daily/` | day 0–181 (172 없음) | 145,083,117 | parquet 61개, 9.8 GiB |
| `instance_week_5min/` | day 140–146 | 140,906,827 | parquet 21개, 4.5 GiB |
| `app_metrics.csv` | day 140–146, 앱 4개 | 7,401 | CSV, `\N` = NULL |

스키마는 `upstream/cluster-trace-v2026-maas/docs/schema.md`.
주의: daily는 instance-day 하나에 **세그먼트가 여러 행**일 수 있음(gpu_hours는 합산). parquet 샤드는 날짜 파티션이 아님.

**인스턴스 수명 (2026-11-12 측정)** — t+1 라벨 설계 전에 반드시 고려
- instance-day 144,702,269개, 고유 instance 134,313,560개 → 95.9%가 하루만 존재 (GPU-hours로는 35%)
- 다음날(t+1)에도 같은 instance가 있는 instance-day 비율: offline 4.4% (GPU-h 28%), online 72.3% (GPU-h 93.8%)
- 행의 96%가 offline/fused. 즉 "내일 유휴?" 라벨은 사실상 online 인스턴스에서만 정의됨

```python
from gpu_reclaim.db import connect
con = connect(threads=8, memory_limit="48GB")   # view: daily, week, app
con.sql("SELECT day, sum(gpu_hours) FROM daily GROUP BY 1 ORDER BY 1").df()
```

다시 받기 (필요할 때):

```bash
cd /ssd/jylee/gpu-reclaim/downloads
for z in instance_daily.zip instance_week_5min.zip app_metrics.csv.zip; do
  curl -fL -C - -o $z https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data/$z; done
cd ../upstream/cluster-trace-v2026-maas/data && for z in ../../../downloads/*.zip; do unzip -n $z; done
```

검증 / 공식 재현:

```bash
cd /ssd/jylee/gpu-reclaim && python -I scripts/verify_data.py
cd upstream/cluster-trace-v2026-maas && python scripts/run_all.py   # 결과: output/
```
