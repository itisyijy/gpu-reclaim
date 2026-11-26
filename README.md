# gpu-reclaim

**Predicting Tomorrow's Idle GPUs** — Alibaba MaaS trace v2026(cluster-trace-v2026-maas)로 LLM 추론 GPU 회수를 위한 유휴 예측.
ITM627 term project.

오늘까지의 기록으로 **내일 거의 쓰이지 않을 app을 골라 GPU를 회수**한다. 많이 회수할수록 GPU가 절약되지만, 잘못 회수하면 cold start가 생긴다.

## 결론

> **"오늘 idle이면 회수"라는 단순 규칙(Baseline 1)이 이미 강하다. 모델이 회수량을 늘리지는 못한다. 대신 같은 개수를 회수할 때 언제나 덜 위험한 app을 고른다.**

홀드아웃 5주(day 147–180) 기준, 95% 구간은 app 부트스트랩 2,000회.

| 숫자 | 의미 |
| ---: | --- |
| 90% | B1이 내일 idle GPU-h 중 회수하는 비율 (recall 0.899) |
| **−12.6%** | B1과 같은 개수를 회수할 때 최종 모델의 잘못 회수 GPU-h (−78k [−160k, −16k], 유의) |
| +0.009 | 같은 개수에서 F1 (0.873 → 0.882, 유의) |
| **−48%** | B1 개수의 90%만 회수할 때, 같은 개수의 단순 규칙 대비 잘못 회수 GPU-h (회수량은 −1.6%, 유의) |
| 주당 70만 GPU-h | B1 규칙의 주당 회수량 (GPU 약 4,200장을 일주일 비우는 양) |

- 최종 모델: **LightGBM + XGBoost 확률 평균**. 운영 규칙은 **매일 B1과 같은 개수**를 점수 상위에서 회수하는 것이다. threshold도 정답 라벨도 필요 없다.
- 모델의 이득은 '규모' 변수에서 나온다. 모델은 큰 app을 덜 회수한다. B1이 잘못 회수한 app-day는 평균 423 GPU-h이고, 맞게 회수한 app-day는 106 GPU-h다.
- 한계
  - test 1주(140–146)에서는 모든 모델이 B1과 거의 같다. 5분 표가 이 주에만 있어서, 실제 cold start 시뮬레이션으로는 이득을 확인할 수 없다.
  - 홀드아웃 이득은 일부 주(161–167)와 상위 app에 몰려 있다.

자세한 수치·그림·해석은 [통합 보고서](reports/slide_report/README.md)([PDF](reports/slide_report/slide_report.pdf))에 있다.

## 문제 정의

| 항목 | 정의 |
| --- | --- |
| 단위 | app × day — 535,846행, app 20,661개, day 0–180 |
| 결정 시점 | day t가 끝날 때, day t+1에 회수할 app을 정함 |
| target | 내일 idle: 내일 GPU가 일하지 않은 시간 비율 ≥ 0.9 (20초 관측 기준) |
| 가중치 | 내일 GPU-h(`gpu_hours_next`), 학습과 평가 모두 |
| 입력 | 23개 변수, 6묶음: 오늘 idle·이력·오늘 사용량·규모·메타·달력 |
| 분할 | train 0–132 · val 133–139 · test 140–146 · **홀드아웃 147–180** (튜닝에는 쓰지 않음. 최종 모델·운영 정책 선택에는 v4의 홀드아웃 결과를 씀) |
| 평가 | walk-forward: 6주(140–180) 각각을 그 주 직전 데이터로 다시 학습(`day ≤ 주 시작 − 2`) |
| 지표 | 회수 GPU-h(클수록 좋음)와 잘못 회수 건수·GPU-h(작을수록 좋음)를 함께 봄 |

**왜 instance가 아니라 app 단위인가:** instance 1.34억 개 중 95.9%가 하루만 존재한다. offline instance는 다음날에도 남아 있는 비율이 4.4%뿐이다. 그래서 instance 단위의 "내일 유휴" 라벨은 사실상 정의되지 않는다(2026-11-12 측정).

비교 모델: Baseline 1(오늘 idle 비율), Baseline 2(sm_util 하위 N), Logistic Regression, SVM, Random Forest, LightGBM, XGBoost. 하이퍼파라미터는 4주 rolling CV와 Optuna로 골랐다(v3).

## 결과물

| 무엇 | 위치 |
| --- | --- |
| 최종 노트북 (과제 제출용: 문제·데이터·방법·결과·논의) | [`notebooks/2026_maas_analysis_final.executed.ipynb`](notebooks/2026_maas_analysis_final.executed.ipynb) |
| 통합 결과 보고서 (그림 8장, 원본값 CSV 20개) | [`reports/slide_report/`](reports/slide_report/README.md) — README, `slide_report.pdf`(A4), `figures/`, `data/` |
| 발표 슬라이드 (7분, 영어) | [Figma Slides](https://www.figma.com/slides/iTUmA7npK5zYqtgVprPKi3), [`reports/slides/gpu_reclaim_final.pptx`](reports/slides/gpu_reclaim_final.pptx)(그림 PPTX), `reports/slides/png/` |
| 야간 실험 요약 (v3·v4) | [`reports/overnight_summary.md`](reports/overnight_summary.md), 수치 모음 `reports/overnight_report.md` |

PPTX는 각 장이 Figma 화면 그림이라 글자를 고칠 수 없다. 글자를 고칠 수 있는 PPTX나 제출용 PDF는 Figma에서 File → Export로 내보낸다.

### 노트북 흐름

| 노트북 | 내용 |
| --- | --- |
| `2026_maas_analysis.ipynb`, `_full.ipynb` | 초기 분석 (CPU, 전체 61샤드, val PR-AUC로 모델 비교) |
| `2026_maas_analysis_v2.ipynb` | 팀원 노트북: app × day 전처리, 이력 변수, test·baseline. **저장소에 없음** (로컬 전용) |
| `2026_maas_analysis_v3_gpu.ipynb` | GPU 개선판: rolling CV + Optuna, 정책별 threshold, 홀드아웃 주 단위 재학습, 부트스트랩, SHAP, 5분 시뮬레이션 |
| `2026_maas_analysis_v4_robust.ipynb` | 견고성: threshold 정책 비교, seed 5개, 변수 묶음 더하기·빼기, 학습 기간 |
| `2026_maas_analysis_final.ipynb` | 최종 모델·결과를 과제 요구사항 순서로 정리 |

v3·v4·final 노트북은 `scripts/make_*_notebook.py`가 만든다. 노트북을 직접 고치지 말고 생성 스크립트를 고친다. `*.executed.ipynb`는 출력이 담긴 실행본이다.

## 저장소 구성

```
gpu-reclaim/
  src/gpu_reclaim/          paths.py(경로·분할 상수), db.py(DuckDB view: daily, week, app)
  scripts/
    verify_data.py            원본 데이터를 manifest와 대조
    build_app_day_v2.py       app × day 전처리 캐시 생성 (v2 노트북 필요)
    make_v3/v4/final_notebook.py   노트북 생성기
    run_v3.sh, run_overnight.sh, run_final.sh, run_v4_smoke.sh, run_sim_samecount.sh   독립 실행 래퍼
    sim_5min_samecount.py     5분 단위 cold start 시뮬레이션
    bench_cpu_gpu.py, gpu_smoke.py   GPU 환경 점검·속도 비교
    make_report.py, make_slide_report.py, report_to_pdf.py, make_slides_pptx.py   보고서·슬라이드
  notebooks/                위 노트북과 실행본
  reports/                  보고서·슬라이드
  pyproject.toml, environment.yml, environment.lock.yml
```

git에 올리지 않는 것 (`.gitignore`):

| 경로 | 내용 |
| --- | --- |
| `upstream/` | alibaba/clusterdata 얕은 클론. `cluster-trace-v2026-maas/data/`에 원본 데이터, `output/`에 공식 재현 결과 |
| `data/raw` | `upstream/cluster-trace-v2026-maas/data`를 가리키는 링크 |
| `data/processed/` | 전처리 캐시 `app_day_v2_max181.parquet` |
| `downloads/` | 원본 zip 3개 (15GB) |
| `outputs/` | 실험 결과: `v3/`, `v4/`(점수 `scores.parquet`), `final/` |
| `env/`, `env-gpu/` | conda prefix 환경 (2.4GB, 9GB) |
| `.cache/` | conda·pip 캐시, TMPDIR, DuckDB spill |

## 환경

루트 디스크(`/`)가 거의 차 있으므로 **캐시·임시파일도 전부 SSD로** 보낸다.

```bash
export CONDA_PKGS_DIRS=/ssd/jylee/gpu-reclaim/.cache/conda-pkgs \
       PIP_CACHE_DIR=/ssd/jylee/gpu-reclaim/.cache/pip \
       TMPDIR=/ssd/jylee/gpu-reclaim/.cache/tmp
```

| 환경 | 용도 | 구성 | Jupyter 커널 |
| --- | --- | --- | --- |
| `env/` | 데이터 검증, 공식 재현, 초기 분석 | Python 3.12, duckdb 1.5.5·numpy 2.2.6·matplotlib 3.10.9(공식 고정), pandas, polars, lightgbm·xgboost(CPU), scikit-learn | `Python (gpu-reclaim)` |
| `env-gpu/` | v3·v4·final, 보고서 | Python 3.11, RAPIDS 24.08(cuML·cuDF), xgboost 2.1(CUDA 11.8), LightGBM 4.6(CUDA), scikit-learn 1.5, Optuna, SHAP | `Python (gpu-reclaim GPU)` |

- 패키지 추가는 `conda install -c conda-forge <pkg>`로 한다. 서버가 glibc 2.27이라 최신 pip 바이너리 휠은 설치되지 않는 경우가 많다.
- 드라이버 470이라 CUDA는 11.8까지 쓸 수 있다. GPU 0번은 다른 사용자가 쓰고 있다. 노트북은 남는 GPU를 자동으로 고르고, `GPU_IDS=1`처럼 직접 지정할 수도 있다.
- GPU가 CPU보다 XGBoost는 47–56배, LightGBM은 64배 빠르다(`outputs/bench_cpu_gpu.csv`). cuML RF는 가중치를 받지 않아서 RF는 sklearn을 쓴다.
- 프로젝트 패키지는 editable로 설치되어 있어 어디서든 `import gpu_reclaim`이 된다.

## 데이터

| 표 | 범위 | 행 수 | 형식 | 쓴 곳 |
| --- | --- | ---: | --- | --- |
| `instance_daily/` | day 0–181 (172 없음) | 145,083,117 | parquet 61개, 9.8 GiB | 라벨·입력 변수 전부 |
| `instance_week_5min/` | day 140–146 | 140,906,827 | parquet 21개, 4.5 GiB | 5분 단위 cold start 확인 |
| `app_metrics.csv` | day 140–146, 앱 4개 | 7,401 | CSV, `\N` = NULL | 참고 |

- 스키마는 `upstream/cluster-trace-v2026-maas/docs/schema.md`에 있다.
- daily는 instance-day 하나가 **여러 세그먼트 행**으로 나뉠 수 있다(gpu_hours는 합산). parquet 샤드는 날짜별로 나뉘어 있지 않다.

```python
from gpu_reclaim.db import connect
con = connect(threads=8, memory_limit="48GB")   # view: daily, week, app
con.sql("SELECT day, sum(gpu_hours) FROM daily GROUP BY 1 ORDER BY 1").df()
```

## 재현

오래 걸리는 실행은 세션이 끊겨도 돌도록 `setsid nohup`으로 띄운다. 래퍼는 축소 실행이 성공해야 전체 실행으로 넘어가고, 끝나면 로그에 `DONE`을 남긴다.

**1. 데이터 받기·검증** (`env`)

```bash
cd /ssd/jylee/gpu-reclaim/downloads
for z in instance_daily.zip instance_week_5min.zip app_metrics.csv.zip; do
  curl -fL -C - -o $z https://tre-clusterdata.oss-cn-hangzhou.aliyuncs.com/cluster-trace-v2026-maas/data/$z; done
cd ../upstream/cluster-trace-v2026-maas/data && for z in ../../../downloads/*.zip; do unzip -n $z; done
```

```bash
cd /ssd/jylee/gpu-reclaim && env/bin/python -I scripts/verify_data.py
```

```bash
cd /ssd/jylee/gpu-reclaim/upstream/cluster-trace-v2026-maas && ../../env/bin/python scripts/run_all.py
```

**2. 전처리 캐시** (`env-gpu`, `notebooks/2026_maas_analysis_v2.ipynb` 필요) → `data/processed/app_day_v2_max181.parquet`

```bash
env-gpu/bin/python scripts/build_app_day_v2.py 181
```

**3. 튜닝(v3) → 견고성(v4) → 요약 보고서**: v3 재실행 후 v4와 `reports/overnight_report.md`까지 이어서 실행한다. 로그는 `outputs/overnight.log`.

```bash
setsid nohup bash scripts/run_v3.sh > /dev/null 2>&1 < /dev/null &
```

```bash
setsid nohup bash scripts/run_overnight.sh > /dev/null 2>&1 < /dev/null &
```

**4. 최종 노트북** (약 30분, 로그 `outputs/final/run.log`)

```bash
env-gpu/bin/python scripts/make_final_notebook.py
```

```bash
setsid nohup bash scripts/run_final.sh > /dev/null 2>&1 < /dev/null &
```

**5. 보고서·슬라이드**

```bash
env-gpu/bin/python scripts/make_slide_report.py
```

```bash
env-gpu/bin/python scripts/report_to_pdf.py
```

```bash
PYTHONPATH=<python-pptx 설치 경로> env-gpu/bin/python scripts/make_slides_pptx.py
```

## 출처

- 데이터: Alibaba cluster-trace-v2026-maas, https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-maas
- Zheng et al., *Model-as-a-Service in the Wild*, ACM SIGOPS ATC 2026 (trace README의 인용 정보)
- 도구: LightGBM, XGBoost, Optuna, SHAP, scikit-learn, RAPIDS cuML, DuckDB
