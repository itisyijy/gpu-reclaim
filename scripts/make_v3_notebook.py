"""팀원 노트북 v2를 바탕으로 GPU 개선판 v3 노트북을 만든다 (출력 없는 .ipynb 생성).

v2의 전처리 결과(scripts/build_app_day_v2.py가 만든 parquet)와 인코더 코드를 그대로 쓰고,
평가·검증·튜닝을 바꾼다. 사용: env-gpu/bin/python scripts/make_v3_notebook.py
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "notebooks" / "2026_maas_analysis_v2.ipynb"
OUT = ROOT / "notebooks" / "2026_maas_analysis_v3_gpu.ipynb"

v2_code = ["".join(c["source"]) for c in json.load(open(V2))["cells"] if c["cell_type"] == "code"]

# --- v2 인코더 셀: app_id를 보조 열로, 가중치 복사, 끝의 동작 확인 코드 제거 ---
enc = next(s for s in v2_code if "def build_linear_encoder" in s)
old_aux = 'AUX_COLS = ["day", "gpu_hours_next", "is_mostly_idle"]'
assert old_aux in enc
enc = enc.replace(old_aux, 'AUX_COLS = ["day", "gpu_hours_next", "is_mostly_idle", "app_id"]  # v3: app_id 추가 (부트스트랩·시뮬레이션용)')
old_ret = 'return X, data[target], data["gpu_hours_next"]'
assert old_ret in enc
enc = enc.replace(old_ret, 'return X, data[target], data["gpu_hours_next"].copy()')
cut = enc.index("# 동작 확인 (테스트용")
enc = enc[:cut].rstrip() + "\n"

# --- v2 XGBoost 셀의 단조 제약 정의 ---
xgb_cell = next(s for s in v2_code if "XGB_MONOTONE = {" in s)
mono = xgb_cell[xgb_cell.index("XGB_MONOTONE = {"):]
mono = mono[:mono.index("}") + 1]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Predicting Tomorrow's Idle GPUs — v3 (GPU 개선판)

팀원 v2 노트북의 **전처리·변수·인코더는 그대로** 두고, 실험 설계를 다음처럼 바꾼다.

| v2에서 부족했던 점 | v3에서 바꾼 것 |
| --- | --- |
| 성공 기준("같은 cold-start 위험에서 더 많이 회수")을 직접 비교하지 않음 | val에서 **Baseline 1과 위험을 맞춘 threshold**를 골라 test에 적용, 회수량-위험 곡선 |
| cold start를 GPU-hours로만 셈 | **잘못 회수 건수**(= cold start 사건 수)를 함께 보고 |
| test를 보고 설계를 바꿈 (단조 제약, 변수 묶음) | 단조 제약 여부를 **여러 주 검증(rolling-origin CV)** 안에서 결정, test는 마지막에 한 번 |
| val 1주로 하이퍼파라미터 선택, 작은 그리드 | **4주 rolling CV + Optuna**, GPU로 XGBoost·LightGBM 학습 |
| test 1주(21,342행)만 평가, 신뢰구간 없음 | **홀드아웃 147–180 주 단위 재학습 평가** + **app 단위 부트스트랩 신뢰구간** |
| SVM은 2만 행 표본, val 959행에서 threshold | **cuML SVC(GPU)**, train 표본 키우고 val은 전부 사용 |
| `instance_week_5min` 미사용 | test 주간 **5분 단위 시뮬레이션**: 실제 cold start 건수와 첫 활동 전까지 회수한 GPU-h |
| 해석 근거 없음 | **GPU TreeSHAP** 변수 기여도, workload별 결과 |

실행 환경: `env-gpu` 커널 (RAPIDS 24.08 cuML, XGBoost 2.1 CUDA, LightGBM 4.6 CUDA).
남는 GPU(사용률·메모리가 낮은 GPU)를 실행 시점에 찾아 전부 쓴다.
""")

md("## 0. 설정 — 남는 GPU 찾기")
code(r"""
# CUDA 라이브러리(cupy/cuml/xgboost)를 import하기 전에 실행해야 한다.
import os
import subprocess
import time


def gpu_status():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            rows.append(tuple(map(int, parts)))
    return rows


def spare_gpus(n_samples=6, max_util=30, max_mem_mb=4000):
    # 잠깐 동안 여러 번 재서 최대 사용률·메모리가 기준 이하인 GPU만 고른다 (다른 사람이 쓰는 GPU 제외)
    peak = {}
    for _ in range(n_samples):
        for i, u, m in gpu_status():
            pu, pm = peak.get(i, (0, 0))
            peak[i] = (max(pu, u), max(pm, m))
        time.sleep(0.5)
    return [i for i, (u, m) in sorted(peak.items()) if u <= max_util and m <= max_mem_mb], peak


GPUS, peak = spare_gpus()
print("GPU 상태 (최대 사용률 %, 메모리 MiB):", peak)
assert GPUS, "남는 GPU가 없음"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPUS))
N_GPU = len(GPUS)
DEVICES = list(range(N_GPU))   # CUDA_VISIBLE_DEVICES 안에서의 번호
print(f"사용할 GPU: 물리 번호 {GPUS} -> 장치 {DEVICES}")
""")

code(r"""
import json
import queue
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path.cwd().resolve().parent          # notebooks/ 의 상위 = 프로젝트 루트
OUT = ROOT / "outputs" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 0

# 실행 규모 (시간이 부족하면 줄인다)
N_TRIALS = 40            # Optuna 시도 횟수 (모델당)
OPTUNA_TIMEOUT = 2400    # 모델당 최대 초
N_BOOT = 2000            # 부트스트랩 반복
SVM_TRAIN_N = None       # cuML SVC 학습 행 수 (None = train 전체; GPU에서 39만 행도 수 초)
SMOKE = os.environ.get("V3_SMOKE") == "1"   # 코드 점검용 축소 실행
if SMOKE:
    N_TRIALS, N_BOOT = 2, 50

import cupy as cp
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.metrics import average_precision_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
print("cupy", cp.__version__, "| xgboost", xgb.__version__, "| lightgbm", lgb.__version__, "| optuna", optuna.__version__)
for d in DEVICES:
    print(f"  장치 {d}:", cp.cuda.runtime.getDeviceProperties(d)["name"].decode())
""")

md("""
## 1. 데이터 — v2 전처리 결과 불러오기

`scripts/build_app_day_v2.py`가 v2 노트북의 전처리 셀(정제 → app×day 집계 → 라벨 → 과거 기록·파생 변수 → day+1 타겟)을
**0–181일 전체**에 그대로 실행해 저장한 표다. v2와 다른 점은 `MAX_DAY`(147 → 181)와 보조 열 `app_id`뿐이다.
""")
code(r"""
DATA = ROOT / "data" / "processed" / "app_day_v2_max181.parquet"
if not DATA.exists():   # 없으면 v2 전처리를 다시 실행 (CPU, 약 7분)
    subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "build_app_day_v2.py"), "181"], check=True)

df = pd.read_parquet(DATA)
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=range(7))   # parquet 왕복에서 int로 풀림 -> 복원
assert "app_id" in df.columns

SPLITS = {"train": (0, 132), "val": (133, 139), "test": (140, 146), "holdout": (147, 180)}


def in_days(data, days):
    return data["day"].between(*days)


def part(name, data=None):
    data = df if data is None else data
    return data[in_days(data, SPLITS[name])]


sizes = {k: len(part(k)) for k in SPLITS}
print(f"전체 {len(df):,} 행 | " + " | ".join(f"{k} {v:,}" for k, v in sizes.items()))
# v2와 같은 데이터인지 확인 (v2: train+val 414,663 / test 21,342)
assert sizes["train"] + sizes["val"] == 414_663 and sizes["test"] == 21_342
print("idle(1) 비율:", {k: round(part(k)["is_mostly_idle_next"].mean() * 100, 2) for k in SPLITS})
""")

md("## 2. v2 인코더 (그대로) + 공통 함수")
code(enc)

code(r"""
# v2 XGBoost 셀의 단조 제약 정의 (v3에서는 쓸지 말지를 CV로 정한다)
""" + mono + r"""

LINEAR_FEATURES = None


def norm_w(w):
    # 학습용 가중치: 평균 1로 정규화 (gpu_hours_next 값이 수천까지 커서 min_child_weight 등의 의미가 흔들리지 않게)
    w = np.asarray(w, dtype="float64")
    return w / w.mean()


def dense32(a):
    import scipy.sparse as sp
    return (a.toarray() if sp.issparse(a) else np.asarray(a)).astype("float32")


# ---------------------------------------------------------------------------
# 평가: 점수(score)와 0/1 판정(pred)을 받아 GPU-hours 기준 + 건수 기준 지표를 낸다
# ---------------------------------------------------------------------------
def evaluate(y, score, pred, w):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int); w = np.asarray(w, dtype="float64")
    tp, fp, fn = (pred == 1) & (y == 1), (pred == 1) & (y == 0), (pred == 0) & (y == 1)
    TP, FP, FN = w[tp].sum(), w[fp].sum(), w[fn].sum()
    return {
        "AP": average_precision_score(y, score, sample_weight=w),
        "F1": 2 * TP / max(2 * TP + FP + FN, 1e-12),
        "precision": TP / max(TP + FP, 1e-12),
        "recall": TP / max(TP + FN, 1e-12),
        "회수 GPU-h": TP, "잘못 회수 GPU-h": FP, "놓친 idle GPU-h": FN,
        "잘못 회수 건수": int(fp.sum()), "회수 건수": int((pred == 1).sum()),
    }


def threshold_curve(y, score, w):
    # 점수 내림차순으로 threshold를 낮춰 가며 누적 회수/잘못 회수 (같은 점수는 한 번에)
    y = np.asarray(y).astype(int); w = np.asarray(w, dtype="float64"); s = np.asarray(score, dtype="float64")
    o = np.argsort(-s, kind="stable"); s, y, w = s[o], y[o], w[o]
    tp, fp, fpn = np.cumsum(w * y), np.cumsum(w * (1 - y)), np.cumsum(1 - y)
    last = np.r_[s[1:] != s[:-1], True]
    P = (w * y).sum()
    c = pd.DataFrame({"thr": s[last], "회수 GPU-h": tp[last], "잘못 회수 GPU-h": fp[last], "잘못 회수 건수": fpn[last]})
    c["F1"] = 2 * c["회수 GPU-h"] / (c["회수 GPU-h"] + c["잘못 회수 GPU-h"] + P)
    return c


POLICIES = ["F1 최대", "위험 맞춤 (GPU-h)", "위험 맞춤 (건수)"]


def pick_threshold(y, score, w, policy, budget=None):
    # val에서 threshold를 고른다.
    #  - F1 최대: v2와 같은 기준 (gpu_hours_next 가중 F1)
    #  - 위험 맞춤: 잘못 회수 비율이 budget(= 같은 기간 Baseline 1의 비율) 이하인 threshold 중 회수량 최대
    #    GPU-h: 잘못 회수 GPU-h / 전체 다음날 GPU-h,  건수: 잘못 회수 건수 / 전체 행 수
    c = threshold_curve(y, score, w)
    if policy == "F1 최대":
        return float(c.loc[c["F1"].idxmax(), "thr"])
    if policy == "위험 맞춤 (GPU-h)":
        ok = c["잘못 회수 GPU-h"] / np.sum(w) <= budget
    else:
        ok = c["잘못 회수 건수"] / len(y) <= budget
    if not ok.any():
        return float("inf")
    return float(c.loc[ok[ok].index[-1], "thr"])   # 조건을 만족하는 가장 낮은 threshold = 회수량 최대


def risk_budget(y, base_pred, w):
    # Baseline 1의 잘못 회수 비율 (GPU-h, 건수)
    y = np.asarray(y).astype(int); p = np.asarray(base_pred).astype(int); w = np.asarray(w, dtype="float64")
    fp = (p == 1) & (y == 0)
    return {"위험 맞춤 (GPU-h)": w[fp].sum() / w.sum(), "위험 맞춤 (건수)": fp.sum() / len(y)}
""")

md("""
### 2.1 Baseline (v2와 같음 + 조정판 하나)

- **Baseline 1**: 오늘 idle(`is_mostly_idle`)이면 내일도 idle. 점수는 오늘 `idle_ratio`
- **Baseline 1′ (조정)**: 오늘 `idle_ratio ≥ t`, t는 val에서 정책별로 선택 — 규칙 하나의 threshold만 바꿔도 얼마나 되는지
- **Baseline 2**: 오늘 `sm_util`이 가장 낮은 N개 (N = 학습 구간 idle 비율 × 행 수)
""")
code(r"""
def base1(data):
    return data["idle_ratio"].to_numpy(dtype="float64"), data["is_mostly_idle"].to_numpy().astype(int)


def base2(data, idle_rate):
    util = data["sm_util"].to_numpy(dtype="float64")
    util = np.where(np.isnan(util), np.nanmax(util) + 1, util)
    n = int(round(idle_rate * len(data)))
    pred = np.zeros(len(data), dtype=int)
    pred[np.argsort(util, kind="stable")[:n]] = 1
    return -util, pred, n
""")

md("""
## 3. 모델 학습 함수 (GPU)

| 모델 | 장치 | 비고 |
| --- | --- | --- |
| Logistic Regression | CPU (sklearn) | v2와 같은 파이프라인. 충분히 빠름 |
| SVM (RBF) | **GPU (cuML)** | **train 전체**로 학습 (v2는 2만 행 표본), 가중치 반영, 점수 = decision function |
| Random Forest | CPU (sklearn, 20코어) | cuML RF는 sample_weight를 받지 않아 val AP가 낮음(0.946 대 0.957). sklearn도 300트리 23초라 CPU 사용 |
| LightGBM | **GPU (CUDA)** | 범주형 그대로, early stopping |
| XGBoost | **GPU (CUDA)** | 범주형 그대로, early stopping, 단조 제약은 CV로 선택 |
""")
code(r"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Fitted:
    # 인코더 + 모델을 묶어서 score(data) -> 내일 idle 점수
    def __init__(self, enc, model, kind, best_iter=None):
        self.enc, self.model, self.kind, self.best_iter = enc, model, kind, best_iter

    def score(self, data):
        X, _, _ = split_xyw(data)
        Z = self.enc.transform(X)
        if self.kind == "svm":
            return np.asarray(self.model.decision_function(dense32(Z))).astype("float64")
        if self.kind == "rf":
            return np.asarray(self.model.predict_proba(dense32(Z)))[:, 1].astype("float64")
        return self.model.predict_proba(Z)[:, 1].astype("float64")


def fit_model(kind, params, tr, device=0, ev=None):
    # kind: lr / svm / rf / lgbm / xgb,  ev가 있으면 트리 모델은 early stopping에 사용
    X, y, w = split_xyw(tr)
    p = dict(params)
    if kind == "lr":
        enc = build_linear_encoder().fit(X)
        m = LogisticRegression(max_iter=2000, C=p["C"]).fit(enc.transform(X), y, sample_weight=norm_w(w))
        return Fitted(enc, m, kind)
    if kind == "svm":
        from cuml.svm import SVC as cuSVC
        enc = build_linear_encoder().fit(X)
        n = len(X) if SVM_TRAIN_N is None else min(SVM_TRAIN_N, len(X))
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), size=n, replace=False) if n < len(X) else np.arange(len(X))
        Z = dense32(enc.transform(X.iloc[idx]))
        with cp.cuda.Device(device):
            m = cuSVC(C=p["C"], gamma=p["gamma"], cache_size=4096)
            m.fit(Z, y.to_numpy()[idx].astype("float32"), sample_weight=norm_w(w.to_numpy()[idx]))
        return Fitted(enc, m, kind)
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        enc = build_linear_encoder().fit(X)
        m = RandomForestClassifier(n_estimators=p.get("n_estimators", 300), max_depth=p["max_depth"],
                                   max_features=p["max_features"], n_jobs=-1, random_state=RANDOM_STATE)
        m.fit(dense32(enc.transform(X)), y, sample_weight=norm_w(w))
        return Fitted(enc, m, kind)

    enc = build_tree_encoder().fit(X)
    Z = enc.transform(X)
    n_est = p.pop("n_estimators", 3000)
    if ev is not None:
        Xe, ye, we = split_xyw(ev)
        Ze = enc.transform(Xe)
    if kind == "lgbm":
        m = lgb.LGBMClassifier(device_type="cuda", gpu_device_id=device, n_estimators=n_est,
                               random_state=RANDOM_STATE, verbose=-1, bagging_freq=1, **p)
        kw = {}
        if ev is not None:
            kw = dict(eval_set=[(Ze, ye)], eval_sample_weight=[norm_w(we)], eval_metric="average_precision",
                      callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        m.fit(Z, y, sample_weight=norm_w(w), **kw)
        bi = m.best_iteration_ if ev is not None else n_est
        return Fitted(enc, m, kind, bi)
    if kind == "xgb":
        mono = p.pop("monotone", False)
        m = xgb.XGBClassifier(enable_categorical=True, tree_method="hist", device=f"cuda:{device}",
                              n_estimators=n_est, random_state=RANDOM_STATE, eval_metric="aucpr",
                              monotone_constraints=XGB_MONOTONE if mono else None,
                              early_stopping_rounds=100 if ev is not None else None, **p)
        kw = dict(eval_set=[(Ze, ye)], sample_weight_eval_set=[norm_w(we)]) if ev is not None else {}
        m.fit(Z, y, sample_weight=norm_w(w), verbose=False, **kw)
        bi = (m.best_iteration + 1) if ev is not None else n_est
        return Fitted(enc, m, kind, bi)
    raise ValueError(kind)


def wap(data, score):
    _, y, w = split_xyw(data)
    return average_precision_score(y, score, sample_weight=w)
""")

md("""
## 4. 하이퍼파라미터 선택 — 4주 rolling-origin CV + Optuna (GPU)

- 검증 주: 112–118, 119–125, 126–132, 133–139. 각 주보다 **앞의 날만** 학습 (시간 순서 유지)
- 점수: 네 주의 `gpu_hours_next` 가중 AP 평균 (v2와 같은 기준)
- 트리 모델은 early stopping으로 트리 수를 정하고, 최종 학습에는 네 주 평균 트리 수 × 1.1을 쓴다
- **XGBoost 단조 제약 사용 여부도 탐색 공간에 넣는다** (v2는 test 사례를 보고 정했음)
- Optuna의 병렬 작업 수 = 남는 GPU 수, 각 시도는 비어 있는 GPU 하나를 가져가서 쓴다
- LR·SVM·RF는 작은 그리드, 같은 4주 CV
""")
code(r"""
CV_WEEKS = [(112, 118), (119, 125), (126, 132), (133, 139)]
trval = df[df["day"] <= 139]


def cv_folds(data=trval):
    for a, b in CV_WEEKS:
        yield data[data["day"] < a], data[data["day"].between(a, b)]


FOLDS = list(cv_folds())
print("CV 폴드 (train 행 / val 행):", [(len(a), len(b)) for a, b in FOLDS])

dev_q = queue.Queue()
for d in DEVICES:
    dev_q.put(d)


def cv_objective(kind, space):
    def objective(trial):
        p = space(trial)
        d = dev_q.get()
        try:
            aps, iters = [], []
            for tr, va in FOLDS:
                f = fit_model(kind, p, tr, device=d, ev=va)
                aps.append(wap(va, f.score(va)))
                iters.append(f.best_iter)
            trial.set_user_attr("iters", iters)
            trial.set_user_attr("fold_AP", [round(a, 4) for a in aps])
            return float(np.mean(aps))
        finally:
            dev_q.put(d)
    return objective


def xgb_space(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": t.suggest_int("max_depth", 3, 10),
            "min_child_weight": t.suggest_float("min_child_weight", 1e-2, 50, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "monotone": t.suggest_categorical("monotone", [False, True])}


def lgbm_space(t):
    return {"learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 300, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 30, log=True)}


studies, BEST = {}, {}
for kind, space in [("xgb", xgb_space), ("lgbm", lgbm_space)]:
    t0 = time.time()
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    # v2 설정을 첫 시도로 넣어 둔다 (탐색이 v2보다 나빠질 수 없게)
    if kind == "xgb":
        st.enqueue_trial({"learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1.0, "subsample": 1.0,
                          "colsample_bytree": 1.0, "reg_lambda": 1.0, "monotone": True})
    else:
        st.enqueue_trial({"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "subsample": 1.0,
                          "colsample_bytree": 1.0, "reg_lambda": 1e-3})
    st.optimize(cv_objective(kind, space), n_trials=N_TRIALS, n_jobs=N_GPU, timeout=OPTUNA_TIMEOUT)
    studies[kind] = st
    bt = st.best_trial
    BEST[kind] = {**bt.params, "n_estimators": int(np.mean(bt.user_attrs["iters"]) * 1.1) + 1}
    v2_trial = st.trials[0]
    print(f"[{kind}] {len(st.trials)}회, {time.time() - t0:.0f}초 | CV AP 최고 {bt.value:.4f} (폴드 {bt.user_attrs['fold_AP']})"
          f" | v2 설정 {v2_trial.value:.4f} (폴드 {v2_trial.user_attrs['fold_AP']})")
    print("   선택:", BEST[kind])
""")

code(r"""
# XGBoost: 단조 제약을 쓴 시도와 안 쓴 시도의 CV AP 비교 (v2가 test를 보고 내린 결정을 CV로 다시 확인)
tr_df = studies["xgb"].trials_dataframe()
tr_df = tr_df[tr_df["state"] == "COMPLETE"]
display(tr_df.groupby("params_monotone")["value"].agg(["count", "max", "median"]).round(4))
for kind, st in studies.items():
    d = st.trials_dataframe()
    d.to_csv(OUT / f"optuna_{kind}.csv", index=False)
""")

code(r"""
# LR (4주 CV), SVM·RF (val 한 주) 그리드
def grid_cv(kind, grid, folds):
    grid = grid[:1] if SMOKE else grid
    rows = []
    for p in grid:
        aps = [wap(va, fit_model(kind, p, tr, device=DEVICES[0]).score(va)) for tr, va in folds]
        rows.append({**p, "AP": float(np.mean(aps)), "fold_AP": [round(a, 4) for a in aps]})
        print(kind, rows[-1], flush=True)
    r = pd.DataFrame(rows).sort_values("AP", ascending=False)
    return r


t0 = time.time()
lr_grid = grid_cv("lr", [{"C": c} for c in [0.01, 0.1, 1, 10]], FOLDS)
BEST["lr"] = {"C": float(lr_grid.iloc[0]["C"])}
print(f"LR {time.time() - t0:.0f}초 -> {BEST['lr']}")

t0 = time.time()
svm_grid = grid_cv("svm", [{"C": c, "gamma": g} for c in [0.1, 1, 10] for g in [0.01, 0.1]], FOLDS)
BEST["svm"] = {"C": float(svm_grid.iloc[0]["C"]), "gamma": float(svm_grid.iloc[0]["gamma"])}
print(f"SVM {time.time() - t0:.0f}초 -> {BEST['svm']}")

t0 = time.time()
rf_grid = grid_cv("rf", [{"max_depth": d, "max_features": f} for d in [16, None] for f in ["sqrt", 0.5]], FOLDS)
_md = rf_grid.iloc[0]["max_depth"]
BEST["rf"] = {"max_depth": None if pd.isna(_md) else int(_md), "max_features": rf_grid.iloc[0]["max_features"]}
print(f"RF {time.time() - t0:.0f}초 -> {BEST['rf']}")

V2_PARAMS = {"xgb": {"learning_rate": 0.05, "max_depth": 3, "n_estimators": 300, "monotone": True},
             "lgbm": {"learning_rate": 0.05, "n_estimators": 300, "num_leaves": 31}}
json.dump({k: {kk: (vv if not isinstance(vv, np.generic) else vv.item()) for kk, vv in v.items()} for k, v in BEST.items()},
          open(OUT / "best_params.json", "w"), ensure_ascii=False, indent=1)
BEST
""")

md("""
## 5. Test 주간(140–146) 평가

1. train(0–132)으로 학습 → val(133–139) 점수로 **정책별 threshold** 선택 (위험 예산 = val에서 Baseline 1의 잘못 회수 비율)
2. train+val(0–139)로 다시 학습 → test에 val threshold 적용

v2 설정(그리드 결과 그대로)의 XGBoost·LightGBM도 같은 방식으로 평가해서, **평가 방식 변화**와 **튜닝 효과**를 나눠 본다.
""")
code(r"""
MODELS = {
    "Logistic Regression": ("lr", BEST["lr"]),
    "SVM (cuML)": ("svm", BEST["svm"]),
    "Random Forest": ("rf", BEST["rf"]),
    "LightGBM v2 설정": ("lgbm", V2_PARAMS["lgbm"]),
    "LightGBM 튜닝": ("lgbm", BEST["lgbm"]),
    "XGBoost v2 설정": ("xgb", V2_PARAMS["xgb"]),
    "XGBoost 튜닝": ("xgb", BEST["xgb"]),
}
val_df, test_df = part("val"), part("test")
_, y_val, w_val = split_xyw(val_df)
_, y_test, w_test = split_xyw(test_df)
B1_val = base1(val_df)
BUDGET_VAL = risk_budget(y_val, B1_val[1], w_val)
print("val의 Baseline 1 잘못 회수 비율:", {k: f"{v * 100:.3f}%" for k, v in BUDGET_VAL.items()})

SCORES, THR = {}, {}
for name, (kind, p) in MODELS.items():
    t0 = time.time()
    s_val = fit_model(kind, p, part("train"), device=DEVICES[0]).score(val_df)
    THR[name] = {pol: pick_threshold(y_val, s_val, w_val, pol, BUDGET_VAL.get(pol)) for pol in POLICIES}
    s_test = fit_model(kind, p, trval, device=DEVICES[0]).score(test_df)
    SCORES[name] = {"val": s_val, "test": s_test}
    print(f"{name:24s} {time.time() - t0:5.0f}초 | val AP {wap(val_df, s_val):.4f} | test AP {wap(test_df, s_test):.4f}", flush=True)

# Baseline
idle_rate = trval["is_mostly_idle_next"].mean()
s1v = B1_val[0]
THR["Baseline 1′ (조정)"] = {pol: pick_threshold(y_val, s1v, w_val, pol, BUDGET_VAL.get(pol)) for pol in POLICIES}
""")

code(r"""
def results_for(data, scores_key, thr=THR):
    _, y, w = split_xyw(data)
    rows = []
    s1, p1 = base1(data)
    s2, p2, n2 = base2(data, idle_rate)
    rows.append({"방법": "Baseline 1: 오늘 idle", "정책": "-", **evaluate(y, s1, p1, w)})
    rows.append({"방법": f"Baseline 2: sm_util 하위 {n2}", "정책": "-", **evaluate(y, s2, p2, w)})
    for pol in POLICIES:
        t = thr["Baseline 1′ (조정)"][pol]
        rows.append({"방법": "Baseline 1′ (조정)", "정책": pol, "threshold": t, **evaluate(y, s1, s1 >= t, w)})
    for name in MODELS:
        s = SCORES[name][scores_key]
        for pol in POLICIES:
            t = thr[name][pol]
            rows.append({"방법": name, "정책": pol, "threshold": t, **evaluate(y, s, s >= t, w)})
    r = pd.DataFrame(rows)
    b = r.iloc[0]
    for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]:
        r[f"Δ{c} vs B1"] = r[c] - b[c]
    return r


test_res = results_for(test_df, "test")
test_res.to_csv(OUT / "test_results.csv", index=False)
show_cols = ["방법", "정책", "AP", "F1", "precision", "recall", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수",
             "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]
print(f"Test {len(test_df):,}행, idle 비율 {y_test.mean() * 100:.2f}%, 다음날 전체 {w_test.sum():,.0f} GPU-h")
for pol in POLICIES:
    print(f"\n=== 정책: {pol} (val에서 threshold 선택) ===")
    sub = test_res[(test_res["정책"] == pol) | (test_res["정책"] == "-")]
    display(sub[show_cols].round(4))
""")

md("""
### 5.1 회수량–위험 곡선 (test)

threshold를 처음부터 끝까지 움직였을 때 회수 GPU-h와 잘못 회수(GPU-h, 건수)의 관계. 점은 Baseline.
곡선이 Baseline 1 점보다 **위**에 있으면 "같은 위험에서 더 많이 회수"가 가능하다는 뜻이다.
아래 표의 '사후 맞춤'은 test에서 위험을 정확히 맞춘 비교라 실제 운영 성능이 아니라 **곡선끼리의 비교**용이다.
""")
code(r"""
plot_models = ["Logistic Regression", "SVM (cuML)", "Random Forest", "LightGBM 튜닝", "XGBoost 튜닝", "XGBoost v2 설정"]
s1t, p1t = base1(test_df)
s2t, p2t, n2t = base2(test_df, idle_rate)
b1 = evaluate(y_test, s1t, p1t, w_test)
b2 = evaluate(y_test, s2t, p2t, w_test)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
curves = {n: threshold_curve(y_test, SCORES[n]["test"], w_test) for n in plot_models}
curves["Baseline 1′ (idle_ratio 점수)"] = threshold_curve(y_test, s1t, w_test)
for ax, xcol, xlim in [(axes[0], "잘못 회수 GPU-h", b1["잘못 회수 GPU-h"] * 3), (axes[1], "잘못 회수 건수", b1["잘못 회수 건수"] * 3)]:
    for n, c in curves.items():
        ax.plot(c[xcol], c["회수 GPU-h"], label=n, lw=1.6, ls="--" if "Baseline" in n else "-")
    ax.scatter([b1[xcol]], [b1["회수 GPU-h"]], c="k", zorder=5, label="Baseline 1")
    ax.scatter([b2[xcol]], [b2["회수 GPU-h"]], c="r", marker="x", zorder=5, label="Baseline 2")
    ax.set_xlim(0, xlim); ax.set_xlabel(xcol); ax.set_ylabel("회수 GPU-h"); ax.grid(alpha=.3)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Test(140–146) 회수량–위험 곡선")
plt.rcParams["font.family"] = plt.rcParams.get("font.family")
fig.tight_layout(); fig.savefig(OUT / "test_tradeoff.png", dpi=130); plt.show()

rows = []
for n, c in curves.items():
    r = {"방법": n}
    for xcol in ["잘못 회수 GPU-h", "잘못 회수 건수"]:
        ok = c[c[xcol] <= b1[xcol]]
        r[f"B1과 같은 {xcol}에서 회수 GPU-h"] = ok["회수 GPU-h"].max() if len(ok) else 0.0
    rows.append(r)
matched = pd.DataFrame(rows)
for col in matched.columns[1:]:
    matched[col + " (Δ vs B1)"] = matched[col] - b1["회수 GPU-h"]
print(f"Baseline 1: 회수 {b1['회수 GPU-h']:,.0f} GPU-h, 잘못 회수 {b1['잘못 회수 GPU-h']:,.0f} GPU-h / {b1['잘못 회수 건수']}건")
display(matched.round(0))
matched.to_csv(OUT / "test_matched_risk.csv", index=False)
""")

md("""
### 5.2 부트스트랩 신뢰구간 (app 단위, test)

같은 app의 날들은 서로 독립이 아니므로 **app을 단위로** 복원 추출한다 (`N_BOOT`회).
각 모델 − Baseline 1의 차이에 대한 95% 구간. 0을 포함하지 않으면 우연으로 보기 어렵다.
""")
code(r"""
def boot_diff(data, preds, base_pred, n_boot=N_BOOT, seed=RANDOM_STATE):
    _, y, w = split_xyw(data)
    y = y.to_numpy().astype(int); w = w.to_numpy().astype("float64")
    codes, uniq = pd.factorize(data["app_id"].astype(str))
    n_app = len(uniq)

    def per_app(pred):
        pred = np.asarray(pred).astype(int)
        cols = np.stack([w * (pred == 1) * (y == 1), w * (pred == 1) * (y == 0), w * (pred == 0) * (y == 1),
                         ((pred == 1) & (y == 0)).astype(float)], axis=1)
        M = np.zeros((n_app, 4)); np.add.at(M, codes, cols); return M

    rng = np.random.default_rng(seed)
    # 각 반복에서 app별 뽑힌 횟수 (다항분포) -> 행렬곱 한 번으로 합계 (GPU)
    C = cp.asarray(rng.multinomial(n_app, np.full(n_app, 1 / n_app), size=n_boot).astype("float32"))
    MB = cp.asarray(per_app(base_pred), dtype=cp.float32)
    SB = cp.asnumpy(C @ MB)
    out = []
    for name, pred in preds.items():
        S = cp.asnumpy(C @ cp.asarray(per_app(pred), dtype=cp.float32))
        f1 = lambda T: 2 * T[:, 0] / (2 * T[:, 0] + T[:, 1] + T[:, 2])
        d = {"Δ회수 GPU-h": S[:, 0] - SB[:, 0], "Δ잘못 회수 GPU-h": S[:, 1] - SB[:, 1],
             "Δ잘못 회수 건수": S[:, 3] - SB[:, 3], "ΔF1": f1(S) - f1(SB)}
        r = {"방법": name}
        for k, v in d.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            r[k] = f"{np.mean(v):+,.4g} [{lo:+,.4g}, {hi:+,.4g}]"
        out.append(r)
    return pd.DataFrame(out)


for pol in ["F1 최대", "위험 맞춤 (GPU-h)"]:
    preds = {n: SCORES[n]["test"] >= THR[n][pol] for n in MODELS}
    preds["Baseline 1′ (조정)"] = s1t >= THR["Baseline 1′ (조정)"][pol]
    print(f"=== 정책: {pol} | 모델 − Baseline 1, 평균 [95% CI], app {test_df['app_id'].nunique():,}개 ===")
    bt = boot_diff(test_df, preds, p1t)
    display(bt)
    bt.to_csv(OUT / f"test_bootstrap_{pol.replace(' ', '_')}.csv", index=False)
""")

md("""
## 6. 홀드아웃(147–180) — 주 단위 재학습 평가

실제 운영처럼 매주 다시 학습한다. 주 `[a, b]`의 행을 예측할 때:
- 최종 모델: `day ≤ a−2`인 행으로 학습 (a−1일 행의 정답은 a일이라 주 시작 시점에 아직 모름)
- threshold: `day ≤ a−9` 행으로 학습한 모델을 `a−8 … a−2` 7일에 적용해 정책별로 선택 (위험 예산도 그 7일의 Baseline 1)
- 하이퍼파라미터는 4장에서 고른 값으로 고정
- 172일이 원본에 없어서 171일 행은 정답이 없고(제외), 173일 행은 어제 값이 결측
""")
code(r"""
HOLD_WEEKS = [(147, 153), (154, 160), (161, 167), (168, 174), (175, 180)]
WF_MODELS = {k: v for k, v in MODELS.items() if k != "XGBoost v2 설정" and k != "LightGBM v2 설정"}
hold_rows, hold_preds = [], {}
t_all = time.time()
for a, b in HOLD_WEEKS:
    wk = df[df["day"].between(a, b)]
    cal = df[df["day"].between(a - 8, a - 2)]
    _, y_c, w_c = split_xyw(cal)
    _, y_k, w_k = split_xyw(wk)
    bud = risk_budget(y_c, base1(cal)[1], w_c)
    s1k, p1k = base1(wk)
    s2k, p2k, _ = base2(wk, df[df["day"] <= a - 2]["is_mostly_idle_next"].mean())
    rec = {("Baseline 1: 오늘 idle", "-"): p1k, ("Baseline 2: sm_util 하위 N", "-"): p2k}
    sc_c = base1(cal)[0]
    for pol in POLICIES:
        rec[("Baseline 1′ (조정)", pol)] = s1k >= pick_threshold(y_c, sc_c, w_c, pol, bud.get(pol))
    scores_k = {"Baseline 1: 오늘 idle": s1k, "Baseline 2: sm_util 하위 N": s2k, "Baseline 1′ (조정)": s1k}
    for name, (kind, p) in WF_MODELS.items():
        s_c = fit_model(kind, p, df[df["day"] <= a - 9], device=DEVICES[0]).score(cal)
        s_k = fit_model(kind, p, df[df["day"] <= a - 2], device=DEVICES[0]).score(wk)
        scores_k[name] = s_k
        for pol in POLICIES:
            rec[(name, pol)] = s_k >= pick_threshold(y_c, s_c, w_c, pol, bud.get(pol))
    for (name, pol), pred in rec.items():
        hold_rows.append({"주": f"{a}–{b}", "방법": name, "정책": pol, **evaluate(y_k, scores_k[name], pred, w_k)})
        hold_preds.setdefault((name, pol), []).append(np.asarray(pred).astype(int))
    print(f"주 {a}–{b}: {len(wk):,}행 완료 ({time.time() - t_all:.0f}초 경과)", flush=True)

hold = pd.DataFrame(hold_rows)
hold.to_csv(OUT / "holdout_weekly.csv", index=False)
""")

code(r"""
# 다섯 주를 합친 결과 (점수 합계 기준) + 주별 F1
# 주별 예측을 이어 붙인 순서와 같게: 주마다 df 원래 순서 그대로 이어 붙인다
hold_df = pd.concat([df[df["day"].between(a, b)] for a, b in HOLD_WEEKS])
assert len(hold_df) == sum(len(p) for p in hold_preds[("Baseline 1: 오늘 idle", "-")])
_, y_h, w_h = split_xyw(hold_df)
pooled = []
for (name, pol), parts_ in hold_preds.items():
    pred = np.concatenate(parts_)
    r = evaluate(y_h, pred.astype(float), pred, w_h)   # 합친 결과는 0/1 판정 기준 (AP는 주별 표 참고)
    r.pop("AP")
    pooled.append({"방법": name, "정책": pol, **r})
pooled = pd.DataFrame(pooled)
b1h = pooled[pooled["방법"] == "Baseline 1: 오늘 idle"].iloc[0]
for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]:
    pooled[f"Δ{c} vs B1"] = pooled[c] - b1h[c]
pooled.to_csv(OUT / "holdout_pooled.csv", index=False)
print(f"홀드아웃 {len(hold_df):,}행 (147–180), idle 비율 {y_h.mean() * 100:.2f}%, 다음날 전체 {w_h.sum():,.0f} GPU-h")
for pol in POLICIES:
    print(f"\n=== 홀드아웃 합계 | 정책: {pol} ===")
    display(pooled[(pooled["정책"] == pol) | (pooled["정책"] == "-")].drop(columns="정책").round(4))

print("\n주별 가중 F1")
display(hold.pivot_table(index=["방법", "정책"], columns="주", values="F1").round(4))
print("주별 AP")
display(hold[hold["정책"].isin(["-", "F1 최대"])].pivot_table(index="방법", columns="주", values="AP").round(4))
""")

code(r"""
# 홀드아웃 부트스트랩 (app 단위, 다섯 주 합계) — 모델 − Baseline 1
p1h = np.concatenate(hold_preds[("Baseline 1: 오늘 idle", "-")])
for pol in ["F1 최대", "위험 맞춤 (GPU-h)"]:
    preds = {n: np.concatenate(hold_preds[(n, pol)]) for n in list(WF_MODELS) + ["Baseline 1′ (조정)"]}
    print(f"=== 홀드아웃 | 정책: {pol} | 모델 − Baseline 1, 평균 [95% CI], app {hold_df['app_id'].nunique():,}개 ===")
    bt = boot_diff(hold_df, preds, p1h)
    display(bt)
    bt.to_csv(OUT / f"holdout_bootstrap_{pol.replace(' ', '_')}.csv", index=False)
""")

md("## 7. workload별 결과 (test + 홀드아웃)")
code(r"""
def by_workload(data, preds, label):
    _, y, w = split_xyw(data)
    rows = []
    for wl in ["online", "offline"]:
        m = (data["workload"] == wl).to_numpy()
        for name, pred in preds.items():
            r = evaluate(y[m], np.asarray(pred, float)[m], np.asarray(pred)[m], w[m]); r.pop("AP")
            rows.append({"구간": label, "workload": wl, "행": int(m.sum()), "방법": name, **r})
    return pd.DataFrame(rows)


pol = "위험 맞춤 (GPU-h)"
pt = {"Baseline 1": p1t, **{n: SCORES[n]["test"] >= THR[n][pol] for n in ["Logistic Regression", "LightGBM 튜닝", "XGBoost 튜닝"]}}
ph = {"Baseline 1": p1h, **{n: np.concatenate(hold_preds[(n, pol)]) for n in ["Logistic Regression", "LightGBM 튜닝", "XGBoost 튜닝"]}}
wl_res = pd.concat([by_workload(test_df, pt, "test"), by_workload(hold_df, ph, "홀드아웃")])
wl_res.to_csv(OUT / "by_workload.csv", index=False)
print(f"정책: {pol}")
display(wl_res[["구간", "workload", "행", "방법", "F1", "precision", "recall", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]].round(4))
""")

md("""
## 8. 변수 기여도 — GPU TreeSHAP (XGBoost 튜닝, test)

XGBoost의 `pred_contribs`를 GPU에서 계산한다. 전체 평균 |SHAP|과, 상태 변화 그룹별 평균 SHAP(로그 오즈).
""")
code(r"""
final_xgb = fit_model("xgb", BEST["xgb"], trval, device=DEVICES[0])
Xt, _, _ = split_xyw(test_df)
Zt = final_xgb.enc.transform(Xt)
booster = final_xgb.model.get_booster()
booster.set_param({"device": f"cuda:{DEVICES[0]}"})
contrib = booster.predict(xgb.DMatrix(Zt, enable_categorical=True), pred_contribs=True)
shap_df = pd.DataFrame(contrib[:, :-1], columns=Zt.columns)

imp = shap_df.abs().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 6))
imp.iloc[::-1].plot.barh(ax=ax); ax.set_xlabel("mean |SHAP| (log-odds)"); ax.set_title("XGBoost 튜닝 — test 변수 기여도")
fig.tight_layout(); fig.savefig(OUT / "shap_importance.png", dpi=130); plt.show()

today = test_df["is_mostly_idle"].to_numpy(); yt = y_test.to_numpy()
grp = np.select([(today == 0) & (yt == 0), (today == 1) & (yt == 1), (today == 0) & (yt == 1), (today == 1) & (yt == 0)],
                ["busy→busy", "idle→idle", "busy→idle", "idle→busy"])
top = imp.index[:8]
display(shap_df[top].groupby(grp).mean().round(3))
imp.to_csv(OUT / "shap_importance.csv")
""")

md("""
## 9. 5분 단위 시뮬레이션 (test 주간, `instance_week_5min`)

일 단위 라벨 대신, 회수한 app이 **다음날 실제로 언제 처음 활동했는지**를 5분 표로 본다.

- 5분 창의 `sm_util`은 20초 관측의 평균이라 `sm_util > 0`이면 그 5분 안에 일한 관측이 하나 이상 있다는 뜻
  (일 단위 `zero_samples` = duty cycle 0인 20초 관측 수와 같은 기준)
- 가정: day t+1 시작에 app을 회수하고, **첫 활동 창에서 cold start 1건** 발생 후 그날 나머지는 다시 올려 둔다
- 회수 GPU-h = 첫 활동 전까지 app 인스턴스들의 `gpu_count × 5분` 합 (활동이 없으면 하루 전체)
- 5분 표가 140–146일뿐이라 오늘 = 140–145인 행(내일 = 141–146)만 쓴다
- 인스턴스 → app 대응은 일 단위 표에서 가져온다 (DuckDB, CPU)
""")
code(r"""
from gpu_reclaim.db import connect

t0 = time.time()
con = connect(threads=16, memory_limit="64GB")
sim = con.sql('''
    WITH inst AS (SELECT DISTINCT instance_id, day, app_id FROM daily WHERE day BETWEEN 141 AND 146),
    w AS (SELECT instance_id, (time_s // 86400)::INT AS day, time_s, gpu_count, sm_util
          FROM week WHERE time_s >= 141 * 86400),
    aw AS (SELECT i.app_id, w.day, w.time_s, sum(w.gpu_count) AS gpu,
                  max(CASE WHEN w.sm_util > 0 THEN 1 ELSE 0 END) AS active
           FROM w JOIN inst i USING (instance_id, day) GROUP BY 1, 2, 3),
    first AS (SELECT app_id, day, min(time_s) FILTER (WHERE active = 1) AS first_active FROM aw GROUP BY 1, 2)
    SELECT aw.app_id, aw.day,
           any_value(f.first_active) AS first_active,
           sum(aw.gpu) * 5 / 60 AS gpu_h_5min,
           sum(CASE WHEN f.first_active IS NULL OR aw.time_s < f.first_active THEN aw.gpu ELSE 0 END) * 5 / 60 AS saved_gpu_h,
           avg(1 - aw.active) AS idle_window_share
    FROM aw JOIN first f USING (app_id, day) GROUP BY 1, 2
''').df()
print(f"5분 집계 {len(sim):,} app-day, {time.time() - t0:.0f}초")

# 오늘 = 140–145인 test 행에 붙인다 (내일 = day + 1)
ts = test_df[test_df["day"] <= 145].copy()
ts["app_id"] = ts["app_id"].astype(str)
sim["app_id"] = sim["app_id"].astype(str)
sim["day"] = sim["day"] - 1   # 내일 값을 오늘 행 키로
ts = ts.merge(sim, on=["app_id", "day"], how="left")
cover = ts["first_active"].notna() | ts["saved_gpu_h"].notna()
print(f"시뮬레이션 대상 {len(ts):,}행, 5분 기록 있음 {ts['gpu_h_5min'].notna().mean() * 100:.1f}%")

# 라벨 확인: 5분 idle 창 비율과 일 단위 라벨의 관계
m = ts["gpu_h_5min"].notna()
print("일 단위 내일 라벨별 5분 idle 창 비율 중앙값:",
      ts[m].groupby("is_mostly_idle_next")["idle_window_share"].median().round(3).to_dict(),
      "| 내일 하루 활동 없음 비율:", ts[m].groupby("is_mostly_idle_next")["first_active"].apply(lambda s: round(s.isna().mean(), 3)).to_dict())
""")

code(r"""
mask_t = (test_df["day"] <= 145).to_numpy()
ts_ok = ts["gpu_h_5min"].notna().to_numpy()
sim_methods = {"Baseline 1: 오늘 idle": p1t[mask_t]}
for pol in ["F1 최대", "위험 맞춤 (GPU-h)", "위험 맞춤 (건수)"]:
    sim_methods[f"Baseline 1′ (조정) | {pol}"] = (s1t >= THR["Baseline 1′ (조정)"][pol])[mask_t]
    for n in ["Logistic Regression", "SVM (cuML)", "LightGBM 튜닝", "XGBoost 튜닝"]:
        sim_methods[f"{n} | {pol}"] = (SCORES[n]["test"] >= THR[n][pol])[mask_t]
sim_methods["(상한) 전부 회수"] = np.ones(mask_t.sum(), dtype=bool)

rows = []
for name, pred in sim_methods.items():
    r_ = np.asarray(pred).astype(bool) & ts_ok
    cold = r_ & ts["first_active"].notna().to_numpy()
    t_first = (ts.loc[cold, "first_active"] - (ts.loc[cold, "day"] + 1) * 86400) / 3600
    rows.append({"방법": name, "회수 app-day": int(r_.sum()), "cold start 건수": int(cold.sum()),
                 "cold start 비율": cold.sum() / max(r_.sum(), 1),
                 "첫 활동까지 시간 중앙값(h)": float(t_first.median()) if len(t_first) else np.nan,
                 "회수 GPU-h (첫 활동 전까지)": ts.loc[r_, "saved_gpu_h"].sum(),
                 "회수 app의 하루 GPU-h": ts.loc[r_, "gpu_h_5min"].sum()})
sim_res = pd.DataFrame(rows)
b = sim_res.iloc[0]
sim_res["Δcold start vs B1"] = sim_res["cold start 건수"] - b["cold start 건수"]
sim_res["Δ회수 GPU-h vs B1"] = sim_res["회수 GPU-h (첫 활동 전까지)"] - b["회수 GPU-h (첫 활동 전까지)"]
sim_res.to_csv(OUT / "sim_5min_test.csv", index=False)
display(sim_res.round(3))
""")

md("## 10. 요약")
code(r"""
pol = "위험 맞춤 (GPU-h)"
keep = ["Baseline 1: 오늘 idle", "Baseline 1′ (조정)", "Logistic Regression", "SVM (cuML)", "Random Forest",
        "LightGBM 튜닝", "XGBoost 튜닝"]
t_ = test_res[test_res["방법"].isin(keep) & test_res["정책"].isin(["-", pol])].set_index("방법")
h_ = pooled[pooled["방법"].isin(keep) & pooled["정책"].isin(["-", pol])].set_index("방법")
summary = pd.DataFrame({
    "test F1": t_["F1"], "test 회수 GPU-h": t_["회수 GPU-h"], "test 잘못 회수 GPU-h": t_["잘못 회수 GPU-h"],
    "test 잘못 회수 건수": t_["잘못 회수 건수"],
    "홀드아웃 F1": h_["F1"], "홀드아웃 회수 GPU-h": h_["회수 GPU-h"], "홀드아웃 잘못 회수 GPU-h": h_["잘못 회수 GPU-h"],
    "홀드아웃 잘못 회수 건수": h_["잘못 회수 건수"],
}).loc[keep]
print(f"정책: {pol} (val/직전 주에서 Baseline 1과 같은 잘못 회수 GPU-h 비율이 되도록 threshold 선택)")
display(summary.round(4))
summary.to_csv(OUT / "summary.csv")
print("저장 위치:", OUT)
""")

def write():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "gpu-reclaim-gpu", "display_name": "Python (gpu-reclaim GPU)", "language": "python"}
    nbf.write(nb, OUT)
    print("written", OUT, len(cells), "cells")


if __name__ == "__main__":   # make_v4_notebook.py가 셀을 가져다 쓸 때는 쓰지 않는다
    write()
