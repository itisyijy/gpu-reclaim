"""팀원 노트북 v2의 데이터·인코더 그대로, 모델별 CPU 대 GPU 학습 시간과 val AP 비교.

- 데이터: data/processed/app_day_v2_max181.parquet 중 day<=146 (노트북과 동일 범위)
- 학습 train(0–132) → val(133–139) gpu_hours_next 가중 AP
- GPU는 CUDA_VISIBLE_DEVICES로 지정 (0번은 다른 사용자가 사용 중이라 1번)
"""
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
cells = ["".join(c["source"]) for c in json.load(open(ROOT / "notebooks/2026_maas_analysis_v2.ipynb"))["cells"] if c["cell_type"] == "code"]
enc_cell = next(s for s in cells if "def build_linear_encoder" in s)
df = pd.read_parquet(ROOT / "data/processed/app_day_v2_max181.parquet")
df = df[df["day"] <= 146].reset_index(drop=True)
# parquet 왕복에서 category가 int로 풀림 -> 노트북과 같게 되돌림
df["day_of_week"] = pd.Categorical(df["day_of_week"], categories=range(7))
g = {"__name__": "__main__", "display": print, "preprocessed_df": df, "RANDOM_STATE": 0, "np": np, "pd": pd}
exec(enc_cell, g)
split_xyw, build_linear_encoder, build_tree_encoder = g["split_xyw"], g["build_linear_encoder"], g["build_tree_encoder"]

tr, va = df[df.day <= 132], df[df.day.between(133, 139)]
Xtr, ytr, wtr = split_xyw(tr); Xva, yva, wva = split_xyw(va)
wtr = wtr.to_numpy().copy(); wva = wva.to_numpy().copy()
print(f"train {len(tr):,} / val {len(va):,}")

te = build_tree_encoder().fit(Xtr); Ttr, Tva = te.transform(Xtr), te.transform(Xva)
le = build_linear_encoder().fit(Xtr)
dense = lambda a: (a.toarray() if sp.issparse(a) else np.asarray(a)).astype("float32")
Ltr, Lva = dense(le.transform(Xtr)), dense(le.transform(Xva))
print("linear features", Ltr.shape[1])

rows = []
def run(name, dev, fit, score):
    t = time.time()
    try:
        m = fit(); el = time.time() - t
        ap = average_precision_score(yva, score(m), sample_weight=wva)
    except Exception as e:
        el, ap = time.time() - t, float("nan"); print("  ERROR", name, dev, repr(e)[:300])
    rows.append({"task": name, "device": dev, "sec": round(el, 1), "val_AP": round(ap, 4)})
    print(rows[-1], flush=True)

only = set(sys.argv[1:])
want = lambda k: not only or k in only

if want("xgb"):
    import xgboost as xgb
    for tag, kw in [("XGB 노트북 최적 (lr.05·d3·300)", dict(learning_rate=0.05, max_depth=3, n_estimators=300)),
                    ("XGB 큰 설정 (lr.02·d8·2000)", dict(learning_rate=0.02, max_depth=8, n_estimators=2000))]:
        for dev in (["cuda"] if "큰" in tag and os.environ.get("NO_CPU_BIG") else ["cpu", "cuda"]):
            run(tag, dev, lambda: xgb.XGBClassifier(enable_categorical=True, tree_method="hist", device=dev,
                n_jobs=20, random_state=0, **kw).fit(Ttr, ytr, sample_weight=wtr), lambda m: m.predict_proba(Tva)[:, 1])

if want("lgbm"):
    import lightgbm as lgb
    for tag, kw in [("LGBM 노트북 최적 (lr.05·300·31잎)", dict(learning_rate=0.05, n_estimators=300, num_leaves=31)),
                    ("LGBM 큰 설정 (lr.02·2000·127잎)", dict(learning_rate=0.02, n_estimators=2000, num_leaves=127))]:
        for dev in (["cuda"] if "큰" in tag and os.environ.get("NO_CPU_BIG") else ["cpu", "cuda"]):
            run(tag, dev, lambda: lgb.LGBMClassifier(device_type=dev, n_jobs=20, random_state=0, verbose=-1, **kw)
                .fit(Ttr, ytr, sample_weight=wtr), lambda m: m.predict_proba(Tva)[:, 1])

if want("rf"):
    from sklearn.ensemble import RandomForestClassifier
    from cuml.ensemble import RandomForestClassifier as cuRF
    run("RF 300트리 깊이16", "cpu", lambda: RandomForestClassifier(300, max_depth=16, n_jobs=20, random_state=0)
        .fit(Ltr, ytr, sample_weight=wtr), lambda m: m.predict_proba(Lva)[:, 1])
    run("RF 300트리 깊이16 (가중치 없음)", "cuda", lambda: cuRF(n_estimators=300, max_depth=16, n_bins=128, random_state=0)
        .fit(Ltr, ytr.to_numpy().astype("int32")), lambda m: np.asarray(m.predict_proba(Lva))[:, 1])

if want("svm"):
    from sklearn.svm import SVC
    from cuml.svm import SVC as cuSVC
    rng = np.random.default_rng(0)
    for n in [20_000, 100_000, len(Ltr)]:
        idx = rng.choice(len(Ltr), size=n, replace=False) if n < len(Ltr) else np.arange(len(Ltr))
        if n <= 20_000:
            run(f"SVM RBF C0.1 γ0.01 train {n:,}행", "cpu", lambda: SVC(C=0.1, gamma=0.01).fit(Ltr[idx], ytr.iloc[idx], sample_weight=wtr[idx]),
                lambda m: m.decision_function(Lva))
        run(f"SVM RBF C0.1 γ0.01 train {n:,}행", "cuda", lambda: cuSVC(C=0.1, gamma=0.01, cache_size=4096).fit(Ltr[idx], ytr.to_numpy()[idx].astype("float32"), sample_weight=wtr[idx]),
            lambda m: np.asarray(m.decision_function(Lva)))

out = ROOT / "outputs" / "bench_cpu_gpu.csv"; out.parent.mkdir(exist_ok=True)
prev = pd.read_csv(out) if out.exists() and only else pd.DataFrame()
pd.concat([prev, pd.DataFrame(rows)]).to_csv(out, index=False)
print(pd.DataFrame(rows).to_string())
