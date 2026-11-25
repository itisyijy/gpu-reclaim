"""v4 노트북: v3에서 test는 모델이 낫고 홀드아웃은 Baseline 1과 비슷했던 이유를 가려내는 후속 실험.

v3 생성 스크립트의 설정·데이터·인코더·공통 함수·학습 함수 셀을 그대로 가져다 쓰고,
하이퍼파라미터는 v3 결과(outputs/v3/best_params.json)로 고정한다.
사용: env-gpu/bin/python scripts/make_v4_notebook.py
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_v3_notebook as v3  # noqa: E402  (import만 하면 v3 노트북 파일은 다시 쓰지 않는다)

OUT = ROOT / "notebooks" / "2026_maas_analysis_v4_robust.ipynb"


def v3_cell(marker):
    hits = [c for c in v3.cells if c.cell_type == "code" and marker in c.source]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0].source


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Predicting Tomorrow's Idle GPUs — v4 (견고성·원인 분석)

v3 결과: **test 주(140–146)** 에서는 튜닝한 LightGBM이 Baseline 1보다 같은 위험에서 더 많이 회수했지만,
**홀드아웃 5주(147–180)** 에서는 이득이 거의 없고 잘못 회수 건수는 오히려 늘었다.
v4는 이 차이가 어디서 오는지 가려낸다. 모든 실험은 **같은 6주(test + 홀드아웃 5주)를 같은 방식**으로 평가한다.

| 실험 | 질문 | 방법 |
| --- | --- | --- |
| E1 threshold 대 순위 | 모델의 **점수 순서**가 나쁜가, 아니면 직전 주에서 고른 **threshold가 어긋나는** 것인가? | 같은 점수에 정책 여러 개 적용: 직전 주 threshold / 같은 날 Baseline 1과 같은 개수 / 그 주에서 위험을 정확히 맞춘 사후 상한 |
| E2 seed | 이득이 학습 난수에 따라 흔들리는가? | LightGBM·XGBoost 튜닝 설정을 seed 5개로 |
| E3 변수 묶음 | Baseline 1(오늘 idle_ratio)에 무엇을 더해야 이득이 생기나? | 변수 묶음을 차례로 더하기 / 하나씩 빼기 (LightGBM) |
| E4 학습 기간 | 오래된 날이 오히려 방해되나(분포 변화)? | 전체 과거 / 최근 56일 / 최근 28일로 학습 |

평가 방식 (v3 홀드아웃과 같음): 주 `[a, b]`를 예측할 때 최종 모델은 `day ≤ a−2`로 학습,
threshold는 `day ≤ a−9` 모델을 `a−8 … a−2`에 적용해서 고른다. 하이퍼파라미터는 v3에서 고른 값으로 고정.

**정책** (모두 같은 점수에 적용)
- `F1 최대 (직전 주)`, `위험 맞춤 GPU-h (직전 주)`, `위험 맞춤 건수 (직전 주)`: v3와 같음
- `같은 날 B1과 같은 개수`: 그날 Baseline 1이 회수하는 app 수 k만큼 점수 상위 k개를 회수 — 정답 없이 **그날 바로 쓸 수 있는** 규칙.
  threshold 이동 문제를 없애고 "누구를 고르느냐"만 비교한다
- `사후 맞춤 GPU-h (상한)`, `사후 맞춤 건수 (상한)`: 그 주의 정답을 보고 Baseline 1과 잘못 회수량을 정확히 맞춘 threshold —
  운영에서는 불가능한 **순위 품질의 상한**
""")

md("## 0. 설정 (v3와 같음)")
gpu_cell = v3_cell("def spare_gpus(")
# 내 다른 실행(v3)이 같은 GPU를 쓰는 중일 수 있어 GPU_IDS 환경 변수로 직접 지정할 수 있게 한다
gpu_cell = gpu_cell.replace("GPUS, peak = spare_gpus()",
    'GPUS, peak = spare_gpus()\nif os.environ.get("GPU_IDS"):\n    GPUS = [int(x) for x in os.environ["GPU_IDS"].split(",")]')
assert "GPU_IDS" in gpu_cell
code(gpu_cell)
imports = v3_cell("N_TRIALS = 40")
imports = imports.replace('OUT = ROOT / "outputs" / "v3"', 'OUT = ROOT / "outputs" / "v4"\nV3_OUT = ROOT / "outputs" / "v3"')
imports = imports.replace('SMOKE = os.environ.get("V3_SMOKE") == "1"', 'SMOKE = os.environ.get("V4_SMOKE") == "1"')
assert 'V3_OUT' in imports and 'V4_SMOKE' in imports
code(imports)

md("## 1. 데이터 (v3와 같음)")
code(v3_cell("DATA = ROOT /"))

md("## 2. 인코더·공통 함수·Baseline·학습 함수 (v3와 같음)")
code(v3.enc)
code(v3_cell("def pick_threshold("))
code(v3_cell("def base2("))
code(v3_cell("def fit_model("))
boot = v3_cell("def boot_diff(")
boot = boot[:boot.index("\nfor pol in")].rstrip() + "\n"
code(boot)

md("""
## 3. 공통 평가 틀

- `EVAL_WEEKS`: test 주 + 홀드아웃 5주. 합계는 **6주 전체**와 **홀드아웃 5주** 두 가지로 낸다
- 하이퍼파라미터: v3의 `best_params.json`
- 점수는 전부 `outputs/v4/scores.parquet`에 저장 (나중에 다시 학습하지 않고 다른 정책을 시험할 수 있게)
""")
code(r"""
BEST = json.load(open(V3_OUT / "best_params.json"))
print("v3에서 고른 하이퍼파라미터:", BEST)
HOLD_WEEKS = [(147, 153), (154, 160), (161, 167), (168, 174), (175, 180)]
EVAL_WEEKS = [(140, 146)] + HOLD_WEEKS
if SMOKE:
    EVAL_WEEKS = EVAL_WEEKS[:2]
WEEK_NAMES = [f"{a}–{b}" for a, b in EVAL_WEEKS]
POOLS = {"6주 (140–180)": WEEK_NAMES, "홀드아웃 5주 (147–180)": [n for n in WEEK_NAMES if n != "140–146"]}

# 주별 고정 정보 (정답·가중치·Baseline 1) — 변수 묶음 실험에서도 원본 df 기준
WK = []
for a, b in EVAL_WEEKS:
    wk, cal = df[df["day"].between(a, b)], df[df["day"].between(a - 8, a - 2)]
    _, yk, wk_w = split_xyw(wk)
    _, yc, wc = split_xyw(cal)
    WK.append({"name": f"{a}–{b}", "a": a, "b": b, "wk": wk, "cal": cal,
               "yk": yk.to_numpy().astype(int), "wk_w": wk_w.to_numpy().astype("float64"),
               "yc": yc.to_numpy().astype(int), "wc": wc.to_numpy().astype("float64"),
               "b1k": base1(wk)[1], "b1c": base1(cal)[1], "s1k": base1(wk)[0], "s1c": base1(cal)[0],
               "day": wk["day"].to_numpy(), "workload": wk["workload"].astype(str).to_numpy()})
    WK[-1]["bud"] = risk_budget(WK[-1]["yc"], WK[-1]["b1c"], WK[-1]["wc"])
print("평가 주:", [(w["name"], len(w["wk"])) for w in WK])


def topk_per_day(s, day, base):
    # 그날 Baseline 1이 회수하는 개수 k만큼 점수 상위 k개 (정답 없이 그날 바로 쓸 수 있음)
    pred = np.zeros(len(s), dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        k = int(base[idx].sum())
        if k:
            pred[idx[np.argsort(-s[idx], kind="stable")[:k]]] = True
    return pred


def oracle_pred(y, s, w, base, col):
    # 그 주 정답을 보고 Baseline 1과 같은 잘못 회수량(col) 이하에서 회수 최대인 threshold (상한, 운영 불가)
    c = threshold_curve(y, s, w)
    fp = (np.asarray(base) == 1) & (y == 0)
    b = w[fp].sum() if col == "잘못 회수 GPU-h" else fp.sum()
    ok = c[col] <= b + 1e-9
    thr = float(c.loc[ok[ok].index[-1], "thr"]) if ok.any() else np.inf
    return s >= thr


POL_NAMES = ["F1 최대 (직전 주)", "위험 맞춤 GPU-h (직전 주)", "위험 맞춤 건수 (직전 주)",
             "같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "사후 맞춤 건수 (상한)"]


def policy_preds(W, s_c, s_k):
    yc, wc, bud = W["yc"], W["wc"], W["bud"]
    return {
        "F1 최대 (직전 주)": s_k >= pick_threshold(yc, s_c, wc, "F1 최대"),
        "위험 맞춤 GPU-h (직전 주)": s_k >= pick_threshold(yc, s_c, wc, "위험 맞춤 (GPU-h)", bud["위험 맞춤 (GPU-h)"]),
        "위험 맞춤 건수 (직전 주)": s_k >= pick_threshold(yc, s_c, wc, "위험 맞춤 (건수)", bud["위험 맞춤 (건수)"]),
        "같은 날 B1과 같은 개수": topk_per_day(s_k, W["day"], W["b1k"]),
        "사후 맞춤 GPU-h (상한)": oracle_pred(W["yk"], s_k, W["wk_w"], W["b1k"], "잘못 회수 GPU-h"),
        "사후 맞춤 건수 (상한)": oracle_pred(W["yk"], s_k, W["wk_w"], W["b1k"], "잘못 회수 건수"),
    }


WEEKLY, POOLED, PRED_STORE, SCORE_STORE = [], [], {}, {}


def record(exp, setting, method, seed, week_scores=None, fixed_pred=None):
    # week_scores: 주별 (cal 점수, week 점수) 목록. fixed_pred: 정책 없는 0/1 판정(Baseline 1)
    preds_by_pol = {}
    for i, W in enumerate(WK):
        if fixed_pred is not None:
            pp, sk = {"-": fixed_pred[i]}, W["s1k"]
        else:
            s_c, sk = week_scores[i]
            pp = policy_preds(W, s_c, sk)
            SCORE_STORE[(exp, setting, method, seed, W["name"])] = sk
        for pol, pred in pp.items():
            WEEKLY.append({"실험": exp, "설정": setting, "방법": method, "seed": seed, "주": W["name"], "정책": pol,
                           **evaluate(W["yk"], sk, pred, W["wk_w"])})
            preds_by_pol.setdefault(pol, []).append(np.asarray(pred).astype(int))
    for pol, plist in preds_by_pol.items():
        PRED_STORE[(exp, setting, method, seed, pol)] = plist
        for pool, names in POOLS.items():
            idx = [i for i, W in enumerate(WK) if W["name"] in names]
            if not idx:
                continue
            y = np.concatenate([WK[i]["yk"] for i in idx]); w = np.concatenate([WK[i]["wk_w"] for i in idx])
            p = np.concatenate([plist[i] for i in idx]); wl = np.concatenate([WK[i]["workload"] for i in idx])
            for seg in ["전체", "online", "offline"]:
                m = np.ones(len(y), bool) if seg == "전체" else (wl == seg)
                r = evaluate(y[m], p[m].astype(float), p[m], w[m]); r.pop("AP")
                POOLED.append({"실험": exp, "설정": setting, "방법": method, "seed": seed, "구간": pool, "workload": seg,
                               "정책": pol, **r})


def walk_forward(kind, params, data=None, seed=0, window=None):
    # 주마다 (cal 점수, week 점수). window: 최근 며칠만 학습 (None = 전체 과거)
    global RANDOM_STATE
    data = df if data is None else data
    RANDOM_STATE = seed
    out = []
    try:
        for W in WK:
            a = W["a"]

            def trn(end):
                m = data["day"] <= end
                if window:
                    m &= data["day"] > end - window
                return data[m]

            cal, wk = data[data["day"].between(a - 8, a - 2)], data[data["day"].between(a, W["b"])]   # WK와 같은 순서
            s_c = fit_model(kind, params, trn(a - 9), device=DEVICES[0]).score(cal)
            s_k = fit_model(kind, params, trn(a - 2), device=DEVICES[0]).score(wk)
            out.append((s_c, s_k))
    finally:
        RANDOM_STATE = 0
    return out


def pooled_view(exp, pol_list=None, seg="전체"):
    p = pd.DataFrame(POOLED)
    p = p[(p["실험"] == exp) & (p["workload"] == seg)]
    b = pd.DataFrame(POOLED)
    b = b[(b["실험"] == "E1") & (b["방법"] == "Baseline 1: 오늘 idle") & (b["workload"] == seg)].set_index("구간")
    for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "F1"]:
        p[f"Δ{c} vs B1"] = p[c] - p["구간"].map(b[c])
    if pol_list:
        p = p[p["정책"].isin(pol_list + ["-"])]
    return p


T0 = time.time()
def elapsed():
    return f"{(time.time() - T0) / 60:.1f}분"
""")

md("""
## 4. E1 — threshold 문제인가, 순위 문제인가

방법: Baseline 1, Baseline 1′(idle_ratio 점수에 정책 적용), LR, RF, LightGBM 튜닝, XGBoost 튜닝, 두 부스팅의 확률 평균. seed 0.
- `사후 맞춤`에서도 모델이 Baseline 1과 비슷하면 → **점수 순서 자체**가 Baseline 1보다 낫지 않다
- `사후 맞춤`에서는 낫고 `직전 주`에서만 못하면 → **threshold가 주마다 어긋나는** 문제
- `같은 날 B1과 같은 개수`는 threshold 없이 바로 쓸 수 있는 대안
""")
code(r"""
record("E1", "기본", "Baseline 1: 오늘 idle", 0, fixed_pred=[W["b1k"] for W in WK])
record("E1", "기본", "Baseline 1′ (idle_ratio 점수)", 0, week_scores=[(W["s1c"], W["s1k"]) for W in WK])
E1_MODELS = {"Logistic Regression": ("lr", BEST["lr"]), "Random Forest": ("rf", BEST["rf"]),
             "LightGBM 튜닝": ("lgbm", BEST["lgbm"]), "XGBoost 튜닝": ("xgb", BEST["xgb"])}
E1_SCORES = {}
for name, (kind, p) in E1_MODELS.items():
    t0 = time.time()
    E1_SCORES[name] = walk_forward(kind, p)
    record("E1", "기본", name, 0, week_scores=E1_SCORES[name])
    print(f"E1 {name}: {time.time() - t0:.0f}초 (누적 {elapsed()})", flush=True)
ens = [((a1 + a2) / 2, (b1 + b2) / 2) for (a1, b1), (a2, b2) in zip(E1_SCORES["LightGBM 튜닝"], E1_SCORES["XGBoost 튜닝"])]
record("E1", "기본", "LGBM+XGB 평균", 0, week_scores=ens)

e1 = pooled_view("E1")
e1.to_csv(OUT / "e1_pooled.csv", index=False)
pd.DataFrame([r for r in WEEKLY if r["실험"] == "E1"]).to_csv(OUT / "e1_weekly.csv", index=False)
cols = ["방법", "정책", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "ΔF1 vs B1", "Δ회수 GPU-h vs B1",
        "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]
for pool in POOLS:
    print(f"\n=== E1 | {pool} | workload 전체 ===")
    display(e1[e1["구간"] == pool][cols].round(4))
""")

code(r"""
# 주별 AP (점수 순서의 품질, threshold와 무관)
w1 = pd.DataFrame([r for r in WEEKLY if r["실험"] == "E1" and r["정책"] in ("-", "F1 최대 (직전 주)")])
display(w1.pivot_table(index="방법", columns="주", values="AP").round(4))
# 주별 Δ회수 (사후 맞춤 GPU-h, 같은 날 같은 개수) — 이득이 특정 주에 몰려 있는지
w2 = pd.DataFrame([r for r in WEEKLY if r["실험"] == "E1"])
b1w = w2[w2["방법"] == "Baseline 1: 오늘 idle"].set_index("주")
for pol in ["같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "위험 맞춤 GPU-h (직전 주)"]:
    t = w2[w2["정책"] == pol].copy()
    t["Δ회수 GPU-h"] = t["회수 GPU-h"] - t["주"].map(b1w["회수 GPU-h"])
    t["Δ잘못 회수 건수"] = t["잘못 회수 건수"] - t["주"].map(b1w["잘못 회수 건수"])
    print(f"\n주별 Δ vs B1 | 정책: {pol}")
    display(t.pivot_table(index="방법", columns="주", values=["Δ회수 GPU-h", "Δ잘못 회수 건수"]).round(0))
""")

code(r"""
# E1 부트스트랩 (app 단위) — 운영에서 쓸 수 있는 두 정책
boots = []
for pool, names in POOLS.items():
    idx = [i for i, W in enumerate(WK) if W["name"] in names]
    pool_df = pd.concat([WK[i]["wk"] for i in idx])
    base = np.concatenate([WK[i]["b1k"] for i in idx])
    for pol in ["위험 맞춤 GPU-h (직전 주)", "같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)"]:
        preds = {m: np.concatenate([PRED_STORE[("E1", "기본", m, 0, pol)][i] for i in idx])
                 for m in ["Baseline 1′ (idle_ratio 점수)", *E1_MODELS, "LGBM+XGB 평균"]}
        bt = boot_diff(pool_df, preds, base)
        bt.insert(0, "정책", pol); bt.insert(0, "구간", pool)
        boots.append(bt)
boots = pd.concat(boots)
boots.to_csv(OUT / "e1_bootstrap.csv", index=False)
display(boots)
print("누적", elapsed())
""")

code(r"""
# E1 workload별 (홀드아웃 5주) — 이득/손해가 online·offline 어디서 나는지
e1w = pd.concat([pooled_view("E1", seg="online"), pooled_view("E1", seg="offline")])
e1w.to_csv(OUT / "e1_pooled_by_workload.csv", index=False)
pool = list(POOLS)[-1]
display(e1w[(e1w["구간"] == pool) & e1w["정책"].isin(["-", "같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "위험 맞춤 GPU-h (직전 주)"])]
        [["workload", "방법", "정책", "F1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]].round(4))
""")

md("""
## 5. E2 — seed에 따른 흔들림

LightGBM·XGBoost 튜닝 설정(subsample·colsample이 1보다 작아 난수 영향이 있음)을 seed 0–4로 다시 학습.
seed마다 합계 Δ vs Baseline 1을 내고 평균·표준편차·최소·최대를 본다.
""")
code(r"""
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
for name, kind in [("LightGBM 튜닝", "lgbm"), ("XGBoost 튜닝", "xgb")]:
    for sd in SEEDS:
        sc = E1_SCORES[name] if sd == 0 else walk_forward(kind, BEST[kind], seed=sd)
        record("E2", "seed", name, sd, week_scores=sc)
    print(f"E2 {name}: 누적 {elapsed()}", flush=True)
e2 = pooled_view("E2")
e2.to_csv(OUT / "e2_seeds_pooled.csv", index=False)
dcols = ["ΔF1 vs B1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]
e2s = e2.groupby(["구간", "방법", "정책"])[dcols].agg(["mean", "std", "min", "max"])
e2s.to_csv(OUT / "e2_seeds_summary.csv")
display(e2s.round(3))
""")

md("""
## 6. E3 — 변수 묶음 (LightGBM 튜닝, seed 0)

v2 입력 변수 23개를 6묶음으로 나눈다. 묶음을 "뺄" 때는 그 열을 상수로 바꾼다(트리가 쓰지 못함, 열 구성·인코더는 그대로).
- **더하기**: idle_ratio만 → +이력 → +오늘 사용량 → +규모 → +메타 → +달력(= 전체)
- **하나씩 빼기**: 전체에서 한 묶음만 뺌
""")
code(r"""
GROUPS = {
    "오늘 idle": ["idle_ratio"],
    "이력": ["idle_ratio_lag1", "idle_streak"],
    "오늘 사용량": ["sm_util", "gpu_mem_util", "mem_usage", "outlier_sm_util", "outlier_gpu_mem_util", "outlier_mem_usage"],
    "규모": ["gpu_type", "gpu_count", "gpu_mem_alloc", "gpu_hours", "gpu_hours_per_gpu"],
    "메타": ["workload", "pd_role", "params_b", "model_arch", "base_arch", "variant_arch", "cluster_id", "site_id"],
    "달력": ["day_of_week"],
}
X_all = split_xyw(df.head())[0].columns
assert sorted(sum(GROUPS.values(), [])) == sorted(X_all), set(X_all) ^ set(sum(GROUPS.values(), []))


def ablate(keep_groups):
    d = df.copy()
    for g, cols in GROUPS.items():
        if g in keep_groups:
            continue
        for c in cols:
            if isinstance(d[c].dtype, pd.CategoricalDtype):
                d[c] = pd.Categorical([d[c].cat.categories[0]] * len(d), categories=d[c].cat.categories)
            else:
                d[c] = np.zeros(len(d), dtype=d[c].dtype)
    return d


order = list(GROUPS)
configs = {}
for i in range(1, len(order)):          # 더하기 (전체는 E1 결과 재사용)
    configs["더하기: " + " + ".join(order[:i])] = order[:i]
for g in order[1:]:                       # 하나씩 빼기 (오늘 idle 빼기도 포함)
    configs[f"빼기: −{g}"] = [x for x in order if x != g]
configs["빼기: −오늘 idle"] = order[1:]
if SMOKE:
    configs = dict(list(configs.items())[:2])
record("E3", "전체 (23개)", "LightGBM 튜닝", 0, week_scores=E1_SCORES["LightGBM 튜닝"])
for cname, keep in configs.items():
    record("E3", cname, "LightGBM 튜닝", 0, week_scores=walk_forward("lgbm", BEST["lgbm"], data=ablate(keep)))
    print(f"E3 {cname}: 누적 {elapsed()}", flush=True)
e3 = pooled_view("E3", ["같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "위험 맞춤 GPU-h (직전 주)"])
e3w = pd.DataFrame([r for r in WEEKLY if r["실험"] == "E3" and r["정책"] == "F1 최대 (직전 주)"])
ap = e3w.groupby("설정")["AP"].mean().rename("주별 AP 평균")
e3 = e3.merge(ap, left_on="설정", right_index=True, how="left")
e3.to_csv(OUT / "e3_ablation.csv", index=False)
for pool in POOLS:
    print(f"\n=== E3 | {pool} ===")
    display(e3[e3["구간"] == pool].pivot_table(index="설정", columns="정책", values="Δ회수 GPU-h vs B1", sort=False).round(0)
            .join(e3[e3["구간"] == pool].groupby("설정")["주별 AP 평균"].first()))
""")

md("""
## 7. E4 — 학습 기간 (LightGBM·XGBoost 튜닝, seed 0)

매주 학습할 때 전체 과거 / 최근 56일 / 최근 28일만 쓴다. 최근 기간이 낫다면 분포 변화(drift)가 홀드아웃 성능을 깎는 것이다.
""")
code(r"""
WINDOWS = [None, 28] if SMOKE else [None, 56, 28]
for name, kind in [("LightGBM 튜닝", "lgbm"), ("XGBoost 튜닝", "xgb")]:
    for win in WINDOWS:
        lab = "전체 과거" if win is None else f"최근 {win}일"
        sc = E1_SCORES[name] if win is None else walk_forward(kind, BEST[kind], window=win)
        record("E4", lab, name, 0, week_scores=sc)
        print(f"E4 {name} {lab}: 누적 {elapsed()}", flush=True)
e4 = pooled_view("E4", ["같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "위험 맞춤 GPU-h (직전 주)"])
e4w = pd.DataFrame([r for r in WEEKLY if r["실험"] == "E4" and r["정책"] == "F1 최대 (직전 주)"])
e4 = e4.merge(e4w.groupby(["설정", "방법"])["AP"].mean().rename("주별 AP 평균").reset_index(), on=["설정", "방법"], how="left")
e4.to_csv(OUT / "e4_window.csv", index=False)
for pool in POOLS:
    print(f"\n=== E4 | {pool} ===")
    display(e4[e4["구간"] == pool][["방법", "설정", "정책", "주별 AP 평균", "F1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1",
                                    "Δ잘못 회수 건수 vs B1"]].round(4))
""")

md("## 8. 저장")
code(r"""
pd.DataFrame(WEEKLY).to_csv(OUT / "weekly_all.csv", index=False)
pd.DataFrame(POOLED).to_csv(OUT / "pooled_all.csv", index=False)
# 점수 저장: 주별 행 순서는 df에서 그 주를 고른 순서 그대로
key_rows = []
for W in WK:
    key_rows.append(pd.DataFrame({"주": W["name"], "row_index": W["wk"].index, "app_id": W["wk"]["app_id"].astype(str).values,
                                  "day": W["day"], "y": W["yk"], "w": W["wk_w"], "b1": W["b1k"], "idle_ratio": W["s1k"]}))
keys = pd.concat(key_rows, ignore_index=True)
cols = {}
for (exp, setting, method, seed, wname) in SCORE_STORE:
    cols.setdefault(f"{exp}|{setting}|{method}|seed{seed}", None)
for cname in cols:
    exp, setting, method, seed = cname.split("|")
    cols[cname] = np.concatenate([SCORE_STORE[(exp, setting, method, int(seed[4:]), W["name"])] for W in WK]).astype("float32")
scores = pd.concat([keys, pd.DataFrame(cols)], axis=1)
scores.to_parquet(OUT / "scores.parquet", index=False)
print(f"저장: {OUT} | 점수 {scores.shape} | 전체 {elapsed()}")
""")


def write():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "gpu-reclaim-gpu", "display_name": "Python (gpu-reclaim GPU)", "language": "python"}
    nbf.write(nb, OUT)
    print("written", OUT, len(cells), "cells")


if __name__ == "__main__":
    write()
