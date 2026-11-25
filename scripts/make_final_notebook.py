"""최종 노트북: 과제 요구사항(문제·데이터·방법·결과·논의)에 맞춰 최종 모델과 결과를 한곳에 모은다.

최종 모델 = LightGBM 튜닝 + XGBoost 튜닝 확률 평균, 운영 규칙 = '같은 날 Baseline 1과 같은 개수' 회수.
v3·v4 생성 스크립트의 셀(설정·데이터·인코더·평가 함수·학습 함수·walk-forward 틀)을 그대로 가져다 쓰고,
하이퍼파라미터는 v3 결과(outputs/v3/best_params.json)로 고정한다 (다시 튜닝하지 않음).
사용: env-gpu/bin/python scripts/make_final_notebook.py
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_v3_notebook as v3  # noqa: E402  (import만 하면 노트북 파일은 다시 쓰지 않는다)
import make_v4_notebook as v4  # noqa: E402

OUT = ROOT / "notebooks" / "2026_maas_analysis_final.ipynb"


def pick(mod, marker):
    hits = [c for c in mod.cells if c.cell_type == "code" and marker in c.source]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0].source


cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Predicting Tomorrow's Idle GPUs — 최종 분석

## 문제와 목표
LLM 추론 서비스는 요청이 없어도 GPU를 붙잡아 둔다. **내일 거의 쓰이지 않을 app**을 오늘 미리 알아내면
그 GPU를 회수해 다른 작업에 쓸 수 있다. 반대로 잘못 회수하면 다음날 요청이 들어올 때 **cold start**(다시 올리는 지연)가 생긴다.

- 목표: 운영자가 매일 쓰는 단순 규칙 **Baseline 1 ("오늘 거의 idle이면 내일도 회수")** 보다
  **같은 양을 회수하면서 cold start로 방해받는 작업을 줄이는** 예측 모델을 만든다.

## 데이터
- Alibaba MaaS trace v2026의 원시 표 세 개를 직접 가공해 데이터셋을 만들었다.
  - `instance_daily`: 인스턴스×일, 1.45억 행
  - `instance_week_5min`: 5분 단위, 1.41억 행
  - `app_metrics`: app 메타데이터
- 이 표들을 **app×day** 단위로 집계했다. 0–181일, 약 51만 행이다(`scripts/build_app_day_v2.py`).
- target: `is_mostly_idle_next`. **내일** idle 비율이 0.9 이상이면 1이다.
- 가중치: `gpu_hours_next`. 내일 그 app이 쓰는 GPU-h로, 큰 app의 실수가 더 크게 반영된다.
- 입력 변수: 23개, 6묶음(아래 1절).

## 방법
- 시간 순서를 지키는 **walk-forward 평가**를 한다.
  - 주 `[a, b]`를 예측할 때 `day ≤ a−2`의 데이터로만 학습한다(라벨이 하루 늦게 나오므로).
  - 평가 구간: test 주 140–146, 홀드아웃 5주 147–180. 모든 주를 직전 데이터로 다시 학습한다.
- 비교 대상:
  - 기준선: Baseline 1, Baseline 1′(idle_ratio 연속 점수), Baseline 2(오늘 sm_util이 가장 낮은 N개)
  - 모델: Logistic Regression, SVM(cuML), Random Forest, LightGBM, XGBoost, LightGBM+XGBoost 평균
- 하이퍼파라미터: v3 노트북의 4주 rolling CV + Optuna(GPU)로 골랐다.
- **운영 규칙(주 정책): 같은 날 B1과 같은 개수.**
  - 매일 Baseline 1이 회수하는 app 수 k만큼, 모델 점수 상위 k개를 회수한다.
  - threshold도 정답도 필요 없어서 그날 바로 쓸 수 있다.
  - 회수량이 B1과 같아지므로, 비교는 **누구를 고르느냐**로 좁혀진다.
- 참고 정책: v3와 같이 직전 주에서 threshold를 고르는 방식(F1 최대, 위험 맞춤)도 함께 보고한다.
- 불확실성: app 단위 부트스트랩 95% 구간.
- 실제 cold start 확인: 5분 단위 시뮬레이션(test 주).

## 최종 모델
**LightGBM 튜닝 + XGBoost 튜닝의 확률 평균**을, 같은 날 B1과 같은 개수 규칙으로 운영한다.
- 선택 근거(v4 견고성 실험):
  - 홀드아웃 5주에서 잘못 회수 GPU-h, 잘못 회수 건수, F1 모두 B1보다 유의하게 나았다.
  - seed 5개에서 효과의 방향이 같았다.
  - 학습 기간을 줄여도 나아지지 않았다.
""")

md("## 0. 설정")
code(pick(v4, "def spare_gpus("))
imports = pick(v3, "N_TRIALS = 40")
imports = imports.replace('OUT = ROOT / "outputs" / "v3"', 'OUT = ROOT / "outputs" / "final"\nV3_OUT = ROOT / "outputs" / "v3"')
imports = imports.replace('SMOKE = os.environ.get("V3_SMOKE") == "1"', 'SMOKE = os.environ.get("FINAL_SMOKE") == "1"')
assert "V3_OUT" in imports and "FINAL_SMOKE" in imports
code(imports)

md("## 1. 데이터와 변수")
code(pick(v3, "DATA = ROOT /"))
code(v3.enc)
code(pick(v3, "def pick_threshold("))
code(pick(v3, "def base2("))
code(pick(v3, "def fit_model("))
boot = pick(v3, "def boot_diff(")
code(boot[:boot.index("\nfor pol in")].rstrip() + "\n")

code(r"""
# 입력 변수 23개, 6묶음
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
FEAT_GROUP = {c: g for g, cols in GROUPS.items() for c in cols}
display(pd.DataFrame([{"묶음": g, "변수 수": len(c), "변수": ", ".join(c)} for g, c in GROUPS.items()]))
print("target: is_mostly_idle_next (내일 idle 비율 ≥ 0.9) | 가중치: gpu_hours_next")
""")

md("""
## 2. 평가 틀 (walk-forward, 정책)

구간 합계는 세 가지로 낸다. **test 주(140–146)**, **홀드아웃 5주(147–180)**, **6주 전체**.
""")
frame = pick(v4, "def topk_per_day(")
old = 'POOLS = {"6주 (140–180)": WEEK_NAMES, "홀드아웃 5주 (147–180)": [n for n in WEEK_NAMES if n != "140–146"]}'
assert old in frame
frame = frame.replace(old, 'POOLS = {"test 주 (140–146)": [n for n in WEEK_NAMES if n == "140–146"],\n'
                           '         "홀드아웃 5주 (147–180)": [n for n in WEEK_NAMES if n != "140–146"], "6주 (140–180)": WEEK_NAMES}')
code(frame)

md("""
## 3. 모델 비교

모든 방법에 같은 정책 6개를 적용한다. 주 정책은 `같은 날 B1과 같은 개수`다.
Baseline 2는 v2 정의를 따른다. 오늘 sm_util이 가장 낮은 N개를 고르고, N = 그 주 학습 구간의 idle 비율 × 행 수다.
""")
code(r"""
record("E1", "기본", "Baseline 1: 오늘 idle", 0, fixed_pred=[W["b1k"] for W in WK])
b2 = []
for W in WK:
    rate = df.loc[df["day"] <= W["a"] - 2, "is_mostly_idle_next"].mean()
    b2.append(base2(W["wk"], rate)[1])
record("E1", "기본", "Baseline 2: sm_util 하위 N", 0, fixed_pred=b2)
record("E1", "기본", "Baseline 1′ (idle_ratio 점수)", 0, week_scores=[(W["s1c"], W["s1k"]) for W in WK])
MODELS = {"Logistic Regression": ("lr", BEST["lr"]), "SVM (cuML)": ("svm", BEST["svm"]),
          "Random Forest": ("rf", BEST["rf"]), "LightGBM 튜닝": ("lgbm", BEST["lgbm"]), "XGBoost 튜닝": ("xgb", BEST["xgb"])}
SC = {}
for name, (kind, p) in MODELS.items():
    t0 = time.time()
    SC[name] = walk_forward(kind, p)
    record("E1", "기본", name, 0, week_scores=SC[name])
    print(f"{name}: {time.time() - t0:.0f}초 (누적 {elapsed()})", flush=True)
FINAL = "최종: LGBM+XGB 평균"
SC[FINAL] = [((a1 + a2) / 2, (b1 + b2_) / 2) for (a1, b1), (a2, b2_) in zip(SC["LightGBM 튜닝"], SC["XGBoost 튜닝"])]
record("E1", "기본", FINAL, 0, week_scores=SC[FINAL])
res = pooled_view("E1")
res.to_csv(OUT / "pooled.csv", index=False)
pd.DataFrame(WEEKLY).to_csv(OUT / "weekly.csv", index=False)
""")

code(r"""
MAIN = "같은 날 B1과 같은 개수"
ORDER = ["Baseline 1: 오늘 idle", "Baseline 2: sm_util 하위 N", "Baseline 1′ (idle_ratio 점수)", *MODELS, FINAL]
cols = ["방법", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "회수 건수",
        "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]


def show(pol, pool):
    d = res[(res["구간"] == pool) & res["정책"].isin(["-", pol])].copy()
    d = d[~((d["정책"] == "-") & ~d["방법"].str.startswith("Baseline 1:") & ~d["방법"].str.startswith("Baseline 2"))]
    d["방법"] = pd.Categorical(d["방법"], ORDER)
    return d.sort_values("방법")[cols]


for pool in POOLS:
    print(f"\n=== 주 정책: {MAIN} | {pool} ===")
    display(show(MAIN, pool).round(4))
""")

code(r"""
# 참고: 직전 주에서 threshold를 고르는 정책 (v3 방식)
for pol in ["F1 최대 (직전 주)", "위험 맞춤 GPU-h (직전 주)"]:
    pool = "홀드아웃 5주 (147–180)"
    print(f"\n=== 참고 정책: {pol} | {pool} ===")
    display(show(pol, pool).round(4))
# 주별 가중 AP (점수 순서 품질). Baseline 1 행은 idle_ratio 점수의 AP
w1 = pd.DataFrame([r for r in WEEKLY if r["정책"] in ("-", "F1 최대 (직전 주)") and r["방법"] != "Baseline 2: sm_util 하위 N"])
w1 = w1[~(w1["방법"].str.startswith("Baseline 1:"))]
display(w1.pivot_table(index="방법", columns="주", values="AP").round(4))
""")

md("""
### 3.1 부트스트랩 (app 단위 95% 구간)

app을 복원 추출해 *방법 − Baseline 1*의 분포를 구한다. ▲/▼는 구간 전체가 0보다 위/아래라는 뜻이다.
""")
code(r"""
import re


def mark(s):
    m = re.match(r"([+-]?[\d.,e+-]+) \[([+-]?[\d.,e+-]+), ([+-]?[\d.,e+-]+)\]", str(s).replace(",", ""))
    if not m:
        return s
    lo, hi = float(m.group(2)), float(m.group(3))
    return s + (" ▲" if lo > 0 else " ▼" if hi < 0 else "")


boots = []
for pool, names in POOLS.items():
    idx = [i for i, W in enumerate(WK) if W["name"] in names]
    if not idx:
        continue
    pool_df = pd.concat([WK[i]["wk"] for i in idx])
    base = np.concatenate([WK[i]["b1k"] for i in idx])
    for pol in [MAIN, "위험 맞춤 GPU-h (직전 주)"]:
        preds = {m: np.concatenate([PRED_STORE[("E1", "기본", m, 0, pol)][i] for i in idx]) for m in [*MODELS, FINAL]}
        if pol == MAIN:
            preds = {"Baseline 2: sm_util 하위 N": np.concatenate([PRED_STORE[("E1", "기본", "Baseline 2: sm_util 하위 N", 0, "-")][i] for i in idx]), **preds}
        bt = boot_diff(pool_df, preds, base)
        bt.insert(0, "정책", pol); bt.insert(0, "구간", pool)
        boots.append(bt)
boots = pd.concat(boots, ignore_index=True)
boots.to_csv(OUT / "bootstrap.csv", index=False)
for pool in POOLS:
    print(f"\n=== {pool} ===")
    display(boots[boots["구간"] == pool].drop(columns="구간").map(mark))
print("누적", elapsed())
""")

code(r"""
# 주별 결과 (최종 모델, 주 정책) — 이득이 특정 주에 몰려 있는지
wk = pd.DataFrame(WEEKLY)
b1w = wk[wk["방법"] == "Baseline 1: 오늘 idle"].set_index("주")
f = wk[(wk["방법"] == FINAL) & (wk["정책"] == MAIN)].set_index("주")
weekly_final = pd.DataFrame({
    "B1 회수 GPU-h": b1w["회수 GPU-h"], "B1 잘못 회수 GPU-h": b1w["잘못 회수 GPU-h"], "B1 잘못 회수 건수": b1w["잘못 회수 건수"],
    "Δ회수 GPU-h": f["회수 GPU-h"] - b1w["회수 GPU-h"], "Δ잘못 회수 GPU-h": f["잘못 회수 GPU-h"] - b1w["잘못 회수 GPU-h"],
    "Δ잘못 회수 건수": f["잘못 회수 건수"] - b1w["잘못 회수 건수"], "ΔF1": f["F1"] - b1w["F1"]})
weekly_final["잘못 회수 GPU-h 변화율"] = weekly_final["Δ잘못 회수 GPU-h"] / weekly_final["B1 잘못 회수 GPU-h"]
weekly_final.to_csv(OUT / "weekly_final.csv")
display(weekly_final.round(3))
""")

md("""
## 4. 최종 모델의 주요 발견

### 4.1 Baseline 1과 무엇을 다르게 고르나

회수하는 개수는 B1과 같다. 따라서 차이는 **B1만 고른 app-day**와 **최종 모델만 고른 app-day**를 맞바꾼 데서 나온다.
홀드아웃 5주에서 두 집합의 특징을 비교한다.
""")
code(r"""
pool = "홀드아웃 5주 (147–180)"
idx = [i for i, W in enumerate(WK) if W["name"] in POOLS[pool]]
H = pd.concat([WK[i]["wk"] for i in idx]).copy()
H["b1"] = np.concatenate([WK[i]["b1k"] for i in idx]).astype(bool)
H["fin"] = np.concatenate([PRED_STORE[("E1", "기본", FINAL, 0, MAIN)][i] for i in idx]).astype(bool)
H["fin_score"] = np.concatenate([SC[FINAL][i][1] for i in idx])
grp = np.select([H["b1"] & H["fin"], H["b1"] & ~H["fin"], ~H["b1"] & H["fin"]],
                ["둘 다 회수", "B1만 회수 (최종 모델이 뺌)", "최종 모델만 회수 (새로 넣음)"], "둘 다 안 함")
H["구분"] = grp
swap = H.groupby("구분").agg(
    app_day=("day", "size"),
    내일_idle_비율=("is_mostly_idle_next", "mean"),
    오늘_idle_ratio_평균=("idle_ratio", "mean"),
    gpu_count_중앙값=("gpu_count", "median"),
    오늘_GPU_h_중앙값=("gpu_hours", "median"),
    내일_GPU_h_합=("gpu_hours_next", "sum"),
    내일_GPU_h_중앙값=("gpu_hours_next", "median"),
    잘못_회수_시_깨우는_GPU_h=("gpu_hours_next", lambda s: s[H.loc[s.index, "is_mostly_idle_next"] == 0].sum()),
    offline_비율=("workload", lambda s: (s.astype(str) == "offline").mean()),
)
swap.to_csv(OUT / "swap_holdout.csv")
display(swap.round(3))
""")

md("""
### 4.2 변수 기여도 — TreeSHAP (test 주, 최종 모델의 두 구성 모델)

test 주를 예측한 모델(`day ≤ 138`로 학습)을 다시 만들어 TreeSHAP을 계산한다. 단위는 로그 오즈다.
최종 점수는 두 확률의 평균이라 SHAP을 그대로 더할 수 없다. 그래서 모델별로 따로 구하고 묶음 단위로 비교한다.
""")
code(r"""
test_wk = df[df["day"].between(140, 146)]
tr = df[df["day"] <= 138]
Xt, yt, _ = split_xyw(test_wk)
shap_rows = {}
for name, kind in [("LightGBM 튜닝", "lgbm"), ("XGBoost 튜닝", "xgb")]:
    fm = fit_model(kind, BEST[kind], tr, device=DEVICES[0])
    Zt = fm.enc.transform(Xt)
    if kind == "xgb":
        bst = fm.model.get_booster(); bst.set_param({"device": f"cuda:{DEVICES[0]}"})
        contrib = bst.predict(xgb.DMatrix(Zt, enable_categorical=True), pred_contribs=True)
    else:
        contrib = np.asarray(fm.model.predict(Zt, pred_contrib=True))
    sd = pd.DataFrame(contrib[:, :-1], columns=Zt.columns)
    shap_rows[name] = sd.abs().mean()
imp = pd.DataFrame(shap_rows)
imp["묶음"] = imp.index.map(lambda c: FEAT_GROUP.get(c, "기타"))
imp = imp.sort_values("LightGBM 튜닝", ascending=False)
imp.to_csv(OUT / "shap_importance.csv")
display(imp.head(12).round(3))
gimp = imp.groupby("묶음")[["LightGBM 튜닝", "XGBoost 튜닝"]].sum().sort_values("LightGBM 튜닝", ascending=False)
display(gimp.round(3))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, name in zip(axes, ["LightGBM 튜닝", "XGBoost 튜닝"]):
    imp[name].sort_values().tail(12).plot.barh(ax=ax); ax.set_title(f"{name} — mean |SHAP| (test)")
fig.tight_layout(); fig.savefig(OUT / "shap_importance.png", dpi=130); plt.show()
""")

md("""
### 4.3 변수 묶음별 운영 이득 (v4 E3 결과 다시 보기)

SHAP 크기와 운영 이득은 같은 것이 아니다. v4에서 LightGBM으로 변수 묶음을 하나씩 더하거나 뺀 결과를 다시 보여준다.
주 정책, 홀드아웃 5주 기준이다.
""")
code(r"""
e3 = pd.read_csv(ROOT / "outputs" / "v4" / "e3_ablation.csv")
e3 = e3[(e3["구간"] == "홀드아웃 5주 (147–180)") & (e3["정책"] == MAIN) & (e3["workload"] == "전체")]
display(e3[["설정", "주별 AP 평균", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]].round(4))
""")

md("### 4.4 workload별 (주 정책)")
code(r"""
wl = pd.concat([pooled_view("E1", seg="online"), pooled_view("E1", seg="offline")])
wl.to_csv(OUT / "pooled_by_workload.csv", index=False)
for pool in list(POOLS)[1:2]:
    d = wl[(wl["구간"] == pool) & wl["정책"].isin(["-", MAIN]) & wl["방법"].isin(["Baseline 1: 오늘 idle", *MODELS, FINAL])]
    d = d[~((d["정책"] == "-") & (d["방법"] != "Baseline 1: 오늘 idle"))]
    print(f"=== {pool} ===")
    display(d[["workload", "방법", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "Δ회수 GPU-h vs B1",
               "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]].round(4))
""")

md("""
## 5. 실제 cold start — 5분 단위 시뮬레이션 (test 주)

일 단위 라벨 대신, 회수한 app이 **다음날 실제로 언제 처음 활동했는지**를 5분 표로 확인한다.
- 가정: day t+1 시작에 회수한다. 첫 활동 5분 창에서 cold start가 1건 생기고, 그날 나머지는 다시 올려 둔다.
- 회수 GPU-h = 첫 활동 전까지의 `gpu_count × 5분` 합이다. 활동이 없으면 하루 전체를 센다.
- 5분 표가 140–146일뿐이라 오늘 = 140–145인 행만 쓴다.
- 집계표는 `scripts/sim_5min_samecount.py`가 만든 `outputs/final/sim_5min_agg.parquet`를 쓴다. 없으면 여기서 만든다.
""")
code(r"""
sys.path.insert(0, str(ROOT / "scripts"))
from sim_5min_samecount import AGG, SQL, sim_table
if AGG.exists():
    agg = pd.read_parquet(AGG)
else:
    from gpu_reclaim.db import connect
    agg = connect(threads=16, memory_limit="64GB").sql(SQL).df()
    agg.to_parquet(AGG, index=False)
W0 = WK[0]; assert W0["name"] == "140–146"
m = W0["day"] <= 145
ts = pd.DataFrame({"app_id": W0["wk"]["app_id"].astype(str).values[m], "day": W0["day"][m],
                   "y": W0["yk"][m], "w": W0["wk_w"][m]})
a = agg.copy(); a["app_id"] = a["app_id"].astype(str); a["day"] = a["day"] - 1
ts = ts.merge(a, on=["app_id", "day"], how="left")
assert len(ts) == m.sum()
print(f"대상 {len(ts):,}행, 5분 기록 있음 {ts['gpu_h_5min'].notna().mean() * 100:.1f}%")
methods = {"Baseline 1: 오늘 idle": W0["b1k"][m].astype(bool)}
for name in [*MODELS, FINAL]:
    methods[f"{name} | {MAIN}"] = PRED_STORE[("E1", "기본", name, 0, MAIN)][0][m].astype(bool)
methods[f"{FINAL} | 위험 맞춤 GPU-h (직전 주)"] = PRED_STORE[("E1", "기본", FINAL, 0, "위험 맞춤 GPU-h (직전 주)")][0][m].astype(bool)
methods["(상한) 전부 회수"] = np.ones(len(ts), dtype=bool)
sim_res = sim_table(ts, methods)
sim_res.to_csv(OUT / "sim_5min_test.csv", index=False)
display(sim_res.round(3))
""")

md("""
## 6. 가치 — 운영 관점에서 얼마인가

홀드아웃 5주(34일)를 주 단위로 환산한다. `GPU 대수 환산` = 주당 GPU-h ÷ 168시간이다. 일주일 내내 비워 둘 수 있는 GPU 대수에 해당한다.
""")
code(r"""
pool = "홀드아웃 5주 (147–180)"
n_days = sum(W["b"] - W["a"] + 1 for W in WK if W["name"] in POOLS[pool])
per_week = 7 / n_days
total_next = sum(W["wk_w"].sum() for W in WK if W["name"] in POOLS[pool])
r = res[(res["구간"] == pool)].set_index(["방법", "정책"])
b = r.loc[("Baseline 1: 오늘 idle", "-")]
f = r.loc[(FINAL, MAIN)]
value = pd.DataFrame({
    "Baseline 1": [b["회수 GPU-h"], b["잘못 회수 GPU-h"], b["잘못 회수 건수"]],
    "최종 모델": [f["회수 GPU-h"], f["잘못 회수 GPU-h"], f["잘못 회수 건수"]],
}, index=["회수 GPU-h (내일 idle인 app)", "잘못 회수로 깨우는 GPU-h", "잘못 회수 건수"]) * per_week
value["차이"] = value["최종 모델"] - value["Baseline 1"]
value["변화율"] = value["차이"] / value["Baseline 1"]
value.loc["GPU 대수 환산 (회수)"] = [b["회수 GPU-h"] * per_week / 168, f["회수 GPU-h"] * per_week / 168, np.nan, np.nan]
value["차이"] = value["최종 모델"] - value["Baseline 1"]
print(f"주당 환산 (홀드아웃 {n_days}일 기준). 이 기간 전체 내일 GPU-h 중 B1 회수 비중: {b['회수 GPU-h'] / total_next * 100:.1f}%")
display(value.round(3))
value.to_csv(OUT / "value_per_week.csv")
""")

md("## 7. 요약")
code(r"""
rows = []
for pool in POOLS:
    for meth, pol in [("Baseline 1: 오늘 idle", "-"), ("Baseline 2: sm_util 하위 N", "-")] + [(m_, MAIN) for m_ in [*MODELS, FINAL]]:
        q = res[(res["구간"] == pool) & (res["방법"] == meth) & (res["정책"] == pol)]
        if len(q):
            q = q.iloc[0]
            rows.append({"구간": pool, "방법": meth, "F1": q["F1"], "회수 GPU-h": q["회수 GPU-h"], "잘못 회수 GPU-h": q["잘못 회수 GPU-h"],
                         "잘못 회수 건수": q["잘못 회수 건수"], "Δ잘못 회수 GPU-h vs B1": q["Δ잘못 회수 GPU-h vs B1"],
                         "Δ잘못 회수 건수 vs B1": q["Δ잘못 회수 건수 vs B1"]})
summary = pd.DataFrame(rows)
summary.to_csv(OUT / "summary.csv", index=False)
display(summary.round(4))

# 점수 저장 (주별 행 순서는 df에서 그 주를 고른 순서)
keys = pd.concat([pd.DataFrame({"주": W["name"], "row_index": W["wk"].index, "app_id": W["wk"]["app_id"].astype(str).values,
                                "day": W["day"], "y": W["yk"], "w": W["wk_w"], "b1": W["b1k"]}) for W in WK], ignore_index=True)
sc_cols = {n: np.concatenate([SC[n][i][1] for i in range(len(WK))]).astype("float32") for n in SC}
pd.concat([keys, pd.DataFrame(sc_cols)], axis=1).to_parquet(OUT / "scores.parquet", index=False)
print("저장 위치:", OUT, "| 전체", elapsed())
""")


def write():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "gpu-reclaim-gpu", "display_name": "Python (gpu-reclaim GPU)", "language": "python"}
    nbf.write(nb, OUT)
    print("written", OUT, len(cells), "cells")


if __name__ == "__main__":
    write()
