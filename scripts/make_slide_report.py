"""슬라이드 제작용 통합 보고서의 그림·표·원본값을 만든다.

입력: outputs/final/* (최종 노트북), outputs/v4/* (seed·변수 묶음·학습 기간), data/processed/app_day_v2_max181.parquet
출력: reports/slide_report/figures/*.png, reports/slide_report/data/*.csv, reports/slide_report/data/numbers.json
보고서 본문(reports/slide_report/README.md)은 사람이 이 숫자를 보고 쓴다.
사용: env-gpu/bin/python scripts/make_slide_report.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIN, V4 = ROOT / "outputs" / "final", ROOT / "outputs" / "v4"
REP = ROOT / "reports" / "slide_report"
FIG, DAT = REP / "figures", REP / "data"
FIG.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager  # noqa: E402
_KO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"   # .ttc는 자동 등록이 안 되어 직접 등록
font_manager.fontManager.addfont(_KO)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=_KO).get_name(), "axes.unicode_minus": False, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
C_B1, C_RULE, C_FIN, C_OTHER, C_BAD = "#8c8c8c", "#b0b0b0", "#1f6fb4", "#9ec3e6", "#d1495b"
HOLD, TEST, SIX = "홀드아웃 5주 (147–180)", "test 주 (140–146)", "6주 (140–180)"
FINAL, MAIN, RISK = "최종: LGBM+XGB 평균", "같은 날 B1과 같은 개수", "위험 맞춤 GPU-h (직전 주)"
B1 = "Baseline 1: 오늘 idle"
SHORT = {B1: "Baseline 1", "Baseline 2: sm_util 하위 N": "Baseline 2", "Baseline 1′ (idle_ratio 점수)": "Baseline 1′",
         "Logistic Regression": "LR", "SVM (cuML)": "SVM", "Random Forest": "RF", "LightGBM 튜닝": "LightGBM",
         "XGBoost 튜닝": "XGBoost", FINAL: "최종 (LGBM+XGB)"}
MODELS = ["Logistic Regression", "SVM (cuML)", "Random Forest", "LightGBM 튜닝", "XGBoost 튜닝", FINAL]
NUM = {}


def save(df, name, index=False):
    df.to_csv(DAT / f"{name}.csv", index=index, encoding="utf-8-sig")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def parse_ci(s):
    import re
    g = re.match(r"([+-]?[\d.,e+-]+) \[([+-]?[\d.,e+-]+), ([+-]?[\d.,e+-]+)\]", str(s)).groups()
    return tuple(float(x.replace(",", "")) for x in g)


# ------------------------------------------------------------------ 데이터
df = pd.read_parquet(ROOT / "data" / "processed" / "app_day_v2_max181.parquet")
splits = {"train": (0, 132), "val": (133, 139), "test": (140, 146), "holdout": (147, 180)}
ds = pd.DataFrame([{"구간": k, "day": f"{a}–{b}", "행 (app-day)": int(df["day"].between(a, b).sum()),
                    "app 수": int(df.loc[df["day"].between(a, b), "app_id"].nunique()),
                    "내일 idle 비율 (건수)": df.loc[df["day"].between(a, b), "is_mostly_idle_next"].mean(),
                    "내일 idle 비율 (GPU-h 가중)": np.average(df.loc[df["day"].between(a, b), "is_mostly_idle_next"],
                                                         weights=df.loc[df["day"].between(a, b), "gpu_hours_next"])}
                   for k, (a, b) in splits.items()])
ds.loc[len(ds)] = {"구간": "전체", "day": "0–180", "행 (app-day)": len(df), "app 수": df["app_id"].nunique(),
                   "내일 idle 비율 (건수)": df["is_mostly_idle_next"].mean(),
                   "내일 idle 비율 (GPU-h 가중)": np.average(df["is_mostly_idle_next"], weights=df["gpu_hours_next"])}
save(ds, "01_dataset_splits")
NUM["dataset_rows"], NUM["dataset_apps"] = len(df), int(df["app_id"].nunique())
NUM["workload_share_rows"] = df["workload"].astype(str).value_counts(normalize=True).round(4).to_dict()

# 상태 전이 (홀드아웃): 오늘 idle × 내일 idle
H = df[df["day"].between(147, 180)]
tr = H.groupby(["is_mostly_idle", "is_mostly_idle_next"]).agg(행=("day", "size"), 내일_GPU_h=("gpu_hours_next", "sum")).reset_index()
tr["상태"] = tr.apply(lambda r: f"{'idle' if r.is_mostly_idle else 'busy'} → {'idle' if r.is_mostly_idle_next else 'busy'}", axis=1)
tr["행 비율"] = tr["행"] / tr["행"].sum()
tr["GPU-h 비율"] = tr["내일_GPU_h"] / tr["내일_GPU_h"].sum()
save(tr[["상태", "행", "행 비율", "내일_GPU_h", "GPU-h 비율"]], "02_state_transition_holdout")
idle_today = H[H["is_mostly_idle"] == 1]
NUM["p_idle_next_given_idle_today_rows"] = float(idle_today["is_mostly_idle_next"].mean())

# ------------------------------------------------------------------ 모델 비교
pooled = pd.read_csv(FIN / "pooled.csv")
pooled = pooled[pooled["workload"] == "전체"]
keep = []
for pool in [TEST, HOLD, SIX]:
    for meth in SHORT:
        for pol in (["-"] if meth.startswith("Baseline 1:") or meth.startswith("Baseline 2") else [MAIN, RISK, "F1 최대 (직전 주)"]):
            q = pooled[(pooled["구간"] == pool) & (pooled["방법"] == meth) & (pooled["정책"] == pol)]
            if len(q):
                keep.append(q.iloc[0])
cmp_ = pd.DataFrame(keep)[["구간", "방법", "정책", "F1", "precision", "recall", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수",
                           "회수 건수", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]]
save(cmp_, "03_model_comparison_all")
main_tbl = cmp_[cmp_["정책"].isin(["-", MAIN])].copy()
save(main_tbl, "04_model_comparison_main_policy")
b1h = cmp_[(cmp_["구간"] == HOLD) & (cmp_["방법"] == B1)].iloc[0]
NUM["B1_holdout"] = {k: float(b1h[k]) for k in ["F1", "precision", "recall", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "회수 건수"]}

boot = pd.read_csv(FIN / "bootstrap.csv")
rows = []
for _, r in boot.iterrows():
    for c in ["Δ회수 GPU-h", "Δ잘못 회수 GPU-h", "Δ잘못 회수 건수", "ΔF1"]:
        m, lo, hi = parse_ci(r[c])
        rows.append({"구간": r["구간"], "정책": r["정책"], "방법": r["방법"], "지표": c, "평균": m, "하한 2.5%": lo, "상한 97.5%": hi,
                     "판정": "유의하게 증가" if lo > 0 else ("유의하게 감소" if hi < 0 else "차이 없음")})
bootl = pd.DataFrame(rows)
save(bootl, "05_bootstrap_ci")
fb = bootl[(bootl["구간"] == HOLD) & (bootl["정책"] == MAIN) & (bootl["방법"] == FINAL)].set_index("지표")
NUM["final_holdout_samecount_ci"] = fb[["평균", "하한 2.5%", "상한 97.5%"]].to_dict("index")

# 그림: 홀드아웃, 주 정책, 모델별 Δ잘못 회수 GPU-h / Δ건수 (CI)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric, unit in [(axes[0], "Δ잘못 회수 GPU-h", 1000), (axes[1], "Δ잘못 회수 건수", 1)]:
    d = bootl[(bootl["구간"] == HOLD) & (bootl["정책"] == MAIN) & (bootl["지표"] == metric)
              & bootl["방법"].isin(MODELS)].set_index("방법").loc[MODELS]
    y = np.arange(len(d))
    col = [C_FIN if m == FINAL else C_OTHER for m in d.index]
    ax.barh(y, d["평균"] / unit, color=col)
    ax.errorbar(d["평균"] / unit, y, xerr=[(d["평균"] - d["하한 2.5%"]) / unit, (d["상한 97.5%"] - d["평균"]) / unit],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, [SHORT[m] for m in d.index]); ax.invert_yaxis()
    ax.set_xlabel(("B1 대비 잘못 회수 GPU-h 차이 (천 GPU-h)" if unit == 1000 else "B1 대비 잘못 회수 건수 차이"))
axes[0].set_title("같은 개수를 회수할 때 B1 대비 위험 (홀드아웃 5주, 95% 구간)", loc="left", fontsize=10)
savefig(fig, "fig04_model_comparison_ci")

# ------------------------------------------------------------------ 회수 개수 비율 곡선 (저장 점수로 계산, 재학습 없음)
sc = pd.read_parquet(FIN / "scores.parquet")
sc["idle_ratio"] = df.loc[sc["row_index"].to_numpy(), "idle_ratio"].to_numpy()
assert (df.loc[sc["row_index"].to_numpy(), "day"].to_numpy() == sc["day"].to_numpy()).all()


def topk(g, col, r):
    pred = np.zeros(len(g), bool); s = g[col].to_numpy(); d = g["day"].to_numpy(); b = g["b1"].to_numpy()
    for dd in np.unique(d):
        idx = np.flatnonzero(d == dd); k = int(round(r * b[idx].sum()))
        if k:
            pred[idx[np.argsort(-s[idx], kind="stable")[:k]]] = True
    return pred


RATIOS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
CURVE_M = {"단순 규칙 (idle_ratio 상위)": "idle_ratio", "LightGBM": "LightGBM 튜닝", "XGBoost": "XGBoost 튜닝", "최종 (LGBM+XGB)": FINAL}
hold_weeks = ["147–153", "154–160", "161–167", "168–174", "175–180"]
crow, PREDS = [], {}
for pool, wks in [(TEST, ["140–146"]), (HOLD, hold_weeks)]:
    g = sc[sc["주"].isin(wks)]
    y, w = g["y"].to_numpy(), g["w"].to_numpy().astype(float)
    for r in RATIOS:
        for lab, col in CURVE_M.items():
            p = topk(g, col, r)
            PREDS[(pool, r, lab)] = p
            crow.append({"구간": pool, "B1 개수 대비": r, "방법": lab, "회수 건수": int(p.sum()),
                         "회수 GPU-h": (w * y)[p].sum(), "잘못 회수 GPU-h": (w * (1 - y))[p].sum(), "잘못 회수 건수": int((1 - y)[p].sum())})
curve = pd.DataFrame(crow)
b1ref = {pool: (sc[sc["주"].isin(wks)]) for pool, wks in [(TEST, ["140–146"]), (HOLD, hold_weeks)]}
for pool, g in b1ref.items():
    y, w, b = g["y"].to_numpy(), g["w"].to_numpy().astype(float), g["b1"].to_numpy().astype(bool)
    m = curve["구간"] == pool
    curve.loc[m, "B1 회수 GPU-h"] = (w * y)[b].sum(); curve.loc[m, "B1 잘못 회수 GPU-h"] = (w * (1 - y))[b].sum()
    curve.loc[m, "B1 잘못 회수 건수"] = int((1 - y)[b].sum())
save(curve, "06_count_ratio_curve")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
d = curve[curve["구간"] == HOLD]
for lab, col, ls in [("단순 규칙 (idle_ratio 상위)", C_B1, "--"), ("최종 (LGBM+XGB)", C_FIN, "-")]:
    q = d[d["방법"] == lab]
    axes[0].plot(q["B1 개수 대비"] * 100, q["잘못 회수 GPU-h"] / 1e3, ls, marker="o", color=col, label=lab)
    axes[1].plot(q["B1 개수 대비"] * 100, q["회수 GPU-h"] / 1e6, ls, marker="o", color=col, label=lab)
axes[0].set_ylabel("잘못 회수 GPU-h (천)"); axes[1].set_ylabel("회수 GPU-h (백만)")
for ax in axes:
    ax.set_xlabel("하루 회수 개수 (B1 개수 대비 %)"); ax.axvline(100, color="k", lw=0.6, ls=":")
axes[0].set_title("같은 개수를 회수할 때 위험: 모델이 항상 낮음", loc="left", fontsize=10)
axes[1].set_title("같은 개수를 회수할 때 회수량: 거의 같음", loc="left", fontsize=10)
axes[0].legend(frameon=False)
savefig(fig, "fig06_count_ratio_curve")

# 비율별 부트스트랩: 최종 모델 vs (a) B1, (b) 같은 비율의 단순 규칙
g = sc[sc["주"].isin(hold_weeks)]
y, w = g["y"].to_numpy().astype(int), g["w"].to_numpy().astype(float)
codes, uniq = pd.factorize(g["app_id"].astype(str)); n_app = len(uniq)
rng = np.random.default_rng(0)
CNT = rng.multinomial(n_app, np.full(n_app, 1 / n_app), size=2000).astype("float32")


def per_app(pred):
    pred = np.asarray(pred).astype(int)
    cols = np.stack([w * (pred == 1) * (y == 1), w * (pred == 1) * (y == 0), w * (pred == 0) * (y == 1),
                     ((pred == 1) & (y == 0)).astype(float)], axis=1)
    M = np.zeros((n_app, 4)); np.add.at(M, codes, cols); return M.astype("float32")


def boot(pred, base):
    S, SB = CNT @ per_app(pred), CNT @ per_app(base)
    f1 = lambda T: 2 * T[:, 0] / (2 * T[:, 0] + T[:, 1] + T[:, 2])
    out = {}
    for k, v in {"Δ회수 GPU-h": S[:, 0] - SB[:, 0], "Δ잘못 회수 GPU-h": S[:, 1] - SB[:, 1],
                 "Δ잘못 회수 건수": S[:, 3] - SB[:, 3], "ΔF1": f1(S) - f1(SB)}.items():
        lo, hi = np.percentile(v, [2.5, 97.5]); out[k] = (float(v.mean()), float(lo), float(hi))
    return out


b1_pred = g["b1"].to_numpy().astype(bool)
brow = []
for r in [0.85, 0.90, 0.95, 1.00]:
    fin = PREDS[(HOLD, r, "최종 (LGBM+XGB)")]
    for ref_name, ref in [("B1 (100%)", b1_pred), (f"단순 규칙 ({int(r * 100)}%)", PREDS[(HOLD, r, "단순 규칙 (idle_ratio 상위)")])]:
        for k, (m, lo, hi) in boot(fin, ref).items():
            brow.append({"B1 개수 대비": r, "비교 대상": ref_name, "지표": k, "평균": m, "하한 2.5%": lo, "상한 97.5%": hi,
                         "판정": "유의하게 증가" if lo > 0 else ("유의하게 감소" if hi < 0 else "차이 없음")})
cboot = pd.DataFrame(brow)
save(cboot, "07_count_ratio_bootstrap_holdout")

# 감소량의 app 집중도 (최종 vs 같은 비율 단순 규칙, 그리고 100%에서 vs B1)
conc = []
for r, ref_lab in [(1.00, "B1"), (0.90, "단순 규칙 90%"), (0.90, "B1")]:
    fin = PREDS[(HOLD, r, "최종 (LGBM+XGB)")]
    ref = b1_pred if ref_lab == "B1" else PREDS[(HOLD, r, "단순 규칙 (idle_ratio 상위)")]
    dcon = w * (y == 0) * (fin.astype(int) - ref.astype(int))
    a = pd.Series(dcon).groupby(codes).sum().sort_values()
    tot = a.sum()
    conc.append({"최종 모델 비율": r, "비교 대상": ref_lab, "Δ잘못 회수 GPU-h 합": tot,
                 "상위 1 app 비중": a.iloc[0] / tot, "상위 2 app 비중": a.iloc[:2].sum() / tot,
                 "상위 5 app 비중": a.iloc[:5].sum() / tot, "상위 10 app 비중": a.iloc[:10].sum() / tot,
                 "기여 app 수(≠0)": int((a != 0).sum())})
conc = pd.DataFrame(conc)
save(conc, "08_concentration_holdout")

# ------------------------------------------------------------------ 주별 (최종, 주 정책)
wf = pd.read_csv(FIN / "weekly_final.csv")
save(wf, "09_weekly_final_samecount")
fig, ax = plt.subplots(figsize=(8, 3.6))
col = [C_FIN if v < 0 else C_BAD for v in wf["Δ잘못 회수 GPU-h"]]
ax.bar(wf["주"], wf["Δ잘못 회수 GPU-h"] / 1e3, color=col)
for i, (v, rr) in enumerate(zip(wf["Δ잘못 회수 GPU-h"], wf["잘못 회수 GPU-h 변화율"])):
    ax.text(i, v / 1e3, f"{rr * 100:+.0f}%", ha="center", va="top" if v < 0 else "bottom", fontsize=9)
ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("B1 대비 잘못 회수 GPU-h (천)")
ax.set_title("주별 효과 (최종 모델, 같은 개수 회수): 모든 주 감소, 크기는 주마다 다름", loc="left", fontsize=10)
savefig(fig, "fig09_weekly_effect")

# ------------------------------------------------------------------ 무엇을 다르게 고르나
sw = pd.read_csv(FIN / "swap_holdout.csv")
save(sw, "10_swap_holdout")
s2 = sw.set_index("구분").loc[["B1만 회수 (최종 모델이 뺌)", "최종 모델만 회수 (새로 넣음)"]]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
labels = ["최종 모델이 뺀 것\n(B1만 회수)", "최종 모델이 넣은 것\n(모델만 회수)"]
for ax, c, t in [(axes[0], "내일_idle_비율", "내일 실제 idle 비율"), (axes[1], "내일_GPU_h_중앙값", "내일 GPU-h 중앙값"),
                 (axes[2], "잘못_회수_시_깨우는_GPU_h", "잘못 회수 시 깨우는 GPU-h 합 (천)")]:
    v = s2[c] / (1000 if "깨우는" in c else 1)
    ax.bar(labels, v, color=[C_BAD, C_FIN]); ax.set_title(t, fontsize=10)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.2f}" if x < 10 else f"{x:,.0f}", ha="center", va="bottom", fontsize=9)
fig.suptitle(f"B1과 다르게 고른 app-day (홀드아웃, 각 {int(s2['app_day'].iloc[0]):,}개 / {int(s2['app_day'].iloc[1]):,}개)", x=0.01, ha="left", fontsize=10)
savefig(fig, "fig10_swap")

shap = pd.read_csv(FIN / "shap_importance.csv").rename(columns={"Unnamed: 0": "변수"})
save(shap, "11_shap_feature")
gshap = shap.groupby("묶음")[["LightGBM 튜닝", "XGBoost 튜닝"]].sum().sort_values("LightGBM 튜닝", ascending=False)
save(gshap, "12_shap_group", index=True)
fig, ax = plt.subplots(figsize=(7, 3.6))
t = shap.head(10).iloc[::-1]
yy = np.arange(len(t))
ax.barh(yy - 0.2, t["LightGBM 튜닝"], 0.4, label="LightGBM", color=C_FIN)
ax.barh(yy + 0.2, t["XGBoost 튜닝"], 0.4, label="XGBoost", color=C_OTHER)
ax.set_yticks(yy, [f"{v} ({gname})" for v, gname in zip(t["변수"], t["묶음"])])
ax.set_xlabel("평균 |SHAP| (로그 오즈)"); ax.legend(frameon=False)
ax.set_title("변수 기여도 상위 10 (test 주)", loc="left", fontsize=10)
savefig(fig, "fig11_shap")

e3 = pd.read_csv(V4 / "e3_ablation.csv")
e3 = e3[(e3["구간"] == "홀드아웃 5주 (147–180)") & (e3["정책"] == MAIN) & (e3["workload"] == "전체")]
e3 = e3[["설정", "주별 AP 평균", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]]
save(e3, "13_feature_group_ablation_v4")
add = e3[e3["설정"].str.startswith("더하기") | (e3["설정"] == "전체 (23개)")].copy()
add["단계"] = add["설정"].map(lambda s: "+ " + s.split(" + ")[-1].replace("더하기: ", "") if s.startswith("더하기") else "+ 달력 (= 전체)")
add.loc[add["설정"] == "더하기: 오늘 idle", "단계"] = "오늘 idle만"
STEP_ORDER = ["오늘 idle만", "+ 이력", "+ 오늘 사용량", "+ 규모", "+ 메타", "+ 달력 (= 전체)"]
add["단계"] = pd.Categorical(add["단계"], STEP_ORDER); add = add.sort_values("단계"); add["단계"] = add["단계"].astype(str)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].plot(add["단계"], add["주별 AP 평균"], marker="o", color=C_FIN); axes[0].set_title("점수 순서 품질 (주별 가중 AP 평균)", loc="left", fontsize=10)
axes[1].bar(add["단계"], add["Δ잘못 회수 GPU-h vs B1"] / 1e3, color=[C_FIN if v < 0 else C_BAD for v in add["Δ잘못 회수 GPU-h vs B1"]])
axes[1].axhline(0, color="k", lw=0.8); axes[1].set_title("운영 이득: B1 대비 잘못 회수 GPU-h (천, 같은 개수)", loc="left", fontsize=10)
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
fig.suptitle("변수 묶음을 차례로 더할 때 (LightGBM, 홀드아웃): 이력은 AP를, 규모는 운영 이득을 만든다", x=0.01, ha="left", fontsize=10)
savefig(fig, "fig13_feature_groups")

# ------------------------------------------------------------------ 견고성: seed, 학습 기간
e2 = pd.read_csv(V4 / "e2_seeds_pooled.csv")
e2 = e2[(e2["workload"] == "전체") & e2["정책"].isin([MAIN, RISK])]
e2 = e2[["구간", "방법", "정책", "seed", "F1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]]
save(e2, "14_seeds_v4")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
q = e2[(e2["구간"] == "홀드아웃 5주 (147–180)") & (e2["정책"] == MAIN)]
for ax, c, t in [(axes[0], "Δ잘못 회수 GPU-h vs B1", "B1 대비 잘못 회수 GPU-h (천)"), (axes[1], "Δ잘못 회수 건수 vs B1", "B1 대비 잘못 회수 건수")]:
    for i, m in enumerate(["LightGBM 튜닝", "XGBoost 튜닝"]):
        v = q[q["방법"] == m][c] / (1000 if "GPU-h" in c else 1)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, color=C_FIN, s=30)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks([0, 1], ["LightGBM", "XGBoost"]); ax.set_xlim(-0.5, 1.5); ax.set_title(t, fontsize=10)
fig.suptitle("seed 5개 (홀드아웃, 같은 개수 회수): 점 하나 = seed 하나", x=0.01, ha="left", fontsize=10)
savefig(fig, "fig14_seeds")

e4 = pd.read_csv(V4 / "e4_window.csv")
e4 = e4[(e4["구간"] == "홀드아웃 5주 (147–180)") & (e4["workload"] == "전체") & e4["정책"].isin([MAIN, RISK])]
save(e4[["방법", "설정", "정책", "주별 AP 평균", "F1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]], "15_train_window_v4")

# ------------------------------------------------------------------ 5분 시뮬레이션, workload, 가치
sim = pd.read_csv(FIN / "sim_5min_test.csv")
save(sim, "16_sim_5min_test")
b = sim.iloc[0]
NUM["sim_B1"] = {k: float(b[k]) for k in ["회수 app-day", "cold start 건수", "cold start 비율", "첫 활동까지 시간 중앙값(h)",
                                         "회수 GPU-h (첫 활동 전까지)", "일 단위 잘못 회수 건수"]}
fig, ax = plt.subplots(figsize=(6.5, 2.4))
n, cold, lab = b["회수 app-day"], b["cold start 건수"], b["일 단위 잘못 회수 건수"]
parts = [("일 단위로도 '잘못 회수'", lab, C_BAD), ("일 단위로는 맞지만 다음날 한 번 이상 활동", cold - lab, "#f0a35e"),
         ("다음날 활동 없음", n - cold, C_FIN)]
left = 0
for name, v, c in parts:
    ax.barh(0, v, left=left, color=c); ax.text(left + v / 2, 0, f"{v / n * 100:.0f}%", ha="center", va="center", color="white", fontsize=10)
    left += v
ax.set_yticks([]); ax.set_xlim(0, n); ax.set_xticks([])
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [p[0] for p in parts], frameon=False,
          loc="upper left", bbox_to_anchor=(0, -0.08), ncol=1, fontsize=9)
ax.set_title(f"B1이 회수한 app-day {int(n):,}개의 다음날 (5분 단위, test 주 오늘 140–145)", loc="left", fontsize=10)
savefig(fig, "fig16_coldstart_5min")

wl = pd.read_csv(FIN / "pooled_by_workload.csv")
wl = wl[(wl["구간"] == HOLD) & (((wl["방법"] == B1) & (wl["정책"] == "-")) | ((wl["방법"].isin(MODELS)) & (wl["정책"] == MAIN)))]
save(wl[["workload", "방법", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1",
         "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]], "17_workload_holdout")

val = pd.read_csv(FIN / "value_per_week.csv").rename(columns={"Unnamed: 0": "항목"})
save(val, "18_value_per_week")

# 회수 개수 90% 운영 시 주당 환산
n_days = 34
pw = 7 / n_days
q90 = curve[(curve["구간"] == HOLD) & (curve["B1 개수 대비"] == 0.90)].set_index("방법")
q100 = curve[(curve["구간"] == HOLD) & (curve["B1 개수 대비"] == 1.00)].set_index("방법")
v90 = pd.DataFrame({
    "B1 (100%)": [b1h["회수 GPU-h"] * pw, b1h["잘못 회수 GPU-h"] * pw, b1h["잘못 회수 건수"] * pw],
    "최종 모델 (100%)": [q100.loc["최종 (LGBM+XGB)", c] * pw for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]],
    "단순 규칙 (90%)": [q90.loc["단순 규칙 (idle_ratio 상위)", c] * pw for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]],
    "최종 모델 (90%)": [q90.loc["최종 (LGBM+XGB)", c] * pw for c in ["회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]],
}, index=["회수 GPU-h / 주", "잘못 회수 GPU-h / 주", "잘못 회수 건수 / 주"])
v90.loc["회수 GPU 대수 환산 (÷168h)"] = v90.loc["회수 GPU-h / 주"] / 168
save(v90, "19_value_per_week_by_operating_point", index=True)

# v2(팀원 노트북) test 주 결과 — v2 노트북 출력값 그대로
v2 = pd.DataFrame([
    {"방법": "Baseline 1 (v2)", "F1": 0.8797, "precision": 0.8696, "recall": 0.8901, "회수 GPU-h": 661692.75, "잘못 회수 GPU-h": 99261.20},
    {"방법": "LightGBM (v2)", "F1": 0.8961, "precision": 0.9269, "recall": 0.8672, "회수 GPU-h": 644686.63, "잘못 회수 GPU-h": 50821.82},
    {"방법": "XGBoost (v2)", "F1": 0.8946, "precision": 0.9118, "recall": 0.8779, "회수 GPU-h": 652675.19, "잘못 회수 GPU-h": 63120.92},
    {"방법": "Logistic Regression (v2)", "F1": 0.8951, "precision": 0.9180, "recall": 0.8732, "회수 GPU-h": 649158.94, "잘못 회수 GPU-h": 57955.41},
    {"방법": "SVM (v2)", "F1": 0.7827, "precision": 0.9516, "recall": 0.6647, "회수 GPU-h": 494178.56, "잘못 회수 GPU-h": 25120.50},
])
save(v2, "20_v2_test_week_reference")

json.dump(NUM, open(DAT / "numbers.json", "w"), ensure_ascii=False, indent=1, default=float)
print("figures:", sorted(p.name for p in FIG.glob("*.png")))
print("data:", sorted(p.name for p in DAT.glob("*.csv")))
