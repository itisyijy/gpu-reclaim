"""v3·v4 결과 CSV를 모아 reports/overnight_report.md를 만든다 (수치 정리만, 해석은 사람이 덧붙인다).

없는 파일은 건너뛰고 그 사실을 적는다. 사용: env-gpu/bin/python scripts/make_report.py
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V3, V4 = ROOT / "outputs" / "v3", ROOT / "outputs" / "v4"
OUT = ROOT / "reports" / "overnight_report.md"
OUT.parent.mkdir(exist_ok=True)
L = []


def h(t, lv=2):
    L.append("\n" + "#" * lv + " " + t + "\n")


def p(t=""):
    L.append(t)


def fmt(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.4f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def table(df, index=False):
    if df is None or len(df) == 0:
        p("(결과 없음)")
        return
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    p("| " + " | ".join(cols) + " |")
    p("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in d.iterrows():
        p("| " + " | ".join(fmt(v).replace("|", "/") for v in r.values) + " |")
    p()


def read(path, **kw):
    if not path.exists():
        p(f"> `{path.relative_to(ROOT)}` 없음 — 이 부분은 실행되지 않았거나 실패함")
        return None
    return pd.read_csv(path, **kw)


CI = re.compile(r"([+-]?[\d.,e+-]+) \[([+-]?[\d.,e+-]+), ([+-]?[\d.,e+-]+)\]")


def ci_parse(s):
    m = CI.match(str(s).replace(",", ""))
    return tuple(float(x) for x in m.groups()) if m else (np.nan,) * 3


def ci_mark(s):
    m, lo, hi = ci_parse(s)
    if np.isnan(m):
        return s
    tag = " ▲" if lo > 0 else (" ▼" if hi < 0 else "")
    return f"{s}{tag}"


p(f"# 야간 실험 보고서 (자동 생성 {datetime.now():%Y-%m-%d %H:%M})")
p()
p("Baseline 1(B1) = '오늘 idle이면 내일도 idle'. Δ는 모두 *방법 − B1*. 부트스트랩 구간 뒤 ▲/▼는 95% 구간이 0보다 위/아래.")
p("'잘못 회수 건수' = 회수했는데 다음날 일한 app-day 수 = cold start 사건 수.")

h("0. 실행 기록")
for name, f in [("v3", V3 / "run.log"), ("v4", V4 / "run.log")]:
    if f.exists():
        p(f"**{name}**")
        p("```")
        p(f.read_text().strip())
        p("```")

# ---------------------------------------------------------------- v3
h("1. v3 — test 주(140–146)와 홀드아웃(147–180)")
s = read(V3 / "summary.csv")
if s is not None:
    p("정책: 위험 맞춤 (GPU-h) — val/직전 주에서 B1과 같은 잘못 회수 GPU-h 비율이 되도록 threshold 선택")
    table(s)

h("1.1 test 결과 (정책별)", 3)
t = read(V3 / "test_results.csv")
if t is not None:
    cols = ["방법", "정책", "AP", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수",
            "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]
    table(t[t["정책"].isin(["-", "위험 맞춤 (GPU-h)"])][cols])
m = read(V3 / "test_matched_risk.csv")
if m is not None:
    p("사후 맞춤 (test 곡선에서 B1과 같은 잘못 회수량일 때 회수 GPU-h)")
    table(m)
for pol in ["위험_맞춤_(GPU-h)", "F1_최대"]:
    b = read(V3 / f"test_bootstrap_{pol}.csv")
    if b is not None:
        p(f"test 부트스트랩 (app 단위) — 정책 {pol.replace('_', ' ')}")
        table(b.map(ci_mark) if hasattr(b, "map") else b)

h("1.2 홀드아웃 합계 (주 단위 재학습)", 3)
hp = read(V3 / "holdout_pooled.csv")
if hp is not None:
    cols = ["방법", "정책", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수",
            "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]
    table(hp[hp["정책"].isin(["-", "위험 맞춤 (GPU-h)", "위험 맞춤 (건수)"])][cols])
for pol in ["위험_맞춤_(GPU-h)", "F1_최대"]:
    b = read(V3 / f"holdout_bootstrap_{pol}.csv")
    if b is not None:
        p(f"홀드아웃 부트스트랩 — 정책 {pol.replace('_', ' ')}")
        table(b.map(ci_mark))
hw = read(V3 / "holdout_weekly.csv")
if hw is not None:
    p("주별 가중 F1 (정책 위험 맞춤 GPU-h)")
    table(hw[hw["정책"].isin(["-", "위험 맞춤 (GPU-h)"])].pivot_table(index="방법", columns="주", values="F1"), index=True)

h("1.3 workload별", 3)
wl = read(V3 / "by_workload.csv")
if wl is not None:
    table(wl[["구간", "workload", "행", "방법", "F1", "회수 GPU-h", "잘못 회수 GPU-h", "잘못 회수 건수"]])

h("1.4 5분 단위 시뮬레이션 (test, 오늘 140–145)", 3)
sim = read(V3 / "sim_5min_test.csv")
if sim is not None:
    table(sim)

h("1.5 변수 기여도 (XGBoost 튜닝, 평균 |SHAP|)", 3)
sh = read(V3 / "shap_importance.csv")
if sh is not None:
    sh.columns = ["변수", "평균 |SHAP|"]
    table(sh.head(12))

# ---------------------------------------------------------------- v4
h("2. v4 — 원인 분석 (6주 = test + 홀드아웃 5주, 같은 방식)")
p("정책: `직전 주` = 직전 7일에서 threshold 선택(운영 가능) / `같은 날 B1과 같은 개수` = 그날 B1 회수 수만큼 점수 상위(운영 가능, 정답 불필요) / "
  "`사후 맞춤` = 그 주 정답을 보고 B1과 잘못 회수량을 맞춤(운영 불가, 순위 품질 상한)")
KEY = ["-", "위험 맞춤 GPU-h (직전 주)", "같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)", "사후 맞춤 건수 (상한)"]
DC = ["F1", "ΔF1 vs B1", "Δ회수 GPU-h vs B1", "Δ잘못 회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]

h("2.1 E1 — threshold 문제인가 순위 문제인가", 3)
e1 = read(V4 / "e1_pooled.csv")
if e1 is not None:
    for pool in e1["구간"].unique():
        p(f"**{pool}**")
        d = e1[(e1["구간"] == pool) & e1["정책"].isin(KEY)]
        table(d[["방법", "정책"] + DC])
e1w = read(V4 / "e1_weekly.csv")
if e1w is not None:
    p("주별 AP (점수 순서 품질)")
    table(e1w[e1w["정책"].isin(["-", "F1 최대 (직전 주)"])].pivot_table(index="방법", columns="주", values="AP"), index=True)
e1b = read(V4 / "e1_bootstrap.csv")
if e1b is not None:
    p("E1 부트스트랩 (app 단위)")
    table(e1b.map(ci_mark))
e1wl = read(V4 / "e1_pooled_by_workload.csv")
if e1wl is not None:
    pool = e1wl["구간"].unique()[-1]
    p(f"workload별 — {pool}")
    d = e1wl[(e1wl["구간"] == pool) & e1wl["정책"].isin(["-", "같은 날 B1과 같은 개수", "사후 맞춤 GPU-h (상한)"])]
    table(d[["workload", "방법", "정책"] + DC])

h("2.2 E2 — seed 5개", 3)
e2 = read(V4 / "e2_seeds_pooled.csv")
if e2 is not None:
    g = e2[e2["정책"].isin(KEY)].groupby(["구간", "방법", "정책"])
    out = pd.DataFrame({
        f"{c} 평균": g[c].mean() for c in ["Δ회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1", "ΔF1 vs B1"]})
    for c in ["Δ회수 GPU-h vs B1", "Δ잘못 회수 건수 vs B1"]:
        out[f"{c} 최소"] = g[c].min()
        out[f"{c} 최대"] = g[c].max()
    out["seed 수"] = g.size()
    table(out, index=True)

h("2.3 E3 — 변수 묶음 (LightGBM 튜닝)", 3)
e3 = read(V4 / "e3_ablation.csv")
if e3 is not None:
    for pool in e3["구간"].unique():
        p(f"**{pool}** — Δ회수 GPU-h vs B1 (정책별) + 주별 AP 평균")
        d = e3[e3["구간"] == pool]
        pv = d.pivot_table(index="설정", columns="정책", values="Δ회수 GPU-h vs B1", sort=False)
        pv["Δ잘못 회수 건수 (같은 날 같은 개수)"] = d[d["정책"] == "같은 날 B1과 같은 개수"].set_index("설정")["Δ잘못 회수 건수 vs B1"]
        pv["주별 AP 평균"] = d.groupby("설정")["주별 AP 평균"].first()
        table(pv, index=True)

h("2.4 E4 — 학습 기간", 3)
e4 = read(V4 / "e4_window.csv")
if e4 is not None:
    for pool in e4["구간"].unique():
        p(f"**{pool}**")
        table(e4[e4["구간"] == pool][["방법", "설정", "정책", "주별 AP 평균"] + DC])

OUT.write_text("\n".join(L) + "\n")
print("written", OUT, f"{len(L)} lines")
