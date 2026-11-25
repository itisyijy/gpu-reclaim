"""5분 단위 시뮬레이션 — '같은 날 B1과 같은 개수' 정책 (v4에 저장된 점수 사용, 다시 학습하지 않음).

v3 노트북 9절과 같은 가정: day t+1 시작에 회수, 첫 활동 5분 창에서 cold start 1건, 그 전까지의 GPU-h를 회수량으로 본다.
5분 표가 140–146일뿐이라 오늘 = 140–145인 행만 쓴다.
5분 집계표는 outputs/final/sim_5min_agg.parquet에 저장해 최종 노트북이 다시 쓴다.
사용: env-gpu/bin/python scripts/sim_5min_samecount.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gpu_reclaim.db import connect  # noqa: E402

OUT = ROOT / "outputs" / "final"
OUT.mkdir(parents=True, exist_ok=True)
AGG = OUT / "sim_5min_agg.parquet"
SQL = """
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
"""


def topk_per_day(s, day, base):
    pred = np.zeros(len(s), dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        k = int(base[idx].sum())
        if k:
            pred[idx[np.argsort(-s[idx], kind="stable")[:k]]] = True
    return pred


def sim_table(ts, methods):
    ok = ts["gpu_h_5min"].notna().to_numpy()
    act = ts["first_active"].notna().to_numpy()
    y, w = ts["y"].to_numpy(), ts["w"].to_numpy()
    rows = []
    for name, pred in methods.items():
        r_ = np.asarray(pred).astype(bool) & ok
        cold = r_ & act
        t_first = (ts.loc[cold, "first_active"] - (ts.loc[cold, "day"] + 1) * 86400) / 3600
        rows.append({"방법": name, "회수 app-day": int(r_.sum()), "cold start 건수": int(cold.sum()),
                     "cold start 비율": cold.sum() / max(r_.sum(), 1),
                     "첫 활동까지 시간 중앙값(h)": float(t_first.median()) if len(t_first) else np.nan,
                     "회수 GPU-h (첫 활동 전까지)": ts.loc[r_, "saved_gpu_h"].sum(),
                     "cold start app의 하루 GPU-h": ts.loc[cold, "gpu_h_5min"].sum(),
                     "일 단위 잘못 회수 건수": int((r_ & (y == 0)).sum()),
                     "일 단위 잘못 회수 GPU-h": float(w[r_ & (y == 0)].sum())})
    res = pd.DataFrame(rows)
    b = res.iloc[0]
    for c in ["회수 app-day", "cold start 건수", "회수 GPU-h (첫 활동 전까지)", "cold start app의 하루 GPU-h",
              "일 단위 잘못 회수 GPU-h"]:
        res[f"Δ{c} vs B1"] = res[c] - b[c]
    return res


def main():
    t0 = time.time()
    if AGG.exists():
        agg = pd.read_parquet(AGG)
    else:
        con = connect(threads=16, memory_limit="64GB")
        agg = con.sql(SQL).df()
        agg.to_parquet(AGG, index=False)
    print(f"5분 집계 {len(agg):,} app-day ({time.time() - t0:.0f}초)", flush=True)

    sc = pd.read_parquet(ROOT / "outputs" / "v4" / "scores.parquet")
    ts = sc[(sc["주"] == "140–146") & (sc["day"] <= 145)].copy()
    a = agg.copy()
    a["app_id"] = a["app_id"].astype(str)
    a["day"] = a["day"] - 1          # 내일 값을 오늘 행 키로
    ts["app_id"] = ts["app_id"].astype(str)
    ts = ts.merge(a, on=["app_id", "day"], how="left")
    print(f"대상 {len(ts):,}행, 5분 기록 있음 {ts['gpu_h_5min'].notna().mean() * 100:.1f}%")

    day, b1 = ts["day"].to_numpy(), ts["b1"].to_numpy().astype(int)
    methods = {"Baseline 1: 오늘 idle": b1.astype(bool)}
    cols = [c for c in ts.columns if c.startswith("E1|")] + [c for c in ts.columns if c.startswith("E2|") and not c.endswith("seed0")]
    for c in cols:
        _, _, name, seed = c.split("|")
        methods[f"{name} ({seed})" if c.startswith("E2|") else name] = topk_per_day(ts[c].to_numpy(), day, b1)
    methods["(상한) 전부 회수"] = np.ones(len(ts), dtype=bool)

    res = sim_table(ts, methods)
    res.to_csv(OUT / "sim_5min_samecount_v4scores.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.round(3).to_string())
    print(f"끝 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
