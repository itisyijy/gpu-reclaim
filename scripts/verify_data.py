"""압축 해제한 데이터를 data/raw/manifest.json 과 대조한다.

    python -I scripts/verify_data.py     (프로젝트 루트에서, env 활성화 상태)

검사 항목: 파일 목록·바이트·파일별 행 수, 스키마(열 이름·타입), 표별 행 수,
daily 의 day 범위·관측일 수·결측일, gpu_hours·samples 합계, week 의 time_s 범위·samples 합계,
app_metrics 행 수. 하나라도 어긋나면 종료 코드 1.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpu_reclaim.db import connect  # noqa: E402
from gpu_reclaim.paths import APP_METRICS_CSV, DATA_RAW, MANIFEST_JSON, MISSING_DAYS  # noqa: E402

failures = []


def check(label, actual, expected, rel=None):
    ok = math.isclose(actual, expected, rel_tol=rel) if rel else actual == expected
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {actual!r} (기대 {expected!r})")
    if not ok:
        failures.append(label)


manifest = json.loads(MANIFEST_JSON.read_text())
con = connect(threads=16)

for ds in manifest["datasets"]:
    table = ds["table"]
    if ds["format"] == "csv":
        n = con.sql("SELECT count(*) FROM app").fetchone()[0]
        check(f"{table} 행 수", n, ds["rows"])
        continue

    folder = DATA_RAW / table
    actual_files = sorted(p.name for p in folder.glob("*.parquet"))
    check(f"{table} 샤드 수", len(actual_files), ds["shards"])
    check(f"{table} 파일 목록", actual_files == sorted(f["name"] for f in ds["files"]), True)
    for f in ds["files"]:
        p = folder / f["name"]
        size = p.stat().st_size if p.exists() else -1
        rows = con.sql(f"SELECT count(*) FROM read_parquet('{p}')").fetchone()[0] if p.exists() else -1
        if size != f["bytes"] or rows != f["rows"]:
            check(f"{table}/{f['name']} 바이트·행", (size, rows), (f["bytes"], f["rows"]))
    check(f"{table} 파일 바이트 합", sum((folder / f["name"]).stat().st_size for f in ds["files"]),
          ds["parquet_bytes"])

    view = "daily" if table == "instance_daily" else "week"
    schema = [(r[0], r[1]) for r in con.sql(f"DESCRIBE {view}").fetchall()]
    check(f"{table} 스키마", schema, [(s["name"], s["type"]) for s in ds["schema"]])
    s = ds["summary"]
    if view == "daily":
        n, d0, d1, nd, gh, sm = con.sql(
            "SELECT count(*), min(day), max(day), count(DISTINCT day), sum(gpu_hours), sum(samples) FROM daily"
        ).fetchone()
        check("daily 행 수", n, s["rows"])
        check("daily 첫날", d0, s["first_day"])
        check("daily 마지막날", d1, s["last_day"])
        check("daily 관측일 수", nd, s["observed_days"])
        present = {r[0] for r in con.sql("SELECT DISTINCT day FROM daily").fetchall()}
        check("daily 결측일", sorted(set(range(d0, d1 + 1)) - present), list(MISSING_DAYS))
        check("daily gpu_hours 합", gh, s["gpu_hours"], rel=1e-9)
        check("daily samples 합", sm, s["samples"])
    else:
        n, t0, t1, sm = con.sql("SELECT count(*), min(time_s), max(time_s), sum(samples) FROM week").fetchone()
        check("week 행 수", n, s["rows"])
        check("week 첫 time_s", t0, s["first_time_s"])
        check("week 마지막 time_s", t1, s["last_time_s"])
        check("week samples 합", sm, s["samples"])

check("app_metrics.csv 존재", APP_METRICS_CSV.exists(), True)
print("\n결과:", "모두 통과" if not failures else f"실패 {len(failures)}건 → {failures}")
sys.exit(1 if failures else 0)
