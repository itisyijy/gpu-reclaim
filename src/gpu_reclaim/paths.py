"""프로젝트 경로와 시간 분할 상수.

데이터 원본은 공식 저장소 위치(upstream/cluster-trace-v2026-maas/data)에 있고
data/raw 는 그곳을 가리키는 심볼릭 링크다.
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("GPU_RECLAIM_ROOT", Path(__file__).resolve().parents[2]))

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / ".cache"
DUCKDB_SPILL = CACHE / "duckdb"

DAILY_DIR = DATA_RAW / "instance_daily"
WEEK_DIR = DATA_RAW / "instance_week_5min"
APP_METRICS_CSV = DATA_RAW / "app_metrics.csv"
APP_CASES_JSON = DATA_RAW / "app_cases.json"
MANIFEST_JSON = DATA_RAW / "manifest.json"

# 시간: day = floor(time_s / 86400). daily 표는 day 0–181, day 172 없음.
FIRST_DAY, LAST_DAY = 0, 181
MISSING_DAYS = (172,)
WEEK_DAYS = (140, 146)  # instance_week_5min, app_metrics 범위

# 제안서 3.2 시간 분할 (양끝 포함)
SPLITS = {
    "train": (0, 132),
    "val": (133, 139),
    "test": (140, 146),
    "holdout": (147, 181),
}


def split_of(day: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= day <= hi:
            return name
    raise ValueError(f"day {day} is outside 0–181")
