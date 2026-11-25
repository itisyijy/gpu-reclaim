"""DuckDB 연결 헬퍼: 세 표를 view 로 등록한다.

    from gpu_reclaim.db import connect
    con = connect()
    con.sql("SELECT day, sum(gpu_hours) FROM daily GROUP BY day ORDER BY day").df()

view 이름
- daily : instance_daily/*.parquet  (instance-day 세그먼트, 여러 행 가능 → gpu_hours 합산)
- week  : instance_week_5min/*.parquet (5분 창)
- app   : app_metrics.csv (시간 단위, 문자열 '\\N' 은 NULL)
"""

import warnings

import duckdb

from .paths import APP_METRICS_CSV, DAILY_DIR, DUCKDB_SPILL, WEEK_DIR


def _q(path) -> str:
    return str(path).replace("'", "''")


def connect(threads: int = 8, memory_limit: str = "48GB", database: str = ":memory:"):
    DUCKDB_SPILL.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database, config={
        "threads": threads,
        "memory_limit": memory_limit,
        "temp_directory": str(DUCKDB_SPILL),
    })
    con.execute("SET preserve_insertion_order=false")
    try:  # Jupyter에서는 ipywidgets가 없으면 이 설정 자체가 오류가 난다
        con.execute("SET enable_progress_bar=false")
    except duckdb.Error:
        pass
    for view, folder in [("daily", DAILY_DIR), ("week", WEEK_DIR)]:
        if any(folder.glob("*.parquet")):
            con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{_q(folder / '*.parquet')}')")
        else:
            warnings.warn(f"{folder} 에 parquet 이 없어 view '{view}' 를 만들지 않음")
    if APP_METRICS_CSV.exists():
        con.execute(
            "CREATE OR REPLACE VIEW app AS SELECT * FROM read_csv("
            f"'{_q(APP_METRICS_CSV)}', header=true, nullstr='\\N', "
            "types={'time_s': 'BIGINT', 'value': 'DOUBLE'})"
        )
    else:
        warnings.warn(f"{APP_METRICS_CSV} 가 없어 view 'app' 을 만들지 않음")
    return con
