"""팀원 노트북 v2(2026_maas_analysis_v2.ipynb)의 전처리 셀을 그대로 실행해 preprocessed_df를 parquet으로 저장한다.

- 다운로드 셀(HTTP Range)은 건너뛰고, 그 셀이 정의하는 DATA_DIR / RANDOM_STATE / part_number만 직접 정의
- 전처리(셀 4)·파이프라인 정의(셀 10)·실행(셀 11)을 순서대로 exec
- app_id를 보조 열로 함께 저장 (모델 입력에서는 빼야 함)
- MAX_DAY는 인자로 바꿀 수 있음 (기본 147 = 노트북과 동일, 181이면 홀드아웃까지)
사용: python scripts/build_app_day_v2.py [MAX_DAY]
"""
import json, re, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "2026_maas_analysis_v2.ipynb"
max_day = int(sys.argv[1]) if len(sys.argv) > 1 else 147
out = ROOT / "data" / "processed" / f"app_day_v2_max{max_day}.parquet"

cells = [("".join(c["source"])) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]
load = next(s for s in cells if "MAX_DAY = 147" in s)
pipe = next(s for s in cells if "def build_preprocessing_pipeline" in s)
run = next(s for s in cells if "build_preprocessing_pipeline(idle_threshold=IDLE_THRESHOLD).fit_transform(df)" in s)

g = {"__name__": "__main__", "display": print}
g["DATA_DIR"] = ROOT / "data" / "raw"
g["RANDOM_STATE"] = 0
g["part_number"] = lambda p: int(re.search(r"part_(\d+)\.parquet$", str(p)).group(1))
t = time.time()
exec(load.replace("MAX_DAY = 147", f"MAX_DAY = {max_day}"), g); print(f"load {time.time()-t:.0f}s", g["df"].shape)
# app_id도 보조 변수로 남긴다 (app 단위 부트스트랩·5분 시뮬레이션용, 모델 입력 아님)
pipe = pipe.replace('auxiliary_cols=("day", "gpu_hours_next", "is_mostly_idle")',
                    'auxiliary_cols=("day", "gpu_hours_next", "is_mostly_idle", "app_id")')
assert '"is_mostly_idle", "app_id")' in pipe
exec(pipe, g)
t = time.time()
exec(run, g); print(f"preprocess {time.time()-t:.0f}s")
out.parent.mkdir(parents=True, exist_ok=True)
g["preprocessed_df"].to_parquet(out, index=False)
print("saved", out)
