"""env-gpu 동작 확인: cupy/cudf/cuml/xgboost/lightgbm이 GPU에서 실제로 도는지 (CUDA_VISIBLE_DEVICES로 GPU 지정)."""
import time
import numpy as np

import cupy as cp
print("cupy", cp.__version__, "devices", cp.cuda.runtime.getDeviceCount(),
      cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
import cudf
df = cudf.DataFrame({"k": np.random.randint(0, 1000, 10_000_000), "v": np.random.rand(10_000_000)})
t = time.time(); r = df.groupby("k").v.mean(); print("cudf", cudf.__version__, f"groupby 1e7 {time.time()-t:.2f}s", len(r))

rng = np.random.default_rng(0)
X = rng.normal(size=(200_000, 50)).astype("float32"); y = (X[:, 0] + rng.normal(size=200_000) > 0).astype("int32")
w = rng.random(200_000).astype("float32")

import cuml
from cuml.svm import SVC
t = time.time(); m = SVC(kernel="rbf", C=1.0, gamma=0.02).fit(X[:50_000], y[:50_000], sample_weight=w[:50_000])
print("cuml", cuml.__version__, f"SVC 5e4 {time.time()-t:.1f}s acc", float((m.predict(X[-10_000:]) == y[-10_000:]).mean()))
from cuml.ensemble import RandomForestClassifier as cuRF
t = time.time(); m = cuRF(n_estimators=100, max_depth=16, random_state=0).fit(X, y)
print(f"cuml RF 2e5x100 {time.time()-t:.1f}s")

import xgboost as xgb
t = time.time(); m = xgb.XGBClassifier(device="cuda", tree_method="hist", n_estimators=300, max_depth=6).fit(X, y, sample_weight=w)
print("xgboost", xgb.__version__, f"cuda 300 trees {time.time()-t:.1f}s")

import lightgbm as lgb
t = time.time(); m = lgb.LGBMClassifier(device_type="cuda", n_estimators=300, verbose=-1).fit(X, y, sample_weight=w)
print("lightgbm", lgb.__version__, f"cuda 300 trees {time.time()-t:.1f}s")
print("OK")
