"""Comprueba que la generación paralela (gen_procs) reproduce la banda 0 guardada por la versión serie."""
import sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from climx_sota.data import Store
from climx_sota.indices import INDEX_NAMES
from climx_sota.pipeline import Config, emulate, _cache_key

ROOT = Path("/home/ai/proyectos/climx")
st = Store(ROOT / "store_full")
cfg = replace(Config(workers=1), members=64, gen_procs=int(sys.argv[1]))
target = st.meta["test_runs"][0]
key = _cache_key(cfg, st.runs, target, None)
ref = np.load(ROOT / "cache_bands" / key / "band000.npz")
t0 = time.time()
res = emulate(st, st.runs, target, cfg, bands=[0], cache_dir=None, log=print)
c1 = st.cells_in_band(0)
worst = 0.0
for k in INDEX_NAMES:
    a, r = res["pred"][k][:, :c1], ref[f"p_{k}"]
    dd = np.abs(a - r)
    d = float(np.nanmax(dd))
    worst = max(worst, d / (float(np.nanmax(np.abs(r))) + 1e-9))
    print(k, "max abs diff", d, "frac celdas-año distintas", float(np.mean(dd > 1e-4)))
print("VERIFY key", key, "rel", worst, "seconds", round(time.time() - t0))
sys.exit(0 if worst < 1e-2 else 3)   # umbral laxo: cruces de umbral de lluvia por coma flotante
