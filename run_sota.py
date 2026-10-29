"""Driver de servidor para el motor climx_sota.

Origen del motor: kernel público de Kaggle maverickss26/climx-sota-main-track (Apache 2.0 por defecto en Kaggle),
extraído sin cambios en climx_sota/ salvo lo marcado. Este driver sustituye las celdas del notebook:
usa los ficheros completos ya descargados (sin borrar nada: keep_raw=True) y la cola de server-ia.

Modos:
  repack                                   construye el store int16 por bandas desde data/full
  validate --holdout 370 [--bands 3,9,..] [--members 16] [--set clave=valor ...]
  predict  [--members 32] [--set clave=valor ...]  -> submission.parquet + índices .npz
"""
import argparse, json, sys, time
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from climx_sota.data import Store, repack_remote, INDEX_VARS
from climx_sota.pipeline import Config, emulate
from climx_sota.score import official_score, to_dataset, write_submission

ROOT = Path("/home/ai/proyectos/climx")
RAW = ROOT / "data/full"
STORE = ROOT / "store_full"
CACHE = ROOT / "cache_bands"
CLIMX_REPO = HERE / "ClimX"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def local_fetch(filename, dest_dir, log=print):
    """Devuelve el fichero local ya descargado; espera a que descarga.log lo marque como completo."""
    p = RAW / Path(filename).name
    dl = ROOT / "data/descarga.log"
    for _ in range(720):
        if dl.exists() and f"OK {p.name}" in dl.read_text():
            return p
        time.sleep(10)
    raise FileNotFoundError(f"{p} no completado en descarga.log")


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v); break
            except ValueError:
                pass
        if v in ("True", "False"):
            v = v == "True"
        out[k] = v
    return out


def subset_score(store, pred, truth):
    w = np.repeat(np.cos(np.deg2rad(store.lat)), store.nlon)
    out = {}
    for k in pred:
        p, t = pred[k].astype(float), truth[k].astype(float)
        ok = np.isfinite(p).all(0) & np.isfinite(t).all(0)
        v = t[:, ok].var(0); good = v > 1e-6
        r2 = 1 - ((p[:, ok] - t[:, ok]) ** 2).mean(0)[good] / v[good]
        out[k] = float(np.average(r2 / (2 - r2), weights=w[ok][good]))
    return {"overall": float(np.mean(list(out.values()))), "metric": "bandas, aprox (sin regiones AR6)", **out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["repack", "validate", "predict"])
    ap.add_argument("--holdout", default="370"); ap.add_argument("--bands", default=None)
    ap.add_argument("--members", type=int, default=None); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--set", nargs="*"); ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default=str(ROOT / "resultados"))
    x = ap.parse_args()
    out = Path(x.out); out.mkdir(parents=True, exist_ok=True)
    if x.mode == "repack":
        from climx_sota.data import FULL_GROUPS, HF_PREFIX
        order = ["ssp126", "ssp370", "ssp585", "historical"]   # el histórico es lo último que se descarga
        groups = sorted([(r, [HF_PREFIX + f for f in fs]) for r, fs in FULL_GROUPS], key=lambda g: order.index(g[0]))
        repack_remote(STORE, RAW, band_rows=4, variables=INDEX_VARS, log=log, fetch=local_fetch,
                      keep_raw=True, workers=4, groups=groups)
        st = Store(STORE)
        for r in st.runs:
            g = st.gmt(r); print(r, len(g), round(float(g[:10].mean()), 3), round(float(g[-10:].mean()), 3))
        return
    st = Store(STORE)
    cfg = Config(workers=x.workers)
    cfg = replace(cfg, **parse_set(x.set))
    if x.members:
        cfg = replace(cfg, members=x.members)
    log("config", asdict(cfg))
    t0 = time.time()
    if x.mode == "validate":
        held = [r for r in st.runs if x.holdout in r]; assert len(held) == 1, st.runs
        train = [r for r in st.runs if r != held[0]]
        bands = [int(b) for b in x.bands.split(",")] if x.bands else None
        res = emulate(st, train, held[0], cfg, truth_run=held[0], bands=bands, cache_dir=CACHE, log=log)
        if bands is None:
            sc = official_score(to_dataset(res["pred"], res["years"], st.lat, st.lon),
                                to_dataset(res["truth"], res["years"], st.lat, st.lon), CLIMX_REPO)
        else:
            sc = subset_score(st, res["pred"], res["truth"])
        sc.update(holdout=x.holdout, bands=x.bands, cfg=asdict(cfg), seconds=round(time.time() - t0))
        (out / f"val_{x.tag}_{x.holdout}.json").write_text(json.dumps(sc, indent=1, default=str))
        log("VAL", x.tag, x.holdout, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in sc.items() if k != "cfg"}))
    else:
        target = st.meta["test_runs"][0]
        res = emulate(st, st.runs, target, cfg, log=log, cache_dir=CACHE)
        pred = to_dataset(res["pred"], res["years"], st.lat, st.lon)
        assert np.isfinite(pred.to_array().values).all(), "faltan bandas"
        p = write_submission(pred, out / f"submission_{x.tag}.parquet", CLIMX_REPO)
        np.savez_compressed(out / f"indices_{x.tag}.npz", years=res["years"], **res["pred"])
        log("SUBMISSION", p, round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
