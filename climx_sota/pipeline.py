"""
End-to-end engine: fit on training runs, emulate a target scenario, compute indices.

Work is split into latitude bands; each band is independent given the shared predictor
matrix and the shared template assignment (shared => spatially coherent weather).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from . import qmap
from .data import INDEX_VARS, Store
from .forcing import PredictorModel, concat_hist
from .indices import INDEX_NAMES, Thresholds, YearState, compute_thresholds, year_indices


@dataclass
class Config:
    members: int = 32               # ensemble members whose indices are averaged
    n_aer_pcs: int = 2              # aerosol pattern predictors (0 = GHG-only pattern scaling)
    two_timescales: bool = True     # fast + slow GHG response
    ridge_ghg: float = 1.0          # ridge penalty (in "years") on GHG columns
    ridge_aer: float = 30.0         # ridge penalty on aerosol columns
    pool_min_year: int = 1850       # template years eligible for resampling
    seed: int = 0
    workers: int = 1
    export_member0: bool = False    # also return member-0 daily fields (for the compliance export)
    export_vars: tuple = ("tas", "tasmax", "tasmin", "pr")
    base_period: tuple = (1961, 1990)
    smooth_baseline: bool = False   # ablation: every day at its conditional median (no weather)
    template_sigma: float = 0.5     # K; draw templates from years whose (emulated) GMT is close to the
                                    # target year's, so the quantile mapping stays a small correction.
                                    # <= 0 -> uniform draw from the whole pool.
    slope_smooth: int = 0           # [sprint 29-10] ancho (celdas de lon) del suavizado de pendientes; 0 = original
    gen_procs: int = 1              # [sprint 29-10] procesos fork para generar miembros dentro de cada banda


# ----------------------------------------------------------------------------
# Predictors
# ----------------------------------------------------------------------------
def build_scenarios(store: Store, names):
    forc = store.forcings()
    hist = forc["historical"]
    out = {}
    for n in names:
        out[n] = hist if n == "historical" else concat_hist(hist, forc[n])
    return out


def fit_predictors(store: Store, train_runs, cfg: Config, log=print):
    scen = build_scenarios(store, train_runs)
    gmt = []
    for r in train_runs:
        g = store.gmt(r)
        yrs = np.unique(store.time(r)[0])
        own = scen[r].own_years
        gmt.append(g[np.isin(yrs, own)])
        if len(gmt[-1]) != len(own):
            raise ValueError(f"{r}: {len(own)} forcing years vs {len(gmt[-1])} target years")
    pm = PredictorModel(n_aer_pcs=cfg.n_aer_pcs, two_timescales=cfg.two_timescales)
    pm.fit([scen[r] for r in train_runs], gmt, store.cell_weights())
    log(f"predictors: {pm.names}  tau_fast={pm.tau_fast} tau_slow={pm.tau_slow}  "
        f"LOSO GMT rmse={np.sqrt(pm.loso_gmt_mse / max(1, len(train_runs) - 1)):.3f} K")
    return pm


def ridge_vector(pm: PredictorModel, cfg: Config) -> np.ndarray:
    g = pm.group_of_columns()
    return np.where(g == 0, 0.0, np.where(g == 1, cfg.ridge_ghg, cfg.ridge_aer))


# ----------------------------------------------------------------------------
# Band worker
# ----------------------------------------------------------------------------
def _years_split(years, doy0):
    uy = np.unique(years)
    return [(y, np.nonzero(years == y)[0]) for y in uy]


def run_band(args):
    (b, store_root, cfg_d, train_runs, designs, ridge, pool, assign, tgt_X, tgt_years, tgt_doy0,
     truth_run, want_vars) = args
    cfg = Config(**cfg_d)
    store = Store(store_root)
    t_start = time.time()
    C = store.cells_in_band(b)
    vars_ = list(dict.fromkeys(INDEX_VARS + list(want_vars)))
    vars_ = [v for v in vars_ if store.has("historical", v)]

    # [sprint 29-10] datos de entrenamiento cargados variable a variable (mismo cálculo, ~4x menos RAM pico)
    tinfo = {r: store.time(r) for r in train_runs}
    splits = {r: _years_split(*tinfo[r]) for r in train_runs}

    # ---- percentile thresholds from the historical base period (truth data) ----------
    yrs_h, doy_h = store.time("historical")
    sel = (yrs_h >= cfg.base_period[0]) & (yrs_h <= cfg.base_period[1])
    hist_full = {v: store.band("historical", v, b) for v in ("tasmax", "tasmin", "pr")}
    thr = compute_thresholds(hist_full["tasmax"][sel], hist_full["tasmin"][sel], hist_full["pr"][sel],
                             doy_h[sel])
    del hist_full

    # ---- fit transforms + quantile models, then PIT of the template pool, per variable --
    trans, models = {}, {}
    pool_s = {}
    for v in vars_:
        raw = {r: store.band(r, v, b) for r in train_runs}
        hist_years = [qmap.to_365(raw["historical"][idx], tinfo["historical"][1][idx])
                      for _, idx in splits["historical"]]
        trans[v] = qmap.VarTransform(v).fit_clim(hist_years)
        del hist_years
        qm = qmap.QuantileModel(n_cols=len(ridge), n_cells=C)
        for r in train_runs:
            X = designs[r]
            for i, (y, idx) in enumerate(splits[r]):
                z = trans[v].fwd(qmap.to_365(raw[r][idx], tinfo[r][1][idx]))
                qm.add_year(X[i], qmap.year_window_quantiles(z))
        models[v] = qm.fit(ridge)
        if cfg.slope_smooth > 1:   # [sprint 29-10]
            models[v].B = qmap.smooth_slopes(models[v].B, store.nlon, cfg.slope_smooth)
        ps = np.empty((len(pool), qmap.N_DOY, C), np.float16)
        for p, (r, yi) in enumerate(pool):
            y, idx = splits[r][yi]
            x = designs[r][yi]
            z = trans[v].fwd(qmap.to_365(raw[r][idx], tinfo[r][1][idx]))
            kd = models[v].daily_knots(models[v].knots(x, v == "pr"))
            s = qmap.to_coord(z, kd)
            if v == "pr":
                s = np.where(z <= 0, qmap.dry_coords(z, kd), s)
            if cfg.smooth_baseline:
                s = np.full_like(s, (qmap.PROBS.size - 1) / 2)
            ps[p] = s
        pool_s[v] = ps
        del raw, qm

    # ---- generate target scenario ------------------------------------------------------
    M = assign.shape[0]
    tsplit = _years_split(tgt_years, tgt_doy0)
    nY = len(tsplit)
    member0 = {v: np.empty((len(tgt_years), C), np.float32) for v in want_vars} if cfg.export_member0 else None
    if cfg.gen_procs > 1 and member0 is None:   # [sprint 29-10] miembros repartidos entre hijos fork
        import multiprocessing as mp
        _GEN.update(models=models, trans=trans, pool_s=pool_s, assign=assign, tgt_X=tgt_X, tsplit=tsplit,
                    tgt_doy0=tgt_doy0, thr=thr, vars_=vars_, C=C)
        n = min(cfg.gen_procs, M)
        with mp.get_context("fork").Pool(n) as p:
            parts = p.map(_gen_members, [list(range(s, M, n)) for s in range(n)])
        _GEN.clear()
        acc = {k: sum(pt[k] for pt in parts) for k in INDEX_NAMES}
        pred = {k: (acc[k] / M).astype(np.float32) for k in INDEX_NAMES}
        return _finish_band(b, store, cfg, truth_run, nY, C, pred, None, t_start, thr)
    acc = {k: np.zeros((nY, C), np.float64) for k in INDEX_NAMES}
    states = [YearState() for _ in range(M)]
    for i, (y, idx) in enumerate(tsplit):
        doy = tgt_doy0[idx]
        kds = {v: models[v].daily_knots(models[v].knots(tgt_X[i], v == "pr")) for v in vars_}
        for j in range(M):
            t = assign[j, i]
            day = {}
            for v in vars_:
                z = qmap.from_coord(pool_s[v][t].astype(np.float32), kds[v])
                day[v] = qmap.from_365(trans[v].inv(z), doy)
            # physical consistency tasmin <= tas <= tasmax
            day["tasmax"] = np.maximum(day["tasmax"], day["tas"])
            day["tasmin"] = np.minimum(day["tasmin"], day["tas"])
            res = year_indices(day["tasmax"], day["tasmin"], day["tas"], day["pr"], doy, thr, states[j])
            for k in INDEX_NAMES:
                acc[k][i] += res[k]
            if member0 is not None and j == 0:
                for v in want_vars:
                    member0[v][idx] = day[v]
    pred = {k: (acc[k] / M).astype(np.float32) for k in INDEX_NAMES}
    return _finish_band(b, store, cfg, truth_run, nY, C, pred, member0, t_start, thr)


_GEN: dict = {}   # estado compartido con los hijos fork (solo lectura)


def _gen_members(js):
    """Índices acumulados (sin dividir) de los miembros `js`; mismo bucle que la versión serie."""
    g = _GEN
    models, trans, pool_s, assign, tgt_X = g["models"], g["trans"], g["pool_s"], g["assign"], g["tgt_X"]
    vars_, thr, tgt_doy0 = g["vars_"], g["thr"], g["tgt_doy0"]
    acc = {k: np.zeros((len(g["tsplit"]), g["C"]), np.float64) for k in INDEX_NAMES}
    states = {j: YearState() for j in js}
    for i, (y, idx) in enumerate(g["tsplit"]):
        doy = tgt_doy0[idx]
        kds = {v: models[v].daily_knots(models[v].knots(tgt_X[i], v == "pr")) for v in vars_}
        for j in js:
            t = assign[j, i]
            day = {}
            for v in vars_:
                z = qmap.from_coord(pool_s[v][t].astype(np.float32), kds[v])
                day[v] = qmap.from_365(trans[v].inv(z), doy)
            day["tasmax"] = np.maximum(day["tasmax"], day["tas"])
            day["tasmin"] = np.minimum(day["tasmin"], day["tas"])
            res = year_indices(day["tasmax"], day["tasmin"], day["tas"], day["pr"], doy, thr, states[j])
            for k in INDEX_NAMES:
                acc[k][i] += res[k]
    return acc


def _finish_band(b, store, cfg, truth_run, nY, C, pred, member0, t_start, thr):
    # ---- truth indices (validation only) ----------------------------------------------
    truth = None
    if truth_run is not None:
        ty, td = store.time(truth_run)
        tr = {v: store.band(truth_run, v, b) for v in INDEX_VARS}
        st = YearState()
        truth = {k: np.zeros((nY, C), np.float32) for k in INDEX_NAMES}
        for i, (y, idx) in enumerate(_years_split(ty, td)):
            res = year_indices(tr["tasmax"][idx], tr["tasmin"][idx], tr["tas"][idx], tr["pr"][idx],
                               td[idx], thr, st)
            for k in INDEX_NAMES:
                truth[k][i] = res[k]
    return b, pred, truth, member0, time.time() - t_start


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def make_pool(store: Store, train_runs, cfg: Config):
    pool = []
    for r in train_runs:
        yrs = np.unique(store.time(r)[0])
        for yi, y in enumerate(yrs):
            # skip truncated years (fewer than 360 days) as templates
            if y >= cfg.pool_min_year and (store.time(r)[0] == y).sum() >= 360:
                pool.append((r, yi))
    return pool


def assign_templates(pool, scen, train_runs, target, pm, cfg: Config, n_years):
    """(M, n_years) pool indices. Shared by all bands => spatially coherent weather."""
    rng = np.random.default_rng(cfg.seed)
    M = min(cfg.members, len(pool))
    if cfg.template_sigma <= 0:
        return np.stack([rng.choice(len(pool), M, replace=False) for _ in range(n_years)], axis=1)
    g_run = {r: pm.predict_gmt(scen[r]) for r in train_runs}
    g_pool = np.array([g_run[r][yi] for r, yi in pool])
    g_tgt = pm.predict_gmt(scen[target])
    cols = []
    for i in range(n_years):
        w = np.exp(-0.5 * ((g_pool - g_tgt[i]) / cfg.template_sigma) ** 2) + 1e-12
        n_eff = w.sum() ** 2 / (w ** 2).sum()
        if n_eff < 2 * M:  # too few close analogues: widen to the 2M nearest
            w = np.zeros_like(w)
            w[np.argsort(np.abs(g_pool - g_tgt[i]))[: 2 * M]] = 1.0
        cols.append(rng.choice(len(pool), M, replace=False, p=w / w.sum()))
    return np.stack(cols, axis=1)


def _cache_key(cfg: Config, train_runs, target, truth_run) -> str:
    import hashlib
    import json
    d = {k: v for k, v in asdict(cfg).items() if k not in ("workers", "gen_procs")}
    d.update(train=list(train_runs), target=target, truth=truth_run)
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def emulate(store: Store, train_runs, target: str, cfg: Config, truth_run=None, time_from=None,
            bands=None, log=print, cache_dir=None):
    """
    Fit on `train_runs`, emulate scenario `target`.
    Returns dict(index -> (n_years, nlat*nlon)), truth (or None), years, member0 daily (or None).
    """
    pm = fit_predictors(store, train_runs, cfg, log)
    scen = build_scenarios(store, train_runs + [target])
    designs = {r: pm.transform(scen[r]) for r in train_runs}
    tgt_X = pm.transform(scen[target])
    log(f"target {target}: predicted GMT {pm.predict_gmt(scen[target])[[0, -1]].round(2)} K (first/last yr)")
    tsrc = time_from or (target if target in store.runs else store.meta["test_time_from"])
    tgt_years, tgt_doy0 = store.time(tsrc)
    uy = np.unique(tgt_years)
    if not np.array_equal(uy, scen[target].own_years):
        raise ValueError(f"target years mismatch: daily {uy[[0, -1]]} ({len(uy)}) vs forcing "
                         f"{scen[target].own_years[[0, -1]]} ({len(tgt_X)})")
    pool = make_pool(store, train_runs, cfg)
    assign = assign_templates(pool, scen, train_runs, target, pm, cfg, len(uy))
    M = assign.shape[0]
    log(f"template pool: {len(pool)} years; members: {M}; template_sigma={cfg.template_sigma}")

    bands = list(range(store.n_bands)) if bands is None else bands
    want = tuple(cfg.export_vars) if cfg.export_member0 else ()
    jobs = [(b, str(store.root), asdict(cfg), train_runs, designs, ridge_vector(pm, cfg), pool, assign,
             tgt_X, tgt_years, tgt_doy0, truth_run, want) for b in bands]
    nY, N = len(uy), store.n_cells
    pred = {k: np.full((nY, N), np.nan, np.float32) for k in INDEX_NAMES}
    truth = {k: np.full((nY, N), np.nan, np.float32) for k in INDEX_NAMES} if truth_run else None
    member0 = {v: np.full((len(tgt_years), N), np.nan, np.float32) for v in want} if want else None

    cache = None
    if cache_dir is not None and not want:
        from pathlib import Path
        cache = Path(cache_dir) / _cache_key(cfg, train_runs, target, truth_run)
        cache.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    n_done = [0]

    def _collect(res):
        b, p, t, m0, dt = res
        if cache is not None and dt >= 0:
            np.savez(cache / f"band{b:03d}.npz", **{f"p_{k}": p[k] for k in INDEX_NAMES},
                     **({f"t_{k}": t[k] for k in INDEX_NAMES} if t is not None else {}))
        n_done[0] += 1
        c0 = b * store.band_cells
        c1 = c0 + store.cells_in_band(b)
        for k in INDEX_NAMES:
            pred[k][:, c0:c1] = p[k]
            if truth is not None:
                truth[k][:, c0:c1] = t[k]
        if member0 is not None:
            for v in want:
                member0[v][:, c0:c1] = m0[v]
        el = time.time() - t_all
        eta = el / n_done[0] * (len(bands) - n_done[0])
        log(f"  band {b + 1}/{store.n_bands} " + ("(cached)" if dt < 0 else f"done in {dt:.0f}s") +
            f"  [{n_done[0]}/{len(bands)}, elapsed {el / 60:.0f} min, ETA {eta / 60:.0f} min]")

    todo = []
    for j in jobs:
        b = j[0]
        f = cache / f"band{b:03d}.npz" if cache is not None else None
        if f is not None and f.exists():
            z = np.load(f)
            p = {k: z[f"p_{k}"] for k in INDEX_NAMES}
            t = {k: z[f"t_{k}"] for k in INDEX_NAMES} if truth is not None else None
            _collect((b, p, t, None, -1.0))
        else:
            todo.append(j)
    if cfg.workers > 1 and len(todo) > 1:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(cfg.workers) as ex:
            futs = [ex.submit(run_band, j) for j in todo]
            for fu in as_completed(futs):
                _collect(fu.result())
    else:
        for j in todo:
            _collect(run_band(j))
    return {"pred": pred, "truth": truth, "years": uy, "member0": member0,
            "time_years": tgt_years, "time_doy0": tgt_doy0, "predictors": pm}