"""
Scoring and submission writing.  Uses the official ClimX code (rnNSE.py, kaggle_submission.py)
when the ClimX repo is available; otherwise falls back to equivalent local implementations
(submission format identical; metric without the AR6 region weighting -> approximate).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .indices import INDEX_NAMES

# official column order of the submission
SUB_COLUMNS = ("TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
               "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII")
LAT_FULL = np.linspace(-90.0, 90.0, 192)
LON_FULL = np.arange(288) * 1.25


def _official(climx_repo):
    try:
        root = str(Path(climx_repo).expanduser())
        if root not in sys.path:
            sys.path.insert(0, root)
        import rnNSE
        import src.utils.kaggle_submission as ks
        from src.utils.original_data_values import LAT, LON
        return rnNSE, ks, np.asarray(LAT, float), np.asarray(LON, float)
    except Exception:
        return None, None, LAT_FULL, LON_FULL


def to_dataset(arrs: dict, years, lat, lon) -> xr.Dataset:
    nlat, nlon = len(lat), len(lon)
    time = np.array([np.datetime64(f"{int(y):04d}-01-01", "ns") for y in years])
    return xr.Dataset({k: (("time", "lat", "lon"), arrs[k].reshape(len(years), nlat, nlon))
                       for k in INDEX_NAMES}, coords={"time": time, "lat": lat, "lon": lon})


def to_official_grid(ds: xr.Dataset, climx_repo) -> xr.Dataset:
    """Lite (12x18) -> official 192x288 grid exactly as in the official playground."""
    _, _, LAT, LON = _official(climx_repo)
    ds = ds.sortby("lat").sortby("lon")
    if ds.sizes["lat"] == len(LAT) and ds.sizes["lon"] == len(LON):
        return ds
    return ds.interp(lat=LAT, lon=LON, method="nearest", kwargs={"fill_value": "extrapolate"})


def _frame(ds: xr.Dataset) -> pd.DataFrame:
    ds = ds.sortby("time").sortby("lat").sortby("lon")
    arr = np.stack([ds[c].transpose("time", "lat", "lon").values.reshape(-1) for c in SUB_COLUMNS], axis=1)
    df = pd.DataFrame(arr, columns=list(SUB_COLUMNS))
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df


def _approx_score(t: pd.DataFrame, p: pd.DataFrame, nlat, nlon, lats) -> float:
    """nNSE with cos-lat weighting over all valid cells (no AR6 region averaging)."""
    T = len(t) // (nlat * nlon)
    w = np.cos(np.deg2rad(lats))[:, None] * np.ones((1, nlon))
    out = []
    for c in [c for c in t.columns if c != "row_id"]:
        g = t[c].values.reshape(T, nlat, nlon)
        q = p[c].values.reshape(T, nlat, nlon)
        v = g.var(0)
        ok = v > 1e-6
        r2 = 1 - ((g - q) ** 2).mean(0) / np.maximum(v, 1e-6)
        n = r2 / (2 - r2)
        out.append(float((n * w)[ok].sum() / w[ok].sum()))
    return float(np.mean(out))


def official_score(pred: xr.Dataset, truth: xr.Dataset, climx_repo) -> dict:
    rnNSE, ks, LAT, _ = _official(climx_repo)
    p = _frame(to_official_grid(pred, climx_repo))
    t = _frame(to_official_grid(truth, climx_repo))
    f = (lambda a, b: rnNSE.score(a.copy(), b.copy(), "row_id")) if rnNSE else \
        (lambda a, b: _approx_score(a, b, 192, 288, LAT))
    out = {"overall": f(t, p), "metric": "official" if rnNSE else "approx (no AR6 regions)"}
    for k in INDEX_NAMES:
        out[k] = f(t[["row_id", k]], p[["row_id", k]])
    return out


def write_submission(pred: xr.Dataset, path, climx_repo) -> Path:
    _, ks, _, _ = _official(climx_repo)
    ds = to_official_grid(pred, climx_repo)
    if not np.isfinite(ds.to_array().values).all():
        raise ValueError("non-finite values in prediction")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if ks is not None:
        return ks.write_kaggle_submission_parquet(ds, path)
    _frame(ds).to_parquet(path, index=False)
    return path