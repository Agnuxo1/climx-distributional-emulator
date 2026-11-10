"""
Data access for ClimX — robust to file layout and independent of `cftime`.

* Files are found anywhere under the data root and classified by name:
    *hist*            historical run (may be split into several files by period)
    *train*ssp###*    one training SSP per file   (or one file with a scenario dimension)
    *test*            test forcings (SSP2-4.5)
    anything else     extra forcing files (e.g. *forcings_all*), searched when a run lacks forcings
* Every file is opened with decode_times=False and time axes are decoded here for
  noleap/365_day, 360_day, all_leap/366_day and standard/gregorian calendars, so no cftime is needed.
* Daily targets are repacked into per-latitude-band memory-mapped arrays so each worker reads a
  contiguous block of cells for all training days in one go.

Store layout (directory):
    meta.json                       grid, bands, runs, variables
    {run}__time.npz                 years (T,), doy0 (T,)
    {run}__{var}.dat                float32 memmap (n_bands, T, band_cells)
    {run}__gmt.npy                  global-mean annual tas for that run's years
    forcings.pkl                    dict run -> ScenarioForcing (annual)
Temperatures are stored in K, precipitation in mm/day.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr

from .forcing import AER_NAMES, GHG_NAMES, ScenarioForcing

INDEX_VARS = ["tas", "tasmax", "tasmin", "pr"]
ALL_VARS = ["tas", "tasmax", "tasmin", "pr", "huss", "psl", "sfcWind"]
KNOWN_VARS = set(ALL_VARS) | set(GHG_NAMES) | set(AER_NAMES)
LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
TIME_NAMES = ("time", "time_year", "time_month", "year", "month", "time_counter")


# ----------------------------------------------------------------------------
# Time decoding (no cftime)
# ----------------------------------------------------------------------------
_CUM_NOLEAP = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_UNIT_DAYS = {"day": 1.0, "days": 1.0, "d": 1.0, "hour": 1 / 24, "hours": 1 / 24, "h": 1 / 24,
              "minute": 1 / 1440, "minutes": 1 / 1440, "second": 1 / 86400, "seconds": 1 / 86400,
              "s": 1 / 86400}


def decode_time(values, attrs) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, 0-based day-of-year) for a time coordinate given raw values + CF attrs."""
    v = np.asarray(values)
    units = str(attrs.get("units", ""))
    cal = str(attrs.get("calendar", "standard")).lower()
    if np.issubdtype(v.dtype, np.datetime64):
        d = v.astype("datetime64[D]")
        y = d.astype("datetime64[Y]").astype(int) + 1970
        return y, (d - d.astype("datetime64[Y]")).astype(int)
    if v.dtype == object:  # already cftime / datetime objects
        y = np.array([t.year for t in v], int)
        doy = np.array([getattr(t, "dayofyr", None) or t.timetuple().tm_yday for t in v], int) - 1
        return y, doy
    if "since" not in units:  # plain year numbers
        return np.round(v.astype(float)).astype(int), np.zeros(len(v), int)
    step, ref = units.split("since")
    mult = _UNIT_DAYS.get(step.strip().lower())
    if mult is None:
        raise ValueError(f"unsupported time units '{units}'")
    m = re.match(r"\s*(-?\d+)-(\d+)-(\d+)", ref)
    if not m:
        raise ValueError(f"cannot parse reference date in '{units}'")
    y0, mo0, d0 = map(int, m.groups())
    days = np.floor(v.astype(float) * mult + 1e-9).astype(np.int64)
    if cal in ("noleap", "365_day"):
        start = y0 * 365 + _CUM_NOLEAP[mo0 - 1] + d0 - 1 + days
        return start // 365, start % 365
    if cal == "360_day":
        start = y0 * 360 + (mo0 - 1) * 30 + d0 - 1 + days
        return start // 360, start % 360
    if cal in ("all_leap", "366_day"):
        cum = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        start = y0 * 366 + cum[mo0 - 1] + d0 - 1 + days
        return start // 366, start % 366
    # standard / gregorian / proleptic_gregorian (proleptic arithmetic is fine for 1850-2100)
    base = np.datetime64(date(max(y0, 1), mo0, d0).isoformat(), "D")
    d = base + days.astype("timedelta64[D]")
    y = d.astype("datetime64[Y]").astype(int) + 1970
    return y, (d - d.astype("datetime64[Y]")).astype(int)


# ----------------------------------------------------------------------------
# Opening and classifying files
# ----------------------------------------------------------------------------
def _open(path) -> xr.Dataset:
    p = str(path)
    if p.endswith(".zarr"):
        ds = xr.open_zarr(p, decode_times=False)
    else:
        err = None
        for eng in (None, "h5netcdf", "netcdf4", "scipy"):
            try:
                ds = xr.open_dataset(p, decode_times=False, engine=eng)
                break
            except Exception as e:  # engine missing / cannot read this file
                err = e
        else:
            raise RuntimeError(f"cannot open {p} with any engine (pip install h5netcdf h5py?)") from err
    ren = {}
    for n in LAT_NAMES:
        if n in ds.dims and n != "lat":
            ren[n] = "lat"
            break
    for n in LON_NAMES:
        if n in ds.dims and n != "lon":
            ren[n] = "lon"
            break
    return ds.rename(ren) if ren else ds


def _time_dim(da: xr.DataArray, ds: xr.Dataset) -> str | None:
    for d in da.dims:
        if d in ds.coords and "since" in str(ds[d].attrs.get("units", "")):
            return d
    for d in da.dims:
        if d in TIME_NAMES:
            return d
    return None


def _extra_dims(da, ds):
    td = _time_dim(da, ds)
    return [d for d in da.dims if d not in ("lat", "lon", td)]


def _select_scenario(da: xr.DataArray, ds: xr.Dataset, name: str | None):
    """Reduce extra (scenario / member) dims. Returns None if the scenario isn't in this array."""
    for d in _extra_dims(da, ds):
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        elif d in ds.coords and ds[d].dtype.kind in "OUS" and name is not None:
            labels = [str(x) for x in ds[d].values]
            hits = [i for i, lab in enumerate(labels) if name in lab or lab in name]
            if not hits:
                return None
            da = da.isel({d: hits[0]})
        else:
            da = da.isel({d: 0})
    return da


class Source:
    """One run (historical or a scenario) possibly spread over several files."""

    def __init__(self, name: str, files: list, extra: list):
        self.name = name
        self.dsets = [_open(f) for f in files]
        self.extra = extra  # list of opened datasets searched for missing forcings

    def close(self):
        for ds in self.dsets:
            ds.close()

    def var_parts(self, var):
        """[(DataArray(time, lat, lon), ds)] in file order for a daily variable."""
        out = []
        for ds in self.dsets:
            if var in ds:
                da = _select_scenario(ds[var], ds, self.name)
                if da is not None:
                    out.append((da.transpose(_time_dim(da, ds), "lat", "lon"), ds))
        return out

    def daily_time(self, var="tas"):
        ys, ds_ = [], []
        for da, ds in self.var_parts(var):
            td = da.dims[0]
            y, d = decode_time(ds[td].values, ds[td].attrs)
            ys.append(y)
            ds_.append(d)
        if not ys:
            raise ValueError(f"{self.name}: variable '{var}' not found")
        y, d = np.concatenate(ys), np.concatenate(ds_)
        return y, d

    def _annual_from(self, ds, var):
        da = _select_scenario(ds[var], ds, self.name)
        if da is None:
            return None
        td = _time_dim(da, ds)
        if td is None:
            return None
        sp = [d for d in ("lat", "lon") if d in da.dims]
        da = da.transpose(td, *sp)
        y, _ = decode_time(ds[td].values, ds[td].attrs)
        vals = np.asarray(da.values, dtype=np.float64).reshape(len(y), -1)
        uy = np.unique(y)
        if len(uy) != len(y):
            vals = np.stack([np.nanmean(vals[y == k], axis=0) for k in uy])
        return uy, vals

    def _collect(self, pool, g):
        """Union over files of annual series for forcing g (later files override overlapping years)."""
        ys, vs = [], []
        for ds in pool:
            if g in ds:
                r = self._annual_from(ds, g)
                if r is not None:
                    ys.append(r[0])
                    vs.append(r[1])
        if not ys:
            return None
        y = np.concatenate(ys)
        v = np.concatenate(vs)
        uy, last = np.unique(y[::-1], return_index=True)  # keep the last occurrence of each year
        idx = len(y) - 1 - last
        return uy, v[idx]

    def forcings(self, years_needed, log=print) -> ScenarioForcing:
        years_needed = np.asarray(years_needed)
        ghg, aer, prov = {}, {}, {}
        for g in GHG_NAMES + AER_NAMES:
            r = self._collect(self.dsets, g)
            src = "run"
            if r is None or not np.isin(years_needed, r[0]).all():
                r2 = self._collect(self.extra, g)
                if r2 is not None:
                    if r is None:
                        r, src = r2, "extra"
                    else:  # fill missing years from the extra files
                        miss = ~np.isin(r2[0], r[0])
                        y = np.concatenate([r[0], r2[0][miss]])
                        v = np.concatenate([r[1], r2[1][miss]])
                        o = np.argsort(y)
                        r, src = (y[o], v[o]), "run+extra"
            if r is None:
                continue
            y, v = r
            covered = np.isin(years_needed, y)
            if not covered.all():
                log(f"  [{self.name}] {g}: {int((~covered).sum())} years missing -> nearest year used")
            idx = np.array([np.argmin(np.abs(y - k)) for k in years_needed])
            sel = v[idx]
            if g in GHG_NAMES:
                ghg[g] = sel.mean(axis=1)
            else:
                aer[g] = sel
            prov[g] = src
        if not ghg:
            raise ValueError(f"{self.name}: no greenhouse-gas forcing found in any file")
        missing = [g for g in GHG_NAMES + AER_NAMES if g not in prov]
        if missing:
            log(f"  [{self.name}] forcings not found anywhere: {missing}")
        if any(v != "run" for v in prov.values()):
            log(f"  [{self.name}] forcing sources: {prov}")
        return ScenarioForcing(name=self.name, years=years_needed, ghg=ghg, aer=aer)


def scan(data_root, log=print):
    root = Path(data_root)
    files = sorted(list(root.rglob("*.nc")) + [p for p in root.rglob("*.zarr") if p.is_dir()])
    files = [f for f in files if not f.name.startswith(".")]
    log("data files:\n  " + "\n  ".join(str(f.relative_to(root)) for f in files))
    nm = lambda f: f.name.lower()
    hist = [f for f in files if "hist" in nm(f)]
    test = [f for f in files if "test" in nm(f) and f not in hist]
    train = [f for f in files if "train" in nm(f) and f not in hist and f not in test]
    extra = [f for f in files if f not in hist + test + train]
    if not (hist and test and train):
        raise FileNotFoundError(f"could not classify files: hist={hist} train={train} test={test}")
    extra_ds = [_open(f) for f in extra]
    sources = {"historical": Source("historical", hist, extra_ds)}
    per_file = {f: re.search(r"ssp\d{3}", nm(f)) for f in train}
    if all(per_file.values()):
        for f, m in per_file.items():
            sources.setdefault(m.group(0), None)
        for s in [k for k in sources if k != "historical"]:
            sources[s] = Source(s, [f for f in train if s in nm(f)], extra_ds)
    else:  # single file(s) holding a scenario dimension
        ds = _open(train[0])
        dims = [d for v in ds.data_vars if v in ALL_VARS for d in _extra_dims(ds[v], ds)]
        if not dims:
            raise ValueError("training file has no scenario dimension and no sspXYZ in its name")
        for s in [str(x) for x in ds[dims[0]].values]:
            sources[s] = Source(s, train, extra_ds)
    tnames = sorted({m.group(0) for f in test if (m := re.search(r"ssp\d{3}", nm(f)))}) or ["ssp245"]
    tests = {t: Source(t, test, extra_ds) for t in tnames}
    log(f"historical: {[f.name for f in hist]}")
    log(f"training scenarios: {[k for k in sources if k != 'historical']}  | test: {tnames}  | extra: {[f.name for f in extra]}")
    return sources, tests


# ----------------------------------------------------------------------------
# Store
# ----------------------------------------------------------------------------
# int16 codec for the compact store: value -> round((f(value) - off) * scale)
CODEC = {
    "tas": ("lin", 260.0, 100.0), "tasmax": ("lin", 260.0, 100.0), "tasmin": ("lin", 260.0, 100.0),
    "pr": ("cbrt", 0.0, 3000.0),         # cube root of mm/day: exact zeros, ~1e-3 mm resolution near 1 mm
    "huss": ("lin", 0.0, 1e6), "psl": ("lin", 101000.0, 1.0), "sfcWind": ("lin", 0.0, 1000.0),
}


def encode(var, x):
    kind, off, sc = CODEC[var]
    y = np.cbrt(np.maximum(x, 0)) if kind == "cbrt" else x
    return np.clip(np.rint((y - off) * sc), -32767, 32767).astype(np.int16)


def decode(var, q):
    kind, off, sc = CODEC[var]
    y = q.astype(np.float32) / np.float32(sc) + np.float32(off)
    return y * y * y if kind == "cbrt" else y


class Store:
    """
    Band-major daily data.  Two layouts are supported:
      legacy : {run}__{var}.dat           float32 (n_bands, T, band_cells)
      compact: {run}__{var}__p{k}.dat     int16   (band_hi-band_lo, T_k, band_cells), one file per
               source file ("part") of the run, holding only bands [band_lo, band_hi).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.compact = self.meta.get("dtype") == "int16"
        self.band_lo = self.meta.get("band_lo", 0)
        self.band_hi = self.meta.get("band_hi", self.meta["n_bands"])

    @property
    def nlat(self): return self.meta["nlat"]
    @property
    def nlon(self): return self.meta["nlon"]
    @property
    def n_cells(self): return self.nlat * self.nlon
    @property
    def band_rows(self): return self.meta["band_rows"]
    @property
    def n_bands(self): return self.meta["n_bands"]
    @property
    def band_cells(self): return self.band_rows * self.nlon
    @property
    def runs(self): return list(self.meta["runs"].keys())
    @property
    def lat(self): return np.array(self.meta["lat"])
    @property
    def lon(self): return np.array(self.meta["lon"])
    @property
    def local_bands(self): return list(range(self.band_lo, self.band_hi))

    def time(self, run):
        z = np.load(self.root / f"{run}__time.npz")
        return z["years"], z["doy0"]

    def _part_file(self, run, var, k):
        return self.root / f"{run}__{var}__p{k}.dat"

    def memmap(self, run, var, mode="r"):  # legacy layout only
        T = self.meta["runs"][run]["T"]
        return np.memmap(self.root / f"{run}__{var}.dat", dtype=np.float32, mode=mode,
                         shape=(self.n_bands, T, self.band_cells))

    def has(self, run, var):
        if self.compact:
            return self._part_file(run, var, 0).exists()
        return (self.root / f"{run}__{var}.dat").exists()

    def band(self, run, var, b) -> np.ndarray:
        """(T, cells_in_band) float32 — padded rows beyond nlat are dropped."""
        n = self.cells_in_band(b)
        if not self.compact:
            return np.array(self.memmap(run, var)[b])[:, :n]
        if not (self.band_lo <= b < self.band_hi):
            raise IndexError(f"band {b} not in this store (bands {self.band_lo}-{self.band_hi - 1})")
        nb = self.band_hi - self.band_lo
        out = []
        for k, Tk in enumerate(self.meta["runs"][run]["parts"]):
            mm = np.memmap(self._part_file(run, var, k), dtype=np.int16, mode="r", shape=(nb, Tk, self.band_cells))
            out.append(decode(var, np.array(mm[b - self.band_lo])[:, :n]))
            del mm
        return np.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def cells_in_band(self, b):
        rows = min(self.band_rows, self.nlat - b * self.band_rows)
        return rows * self.nlon

    def gmt(self, run):
        return np.load(self.root / f"{run}__gmt.npy")

    def forcings(self) -> dict:
        with open(self.root / "forcings.pkl", "rb") as f:
            return pickle.load(f)

    def cell_weights(self):
        w = np.cos(np.deg2rad(self.lat))[:, None] * np.ones((1, self.nlon))
        return w.ravel()

    def delete_daily(self):
        """Remove the daily data files (keeps meta/time/gmt/forcings)."""
        for f in self.root.glob("*.dat"):
            f.unlink()


def _unit_fix(var, x, run, log):
    """Decide (scale, offset) once per variable from the first slab."""
    scale, offset = np.float32(1.0), np.float32(0.0)
    med = float(np.nanmedian(x))
    if var == "pr":
        if np.nanmax(x) < 0.05:
            scale = np.float32(86400.0)  # kg m-2 s-1 -> mm/day
        else:
            log(f"  WARNING {run} pr looks like mm/day already (max={np.nanmax(x):.3g}); not converting")
    if var in ("tas", "tasmax", "tasmin") and med < 150:
        offset = np.float32(273.15)
        log(f"  WARNING {run} {var} looks like degC (median={med:.1f}); converting to K")
    return scale, offset


def _init_meta(src_hist, band_rows, variant):
    ref = src_hist.var_parts("tas")[0][0]
    lat = np.asarray(ref["lat"].values, float)
    lon = np.asarray(ref["lon"].values, float)
    if lat[0] > lat[-1]:
        raise ValueError("latitude is descending; expected ascending (south -> north)")
    nlat, nlon = len(lat), len(lon)
    return {"nlat": nlat, "nlon": nlon, "lat": lat.tolist(), "lon": lon.tolist(), "band_rows": band_rows,
            "n_bands": int(np.ceil(nlat / band_rows)), "runs": {}, "variables": [], "variant": variant}


def _repack_run(run, src, out: Path, meta, variables, slab_days, log):
    """Write one run's daily variables into band memmaps; returns its annual forcings."""
    nlat, nlon, band_rows, n_bands = meta["nlat"], meta["nlon"], meta["band_rows"], meta["n_bands"]
    band_cells = band_rows * nlon
    lat = np.asarray(meta["lat"])
    w = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
    w = (w / w.sum()).ravel()
    years, doy0 = src.daily_time("tas")
    order = np.lexsort((doy0, years))
    if not np.all(order == np.arange(len(order))):
        raise ValueError(f"{run}: time axis not sorted across files")
    T = len(years)
    np.savez(out / f"{run}__time.npz", years=years, doy0=doy0.astype(np.int16))
    meta["runs"][run] = {"T": T, "first_year": int(years[0]), "last_year": int(years[-1])}
    uy = np.unique(years)
    log(f"[{run}] {T} days, {uy[0]}-{uy[-1]}, {len(src.var_parts('tas'))} file(s)")
    gmt_acc, cnt = np.zeros(len(uy)), np.zeros(len(uy))
    for var in variables:
        vparts = src.var_parts(var)
        if not vparts:
            log(f"  [{run}] {var} missing, skipped")
            continue
        t_start = time.time()
        mm = np.memmap(out / f"{run}__{var}.dat", dtype=np.float32, mode="w+", shape=(n_bands, T, band_cells))
        t_off, scale, offset = 0, None, None
        for da, _ in vparts:
            n = da.sizes[da.dims[0]]
            for a0 in range(0, n, slab_days):
                a1 = min(n, a0 + slab_days)
                x = np.asarray(da.isel({da.dims[0]: slice(a0, a1)}).values, np.float32).reshape(a1 - a0, -1)
                if scale is None:
                    scale, offset = _unit_fix(var, x, run, log)
                if not np.isfinite(x).all():
                    x = np.where(np.isfinite(x), x, np.nanmean(x, axis=0, keepdims=True)).astype(np.float32)
                x = x * scale + offset
                t0, t1 = t_off + a0, t_off + a1
                if var == "tas":
                    yi = np.searchsorted(uy, years[t0:t1])
                    np.add.at(gmt_acc, yi, x @ w)
                    np.add.at(cnt, yi, 1)
                for b in range(n_bands):
                    c0 = b * band_cells
                    c1 = min(nlat * nlon, c0 + band_cells)
                    mm[b, t0:t1, : c1 - c0] = x[:, c0:c1]
            t_off += n
        if t_off != T:
            raise ValueError(f"{run} {var}: {t_off} days vs {T} for tas")
        mm.flush()
        del mm
        if var not in meta["variables"]:
            meta["variables"].append(var)
        log(f"  [{run}] {var} done in {time.time() - t_start:.0f}s")
    np.save(out / f"{run}__gmt.npy", gmt_acc / np.maximum(cnt, 1))
    try:
        return src.forcings(uy, log)
    except ValueError as e:
        log(f"  [{run}] {e}")
        return None


def _finish(out: Path, meta, forc, test_names, first_ssp, log):
    with open(out / "forcings.pkl", "wb") as f:
        pickle.dump(forc, f)
    meta["test_runs"] = list(test_names)
    meta["test_time_from"] = first_ssp
    for r, f in forc.items():
        log(f"  forcings {r}: GHG {sorted(f.ghg)} | aerosol {sorted(f.aer)} | {f.years[0]}-{f.years[-1]}")
    (out / "meta.json").write_text(json.dumps(meta))
    log("repack done")
    return Store(out)


def repack(data_root, variant=None, climx_repo=None, out_dir="store", band_rows=4, variables=None,
           slab_days=730, log=print):
    """Local files: one pass over the dataset -> band memmaps + forcings + global-mean tas."""
    variables = variables or ALL_VARS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources, tests = scan(data_root, log)
    meta = _init_meta(sources["historical"], band_rows, variant)
    forc = {run: _repack_run(run, src, out, meta, variables, slab_days, log) for run, src in sources.items()}
    if any(v is None for v in forc.values()):
        raise ValueError(f"no forcings for runs {[r for r, v in forc.items() if v is None]}")
    first_ssp = next(r for r in sources if r != "historical")
    ty, _ = sources[first_ssp].daily_time("tas")
    for t, src in tests.items():
        forc[t] = src.forcings(np.unique(ty), log)
    return _finish(out, meta, forc, tests, first_ssp, log)


# ----------------------------------------------------------------------------
# Full resolution from Hugging Face: one file at a time, compact int16 store, band subsets
# ----------------------------------------------------------------------------
HF_REPO = "isp-uv-es/ClimX"
HF_PREFIX = "full/NorESM2-MM_r1i1p1f1_with_forcing_"
FULL_GROUPS = [
    ("historical", ["hist_train_1850_1931.nc", "hist_train_1932_2014.nc"]),
    ("ssp126", ["train_ssp126.nc"]),
    ("ssp370", ["train_ssp370.nc"]),
    ("ssp585", ["train_ssp585.nc"]),
]
FULL_TEST = ("ssp245", ["test_forcings_ssp245.nc"])


def hf_fetch(filename: str, dest_dir: Path, log=print) -> Path:
    """Download one file of the full dataset. Returns the local path."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / Path(filename).name
    if target.exists():
        return target
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        p = Path(hf_hub_download(HF_REPO, filename, repo_type="dataset", local_dir=str(dest_dir)))
        if p != target:
            p.replace(target)
    except Exception as e:  # fall back to a plain streamed HTTP download
        log(f"  huggingface_hub failed ({type(e).__name__}: {e}); plain HTTP download")
        import urllib.request
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{filename}"
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            while True:
                buf = r.read(64 << 20)
                if not buf:
                    break
                f.write(buf)
        tmp.replace(target)
    # clear any hub cache left inside dest_dir
    import shutil
    shutil.rmtree(dest_dir / ".cache", ignore_errors=True)
    gb = target.stat().st_size / 1e9
    log(f"  downloaded {target.name}: {gb:.1f} GB in {time.time() - t0:.0f}s ({gb * 1e3 / max(1, time.time() - t0):.0f} MB/s)")
    return target


def _write_var_part(args):
    """Worker: read one variable of one source file, write the int16 band file for bands [lo, hi)."""
    path, run, var, out_file, lo, hi, band_rows, nlat, nlon, slab_days, w = args
    src = Source(run, [path], [])
    parts = src.var_parts(var)
    if not parts:
        src.close()
        return var, None, None, []
    da = parts[0][0]
    n = da.sizes[da.dims[0]]
    band_cells = band_rows * nlon
    c_lo = lo * band_cells
    c_hi = min(nlat * nlon, hi * band_cells)
    mm = np.memmap(out_file, dtype=np.int16, mode="w+", shape=(hi - lo, n, band_cells))
    msgs, scale, offset = [], None, None
    gm = np.zeros(n) if var == "tas" else None
    for a0 in range(0, n, slab_days):
        a1 = min(n, a0 + slab_days)
        x = np.asarray(da.isel({da.dims[0]: slice(a0, a1)}).values, np.float32).reshape(a1 - a0, -1)
        if scale is None:
            scale, offset = _unit_fix(var, x, run, msgs.append)
        if not np.isfinite(x).all():
            x = np.where(np.isfinite(x), x, np.nanmean(x, axis=0, keepdims=True)).astype(np.float32)
        x = x * scale + offset
        if gm is not None:
            gm[a0:a1] = x @ w
        q = encode(var, x[:, c_lo:c_hi])
        for b in range(lo, hi):
            s0 = (b - lo) * band_cells
            s1 = min(c_hi - c_lo, s0 + band_cells)
            mm[b - lo, a0:a1, : s1 - s0] = q[:, s0:s1]
    mm.flush()
    del mm
    src.close()
    return var, n, gm, msgs


def _merge_forcings(a: ScenarioForcing | None, b: ScenarioForcing) -> ScenarioForcing:
    if a is None:
        return b
    years = np.concatenate([a.years, b.years])
    o = np.argsort(years, kind="stable")
    cat = lambda d1, d2: {k: np.concatenate([d1[k], d2[k]])[o] for k in d1 if k in d2}
    return ScenarioForcing(name=a.name, years=years[o], ghg=cat(a.ghg, b.ghg), aer=cat(a.aer, b.aer))


def repack_remote(out_dir, raw_dir, band_lo=0, band_hi=None, band_rows=4, variables=None, slab_days=365,
                  log=print, fetch=None, groups=None, test=None, keep_raw=False, workers=4):
    """
    Build a compact (int16) band store for bands [band_lo, band_hi) of the full-resolution data,
    holding at most ONE raw file on disk at a time:  download file -> write its parts -> delete it.
    Variables are written in parallel (one process each). Resumable within a session.
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    variables = variables or INDEX_VARS
    fetch = fetch or hf_fetch
    groups = groups or [(r, [HF_PREFIX + f for f in fs]) for r, fs in FULL_GROUPS]
    test = test or (FULL_TEST[0], [HF_PREFIX + f for f in FULL_TEST[1]])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prog_f = out / "progress.pkl"
    prog = pickle.load(open(prog_f, "rb")) if prog_f.exists() else {"meta": None, "done": [], "forc": {},
                                                                      "time": {}, "gmt": {}}
    meta = prog["meta"]
    for run, files in groups:
        for k, fn in enumerate(files):
            key = f"{run}:{k}"
            if key in prog["done"]:
                log(f"[{run} part {k}] already done")
                continue
            free = shutil.disk_usage(raw_dir if Path(raw_dir).exists() else "/tmp").free / 1e9
            log(f"[{run} part {k}] fetching {Path(fn).name} (free disk reported {free:.0f} GB)")
            path = fetch(fn, raw_dir, log)
            src = Source(run, [path], [])
            if meta is None:
                meta = _init_meta(src, band_rows, "full")
                meta.update(dtype="int16", band_lo=band_lo, band_hi=band_hi or meta["n_bands"])
                band_hi = meta["band_hi"]
            years, doy0 = src.daily_time("tas")
            uy = np.unique(years)
            lat = np.asarray(meta["lat"])
            w = (np.cos(np.deg2rad(lat))[:, None] * np.ones((1, meta["nlon"]))).ravel()
            w = w / w.sum()
            try:
                forc_part = src.forcings(uy, log)
            except ValueError:  # forcings stored separately
                src.extra = [_open(fetch(HF_PREFIX + "forcings_all.nc", raw_dir, log))]
                forc_part = src.forcings(uy, log)
            src.close()
            t0 = time.time()
            gm_days = None
            var_list = list(variables)
            if "tas" not in var_list:
                raise ValueError("'tas' must be among the variables (needed for global-mean temperature)")
            jobs = [(str(path), run, v, str(out / f"{run}__{v}__p{k}.dat"), meta["band_lo"], meta["band_hi"],
                     band_rows, meta["nlat"], meta["nlon"], slab_days, w) for v in var_list]
            with ProcessPoolExecutor(max(1, min(workers, len(jobs)))) as ex:
                for var, n, gm, msgs in ex.map(_write_var_part, jobs):
                    for m in msgs:
                        log("  " + m)
                    if n is None:
                        log(f"  [{run}] {var} missing in {path.name}")
                        continue
                    if n != len(years):
                        raise ValueError(f"{run} {var}: {n} days vs {len(years)}")
                    if gm is not None:
                        gm_days = gm
            log(f"  [{run} part {k}] {len(years)} days {uy[0]}-{uy[-1]}: {var_list} written in {time.time() - t0:.0f}s")
            gmt = np.array([gm_days[years == y].mean() for y in uy])
            prog["time"].setdefault(run, []).append((years, doy0))
            prog["gmt"].setdefault(run, []).append(gmt)
            prog["forc"][run] = _merge_forcings(prog["forc"].get(run), forc_part)
            if not keep_raw:
                path.unlink(missing_ok=True)
            prog["done"].append(key)
            prog["meta"] = meta
            pickle.dump(prog, open(prog_f, "wb"))
        # finalize run
        if run not in meta["runs"]:
            ys = np.concatenate([t[0] for t in prog["time"][run]])
            ds_ = np.concatenate([t[1] for t in prog["time"][run]])
            if not np.all(np.lexsort((ds_, ys)) == np.arange(len(ys))):
                raise ValueError(f"{run}: parts not in time order")
            np.savez(out / f"{run}__time.npz", years=ys, doy0=ds_.astype(np.int16))
            np.save(out / f"{run}__gmt.npy", np.concatenate(prog["gmt"][run]))
            meta["runs"][run] = {"T": int(len(ys)), "parts": [int(len(t[0])) for t in prog["time"][run]],
                                 "first_year": int(ys[0]), "last_year": int(ys[-1])}
            prog["meta"] = meta
            pickle.dump(prog, open(prog_f, "wb"))
    meta["variables"] = list(variables)
    first_ssp = groups[1][0]
    forc = dict(prog["forc"])
    if test[0] not in forc:
        ty = np.load(out / f"{first_ssp}__time.npz")["years"]
        tname, tfiles = test
        tpaths = [fetch(f, raw_dir, log) for f in tfiles]
        tsrc = Source(tname, tpaths, [])
        forc[tname] = tsrc.forcings(np.unique(ty), log)
        tsrc.close()
        if not keep_raw:
            for p in tpaths:
                p.unlink(missing_ok=True)
        prog["forc"] = forc
        pickle.dump(prog, open(prog_f, "wb"))
    return _finish(out, meta, forc, [test[0]], first_ssp, log)