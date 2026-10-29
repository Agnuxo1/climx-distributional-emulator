"""
Fast, vectorised NumPy implementation of the 15 ClimX extreme indices.

Why re-implement?  The official xclim-based code (src/indices_xclim.py) is correct
but slow and memory hungry on 86 years x 192 x 288 x many ensemble members.  This
module computes the same quantities year-by-year on (days, cells) arrays.  It is
tested for exact agreement with the official code in tests/test_indices_vs_official.py.

Conventions
-----------
* Temperatures in Kelvin, precipitation in mm/day (= kg m-2 s-1 * 86400).
* One calendar year at a time: arrays of shape (n_days, n_cells).
* `doy0` is the 0-based day-of-year of each day (0..364 for a noleap calendar).
* Percentile thresholds are (365, n_cells) arrays from `percentile_doy`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KG_M2_S_TO_MM_DAY = 86400.0
T0 = 273.15

INDEX_NAMES = (
    "TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
    "Rx5day", "CDD", "CWD", "R95p", "SDII", "R10mm",
)


# ----------------------------------------------------------------------------
# Run-length helpers (vectorised over cells, loop over <=366 days)
# ----------------------------------------------------------------------------
def _forward_runs(mask: np.ndarray) -> np.ndarray:
    """r[t] = length of the True-run ending at t (0 where mask is False)."""
    out = np.zeros(mask.shape, dtype=np.int16)
    run = np.zeros(mask.shape[1:], dtype=np.int16)
    for t in range(mask.shape[0]):
        run = (run + 1) * mask[t]
        out[t] = run
    return out


def _backward_runs(mask: np.ndarray) -> np.ndarray:
    """s[t] = length of the True-run starting at t (0 where mask is False)."""
    out = np.zeros(mask.shape, dtype=np.int16)
    run = np.zeros(mask.shape[1:], dtype=np.int16)
    for t in range(mask.shape[0] - 1, -1, -1):
        run = (run + 1) * mask[t]
        out[t] = run
    return out


def longest_run(mask: np.ndarray) -> np.ndarray:
    run = np.zeros(mask.shape[1:], dtype=np.int16)
    best = np.zeros(mask.shape[1:], dtype=np.int16)
    for t in range(mask.shape[0]):
        run = (run + 1) * mask[t]
        np.maximum(best, run, out=best)
    return best.astype(np.float32)


def windowed_run_count(mask: np.ndarray, window: int) -> np.ndarray:
    """Number of days belonging to runs of at least `window` consecutive True."""
    r = _forward_runs(mask)
    s = _backward_runs(mask)
    full = r.astype(np.int32) + s - 1
    return ((full >= window) & mask).sum(axis=0).astype(np.float32)


# ----------------------------------------------------------------------------
# Percentile thresholds (replicates xclim.core.calendar.percentile_doy)
# ----------------------------------------------------------------------------
def percentile_doy(base: np.ndarray, doy0: np.ndarray, per: float, window: int = 5,
                   n_doy: int = 365) -> np.ndarray:
    """
    base : (T, C) daily values over the base period (contiguous days)
    doy0 : (T,) 0-based day-of-year of each day
    Returns (n_doy, C) thresholds.

    xclim: rolling(center=True, min_periods=1, window) -> stack (year, window) ->
    percentile (Hyndman & Fan type 8, alpha=beta=1/3) ignoring NaN.  Values outside the
    series edges are NaN and therefore ignored.
    """
    T, C = base.shape
    half = window // 2
    padded = np.full((T + 2 * half, C), np.nan, dtype=np.float64)
    padded[half:half + T] = base
    out = np.empty((n_doy, C), dtype=np.float64)
    for d in range(n_doy):
        idx = np.nonzero(doy0 == d)[0]
        if idx.size == 0:
            out[d] = np.nan
            continue
        offs = (idx[:, None] + np.arange(window)[None, :]).ravel()  # into padded
        # NaN rows only occur at the series edges and span all cells: drop them so the
        # (much faster) vectorised np.percentile gives the same result as nanpercentile.
        offs = offs[(offs >= half) & (offs < half + T)]
        sample = padded[offs]  # (n_years*window, C)
        if np.isnan(sample).any():
            out[d] = np.nanpercentile(sample, per, axis=0, method="median_unbiased")
        else:
            out[d] = np.percentile(sample, per, axis=0, method="median_unbiased")
    return out


# ----------------------------------------------------------------------------
# Index computation for one year
# ----------------------------------------------------------------------------
@dataclass
class Thresholds:
    tx90: np.ndarray  # (365, C) K
    tn10: np.ndarray  # (365, C) K
    pr95: np.ndarray  # (365, C) mm/day


@dataclass
class YearState:
    """Carries the last 4 days of precipitation to make Rx5day continuous across years."""
    pr_tail: np.ndarray | None = None


def _gsl(tas: np.ndarray, doy0: np.ndarray, thresh: float = T0 + 5.0, window: int = 6,
         mid_doy0: int | None = None) -> np.ndarray:
    """
    xclim growing_season_length (op '>=', mid_date '07-01', window 6), via rl.season_length:
      start: first day of the first run of >= window days with tas >= thresh, and start < mid date
      end:   first day (on/after mid date, after start) of the first run of >= window days with
             tas < thresh; if none -> season lasts to the end of the period.
      length = end - start ; 0 if no valid start.
    """
    n, C = tas.shape
    if mid_doy0 is None:
        mid_doy0 = 181  # 1 July in a 365-day calendar (0-based)
    mid_idx = int(np.searchsorted(doy0, mid_doy0))
    warm = tas >= thresh
    s_warm = _backward_runs(warm)
    cold = ~warm
    s_cold = _backward_runs(cold)

    start_ok = s_warm >= window
    big = n + 10
    t_idx = np.arange(n)[:, None]
    start = np.where(start_ok, t_idx, big).min(axis=0)
    has_start = start < mid_idx

    end_ok = (s_cold >= window) & (t_idx >= mid_idx) & (t_idx > start[None, :])
    end = np.where(end_ok, t_idx, big).min(axis=0)
    end = np.where(end >= big, n, end)
    length = np.where(has_start, end - start, 0)
    return length.astype(np.float32)


def year_indices(tasmax: np.ndarray, tasmin: np.ndarray, tas: np.ndarray, pr_mm: np.ndarray,
                 doy0: np.ndarray, thr: Thresholds, state: YearState | None = None) -> dict:
    """Compute all 15 annual indices for one year. Arrays (n_days, C)."""
    out = {}
    d = np.clip(doy0, 0, thr.tx90.shape[0] - 1)
    # compare in the data dtype (xclim compares float32 data with a float32-cast threshold)
    c = lambda v, a: a.dtype.type(v)

    out["TXx"] = tasmax.max(axis=0).astype(np.float32)
    out["TNn"] = tasmin.min(axis=0).astype(np.float32)
    out["SU"] = (tasmax > c(T0 + 25.0, tasmax)).sum(axis=0).astype(np.float32)
    out["TR"] = (tasmin > c(T0 + 20.0, tasmin)).sum(axis=0).astype(np.float32)
    out["FD"] = (tasmin < c(T0, tasmin)).sum(axis=0).astype(np.float32)
    out["ID"] = (tasmax < c(T0, tasmax)).sum(axis=0).astype(np.float32)
    out["WSDI"] = windowed_run_count(tasmax > thr.tx90[d], 6)
    out["CSDI"] = windowed_run_count(tasmin < thr.tn10[d], 6)
    out["GSL"] = _gsl(tas, doy0, thresh=c(T0 + 5.0, tas))

    # Rx5day: max of 5-day running sums whose last day falls in this year.
    if state is not None and state.pr_tail is not None:
        ext = np.concatenate([state.pr_tail, pr_mm], axis=0)
        k0 = state.pr_tail.shape[0]
    else:
        ext, k0 = pr_mm, 0
    cs = np.concatenate([np.zeros((1,) + ext.shape[1:], dtype=np.float64), np.cumsum(ext, axis=0, dtype=np.float64)])
    sums = cs[5:] - cs[:-5]  # sums[i] = ext[i..i+4], ends at i+4
    first_end = max(k0, 4)
    sums = sums[first_end - 4:]
    out["Rx5day"] = sums.max(axis=0).astype(np.float32)
    if state is not None:
        state.pr_tail = pr_mm[-4:].copy()

    out["CDD"] = longest_run(pr_mm < c(1.0, pr_mm))
    wet = pr_mm >= c(1.0, pr_mm)
    out["CWD"] = longest_run(wet)
    out["R95p"] = np.where(pr_mm > thr.pr95[d], pr_mm, 0.0).sum(axis=0).astype(np.float32)
    nwet = wet.sum(axis=0)
    wsum = np.where(wet, pr_mm, 0.0).sum(axis=0)
    out["SDII"] = np.where(nwet == 0, 0.0, wsum / np.maximum(nwet, 1)).astype(np.float32)
    out["R10mm"] = (pr_mm >= c(10.0, pr_mm)).sum(axis=0).astype(np.float32)
    return out


def compute_thresholds(tasmax_base, tasmin_base, pr_base_mm, doy0_base) -> Thresholds:
    return Thresholds(
        tx90=percentile_doy(tasmax_base, doy0_base, 90.0).astype(np.float32),
        tn10=percentile_doy(tasmin_base, doy0_base, 10.0).astype(np.float32),
        pr95=percentile_doy(pr_base_mm, doy0_base, 95.0).astype(np.float32),
    )