"""
Distributional pattern scaling + weather resampling (the emulator core).

For every grid cell c, variable v, calendar month m and quantile level p_k we model the
(transformed, deseasonalised) daily-value quantile as a linear function of the shared
forcing predictors x_y of year y:

        q_{v,c,m,k}(y) = x_y . B_{v,c,m,k}                      (ridge regression)

The quantile functions are blended smoothly between months, so every day of the year has
its own conditional quantile function Q_y(.; doy).

Weather (internal variability) is NOT predicted.  It is borrowed from the training runs:
each training day is mapped to a forcing-free "knot coordinate" s = Q_{y_train}^{-1}(value)
(a probability-integral transform in quantile-knot space), and an emulated day for target
year y* is produced as  value* = Q_{y*}(s).  Because the same template day is used for all
variables and all grid cells, the emulated fields keep realistic spatial coherence,
day-to-day persistence (spells!) and cross-variable consistency, while their distribution
(mean, variance, tails, wet-day frequency) follows the forcing.

Indices computed on several such members (different template years) and averaged give an
estimate of the forced expectation of each index, which is what the nNSE metric rewards.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_DOY = 365

# quantile levels: dense in the tails, where the extreme indices live
PROBS = np.array([
    0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.965, 0.98,
    0.99,
])

TEMP_LIKE = {"tas", "tasmax", "tasmin", "psl", "huss", "sfcWind"}
POSITIVE = {"huss", "sfcWind"}
PR_ZERO_MM = 1e-3        # mm/day below which precipitation is treated as an exact zero
TAIL_J = 4               # tail scale = mean width of the outermost TAIL_J knot segments
S_CLIP = 30.0            # safety clip on tail coordinates (in tail-scale units)


# ----------------------------------------------------------------------------
# Calendar helpers
# ----------------------------------------------------------------------------
def month_windows(half_width: int = 30) -> np.ndarray:
    """(12, 2*half_width+1) day-of-year indices of a circular window centred on each mid-month."""
    centres = np.round((np.arange(12) + 0.5) * N_DOY / 12 - 0.5).astype(int)
    offs = np.arange(-half_width, half_width + 1)
    return (centres[:, None] + offs[None, :]) % N_DOY


def doy_blend() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = (np.arange(N_DOY) + 0.5) * 12 / N_DOY - 0.5
    m0 = np.floor(mu).astype(int)
    w = (mu - m0).astype(np.float32)
    return m0 % 12, (m0 + 1) % 12, w


M0, M1, WBL = doy_blend()
WIN = month_windows()
BLEND = np.zeros((N_DOY, 12), np.float32)        # daily_knots = BLEND @ monthly knots
BLEND[np.arange(N_DOY), M0] += 1 - WBL
BLEND[np.arange(N_DOY), M1] += WBL


def to_365(arr: np.ndarray, doy0: np.ndarray) -> np.ndarray:
    """Place one year's (n_days, C) values on a 365-day grid (drop Feb-29/doy 366, fill gaps)."""
    out = np.empty((N_DOY,) + arr.shape[1:], dtype=arr.dtype)
    ok = doy0 < N_DOY
    have = np.zeros(N_DOY, bool)
    out[doy0[ok]] = arr[ok]
    have[doy0[ok]] = True
    if not have.all():  # e.g. truncated last year: copy nearest available day
        idx = np.where(have)[0]
        for d in np.where(~have)[0]:
            out[d] = out[idx[np.argmin(np.abs(idx - d))]]
    return out


def from_365(arr365: np.ndarray, doy0: np.ndarray) -> np.ndarray:
    """Inverse of to_365 for an output calendar (doy 366 -> copy of doy 365)."""
    return arr365[np.minimum(doy0, N_DOY - 1)]


# ----------------------------------------------------------------------------
# Variable transforms
# ----------------------------------------------------------------------------
def harmonic_basis(n_harm: int = 3) -> np.ndarray:
    d = np.arange(N_DOY)
    cols = [np.ones(N_DOY)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * d / N_DOY), np.sin(2 * np.pi * k * d / N_DOY)]
    return np.stack(cols, axis=1)


@dataclass
class VarTransform:
    var: str
    clim: np.ndarray | None = None  # (365, C) seasonal climatology removed before quantiles

    def fit_clim(self, years365: list[np.ndarray]):
        """years365: list of (365, C) arrays of the historical run."""
        if self.var not in TEMP_LIKE:
            return self
        mean_doy = np.mean(np.stack(years365), axis=0)  # (365, C)
        Bh = harmonic_basis()
        coef, *_ = np.linalg.lstsq(Bh, mean_doy, rcond=None)
        self.clim = (Bh @ coef).astype(np.float32)
        return self

    def fwd(self, x365: np.ndarray) -> np.ndarray:
        if self.var == "pr":
            x = np.where(x365 < PR_ZERO_MM, 0.0, x365)
            return np.cbrt(x).astype(np.float32)
        return (x365 - self.clim).astype(np.float32)

    def inv(self, z365: np.ndarray) -> np.ndarray:
        if self.var == "pr":
            z = np.maximum(z365, 0.0)
            x = z * z * z
            return np.where(x < PR_ZERO_MM, 0.0, x).astype(np.float32)
        x = z365 + self.clim
        if self.var in POSITIVE:
            x = np.maximum(x, 0.0)
        return x.astype(np.float32)


# ----------------------------------------------------------------------------
# Quantile regression model (one variable, one band of cells)
# ----------------------------------------------------------------------------
def year_window_quantiles(z365: np.ndarray, probs: np.ndarray = PROBS) -> np.ndarray:
    """(365, C) -> (12, K, C) empirical quantiles in each month window (linear interpolation)."""
    S = np.sort(z365[WIN], axis=1)  # (12, n_w, C)
    n_w = S.shape[1]
    pos = probs * (n_w - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_w - 1)
    f = (pos - lo).astype(np.float32)[None, :, None]
    return S[:, lo] * (1 - f) + S[:, hi] * f


class QuantileModel:
    def __init__(self, n_cols: int, n_cells: int, probs: np.ndarray = PROBS):
        self.K = len(probs)
        self.C = n_cells
        self.P = n_cols
        self.XtX = np.zeros((n_cols, n_cols))
        self.XtY = np.zeros((n_cols, 12 * self.K * n_cells))
        self.n = 0
        self.B = None

    def add_year(self, x: np.ndarray, q: np.ndarray):
        self.XtX += np.outer(x, x)
        self.XtY += np.outer(x, q.reshape(-1).astype(np.float64))
        self.n += 1

    def fit(self, ridge: np.ndarray):
        """ridge: (P,) penalties (0 for the intercept), in 'equivalent years'."""
        A = self.XtX + np.diag(ridge)
        self.B = np.linalg.solve(A, self.XtY).reshape(self.P, 12, self.K, self.C).astype(np.float32)
        del self.XtY
        return self

    def knots(self, x: np.ndarray, is_pr: bool) -> np.ndarray:
        """(12, K, C) monotone quantile knots for design row x."""
        q = np.tensordot(x.astype(np.float32), self.B, axes=(0, 0))
        q = np.sort(q, axis=1)  # rearrangement fixes quantile crossing
        if is_pr:
            q = np.maximum(q, 0.0)
        # strictly increasing (tiny ramp) so every segment is invertible
        q = q + (np.arange(self.K, dtype=np.float32) * 1e-6)[None, :, None]
        return q

    @staticmethod
    def daily_knots(qm: np.ndarray) -> np.ndarray:
        """(12, K, C) -> (365, K, C) by linear blending between adjacent months (one BLAS matmul)."""
        K, C = qm.shape[1], qm.shape[2]
        out = BLEND @ qm.reshape(12, K * C).astype(np.float32, copy=False)
        return out.reshape(N_DOY, K, C)


# ----------------------------------------------------------------------------
# PIT (value -> knot coordinate) and inverse
# ----------------------------------------------------------------------------
def _tail_scales(kd: np.ndarray):
    K = kd.shape[1]
    d_lo = (kd[:, TAIL_J] - kd[:, 0]) / TAIL_J
    d_hi = (kd[:, K - 1] - kd[:, K - 1 - TAIL_J]) / TAIL_J
    return d_lo, d_hi


def _gather(kd: np.ndarray, i: np.ndarray):
    """lo = kd[d, i[d,c], c], hi = kd[d, i[d,c]+1, c] via flat indexing (faster than take_along_axis)."""
    n, K, C = kd.shape
    flat = kd.reshape(-1)
    fi = (np.arange(n, dtype=np.int64)[:, None] * K + i) * C + np.arange(C, dtype=np.int64)[None, :]
    return flat[fi], flat[fi + C]


def to_coord(z365: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """
    z365 (365, C), kd (365, K, C) -> continuous knot coordinate s (365, C).
    Inside [q_0, q_{K-1}]: piecewise-linear position between knots.
    Beyond the outer knots: distance measured in units of the local tail scale (mean width of the
    outer TAIL_J segments), so extremes are carried over as 'how many tail-widths beyond the 1st/99th
    percentile' and re-scaled by the target's tail width (captures changes in tail shape).
    """
    K = kd.shape[1]
    idx = (kd <= z365[:, None, :]).sum(axis=1) - 1
    idx = np.clip(idx, 0, K - 2)
    lo, hi = _gather(kd, idx)
    s = idx + (z365 - lo) / (hi - lo)
    d_lo, d_hi = _tail_scales(kd)
    q0, qK = kd[:, 0], kd[:, K - 1]
    s = np.where(z365 > qK, (K - 1) + (z365 - qK) / d_hi, s)
    s = np.where(z365 < q0, (z365 - q0) / d_lo, s)
    return np.clip(s, -S_CLIP, K - 1 + S_CLIP).astype(np.float32)


def from_coord(s: np.ndarray, kd: np.ndarray) -> np.ndarray:
    K = kd.shape[1]
    i = np.clip(np.floor(s).astype(np.int64), 0, K - 2)
    lo, hi = _gather(kd, i)
    z = lo + (s - i) * (hi - lo)
    d_lo, d_hi = _tail_scales(kd)
    z = np.where(s > K - 1, kd[:, K - 1] + (s - (K - 1)) * d_hi, z)
    z = np.where(s < 0, kd[:, 0] + s * d_lo, z)
    return z


def dry_coords(z365: np.ndarray, kd: np.ndarray) -> np.ndarray:
    """
    Knot coordinates for exactly-dry days (z == 0).  All dry days tie at 0, so a value alone
    cannot say how 'deep' into the dry part of the distribution a day is.  We rank dry days by
    their distance to the nearest wet day: days in the middle of long dry spells get the
    lowest coordinates (they stay dry if the climate gets wetter), days next to rain get the
    highest (they are the first to turn wet).  This preserves dry-spell structure (CDD) under
    changing wet-day frequency much better than random tie-breaking.
    Returns s for every day (only meaningful where z == 0).
    """
    n, C = z365.shape
    dry = z365 <= 0
    # coordinate of the top of the zero plateau (knots at ~0 in transformed space)
    s0 = (kd <= 1e-4).sum(axis=1).astype(np.float32)  # (365, C), number of ~zero knots
    # distance to nearest wet day (forward and backward sweeps)
    big = np.int32(10_000)
    dist_f = np.empty((n, C), np.int32)
    run = np.full(C, big, np.int32)
    for t in range(n):
        run = np.where(dry[t], np.minimum(run + 1, big), 0)
        dist_f[t] = run
    dist_b = np.empty((n, C), np.int32)
    run = np.full(C, big, np.int32)
    for t in range(n - 1, -1, -1):
        run = np.where(dry[t], np.minimum(run + 1, big), 0)
        dist_b[t] = run
    depth = np.minimum(dist_f, dist_b).astype(np.float32)

    # rank among the year's dry days -> r in (0, 1]; deeper -> smaller r
    key = np.where(dry, -depth + 1e-3 * np.arange(n)[:, None] / n, np.inf)
    order = np.argsort(key, axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[:, None].repeat(C, 1), axis=0)
    ndry = dry.sum(axis=0).clip(min=1)
    r = (ranks.astype(np.float32) + 0.5) / ndry[None, :]  # deepest dry day -> r ~ 0
    # place inside the plateau [0, s0 - 1] (coordinate range where Q == 0)
    s = r * np.maximum(s0 - 1.0, 0.0)
    # if this day has no zero plateau (wet climate), push slightly below the first knot
    s = np.where(s0 <= 0, -0.5, s)
    return s.astype(np.float32)

# ---- [sprint ClimX 29-10, añadido] suavizado zonal de las pendientes forzadas -------------------
def smooth_slopes(B: np.ndarray, nlon: int, width: int) -> np.ndarray:
    """Media móvil circular en longitud (ancho impar) de las filas no-intercepto de B (P, 12, K, C).
    El intercepto (clima local) no se toca; solo se promedia la respuesta al forzamiento entre vecinos,
    para reducir el ruido de estimarla con una sola realización por escenario."""
    if width <= 1:
        return B
    assert width % 2 == 1, "slope_smooth debe ser impar"
    P, M, K, C = B.shape
    S = B[1:].reshape(P - 1, M, K, C // nlon, nlon).astype(np.float64)
    h = width // 2
    acc = np.zeros_like(S)
    for s in range(-h, h + 1):
        acc += np.roll(S, s, axis=-1)
    B = B.copy()
    B[1:] = (acc / width).reshape(P - 1, M, K, C).astype(B.dtype)
    return B
