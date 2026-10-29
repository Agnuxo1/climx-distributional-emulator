"""
Forcing -> predictor model.

Turns the raw ClimX forcings (5 well-mixed GHG concentrations, 5 spatial aerosol emission
fields) into a small predictor matrix X (years x p) that is SHARED by every grid cell.
Each grid cell then learns its own linear response to X (pattern scaling).

    F_ghg(t)   simplified effective radiative forcing of the GHGs (Myhre-style formulas)
    G_f, G_s   F_ghg low-pass filtered with a fast (~years) and slow (~centuries) timescale:
               a two-box (fast mixed layer / deep ocean) response. Local patterns differ
               between fast and slow warming (e.g. Southern Ocean, North Atlantic), which is
               why two components beat a single global-mean-temperature regressor.
    A_k        leading principal components of the (standardised, log-scaled) aerosol emission
               fields, low-pass filtered with the fast timescale.  These let each cell learn a
               local aerosol response (South/East Asia cooling, monsoon precipitation shifts).

The two timescales are chosen by leave-one-scenario-out fitting of global-mean annual tas.
Everything is fit on training scenarios only (historical + SSP1-2.6/3-7.0/5-8.5) so the
SSP2-4.5 test predictors are a pure function of its forcings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GHG_NAMES = ["CO2_LBC", "CH4_LBC", "N2O_LBC", "CFC11eq_LBC", "CF2CL2_LBC"]
AER_NAMES = ["SO2", "SO4_PR", "BC_AX", "BC_N", "OM_NI"]


# ----------------------------------------------------------------------------
# Radiative forcing proxy
# ----------------------------------------------------------------------------
def _to_ppm_ppb(name: str, x: np.ndarray) -> np.ndarray:
    """Best-effort unit normalisation: CO2 -> ppm, CH4/N2O -> ppb, halocarbons -> ppb."""
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmedian(x)
    if name.startswith("CO2"):
        if m < 1e-2:   # mole fraction
            return x * 1e6
        return x
    if name.startswith(("CH4", "N2O")):
        if m < 1e-4:   # mole fraction
            return x * 1e9
        if m < 10:     # ppm
            return x * 1e3
        return x
    # halocarbons
    if m < 1e-6:       # mole fraction
        return x * 1e9
    if m > 20:         # ppt
        return x / 1e3
    return x


def ghg_erf(ghg: dict[str, np.ndarray], ref: dict[str, float] | None = None) -> tuple[np.ndarray, dict]:
    """Simplified ERF (W m-2) relative to `ref` (default: first value, i.e. 1850)."""
    conc = {k: _to_ppm_ppb(k, v) for k, v in ghg.items() if k in GHG_NAMES}
    if ref is None:
        ref = {k: float(v[0]) for k, v in conc.items()}
    F = np.zeros_like(next(iter(conc.values())))
    if "CO2_LBC" in conc:
        F += 5.35 * np.log(conc["CO2_LBC"] / ref["CO2_LBC"])
    if "CH4_LBC" in conc:
        F += 0.036 * (np.sqrt(conc["CH4_LBC"]) - np.sqrt(ref["CH4_LBC"]))
    if "N2O_LBC" in conc:
        F += 0.12 * (np.sqrt(conc["N2O_LBC"]) - np.sqrt(ref["N2O_LBC"]))
    if "CFC11eq_LBC" in conc:
        F += 0.26 * (conc["CFC11eq_LBC"] - ref["CFC11eq_LBC"])
    if "CF2CL2_LBC" in conc:
        F += 0.32 * (conc["CF2CL2_LBC"] - ref["CF2CL2_LBC"])
    return F, ref


def lowpass(x: np.ndarray, tau: float) -> np.ndarray:
    """First-order response y_t = a y_{t-1} + (1-a) x_t, a = exp(-1/tau); y_0 = x_0 (equilibrium start)."""
    a = np.exp(-1.0 / tau)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * y[t - 1] + (1 - a) * x[t]
    return y


# ----------------------------------------------------------------------------
# Scenario container
# ----------------------------------------------------------------------------
@dataclass
class ScenarioForcing:
    """Annual forcings for a full trajectory (historical prepended for SSPs)."""
    name: str
    years: np.ndarray                      # (n,)
    ghg: dict[str, np.ndarray]             # gas -> (n,)
    aer: dict[str, np.ndarray]             # species -> (n, n_cells) annual mean emission fields
    own_years: np.ndarray | None = None    # years that belong to this scenario (subset of `years`)

    def __post_init__(self):
        if self.own_years is None:
            self.own_years = self.years


def concat_hist(hist: ScenarioForcing, ssp: ScenarioForcing) -> ScenarioForcing:
    """Prepend historical forcing to an SSP so the low-pass filters see the full history."""
    keep = hist.years < ssp.years[0]
    return ScenarioForcing(
        name=ssp.name,
        years=np.concatenate([hist.years[keep], ssp.years]),
        ghg={k: np.concatenate([hist.ghg[k][keep], ssp.ghg[k]]) for k in ssp.ghg},
        aer={k: np.concatenate([hist.aer[k][keep], ssp.aer[k]]) for k in ssp.aer},
        own_years=ssp.years,
    )


# ----------------------------------------------------------------------------
# Predictor model
# ----------------------------------------------------------------------------
@dataclass
class PredictorModel:
    n_aer_pcs: int = 2
    tau_fast_grid: tuple = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    tau_slow_grid: tuple = (20.0, 40.0, 70.0, 100.0, 150.0, 250.0, 400.0)
    two_timescales: bool = True
    # fitted
    tau_fast: float = 4.0
    tau_slow: float = 100.0
    erf_ref: dict = field(default_factory=dict)
    aer_mu: dict = field(default_factory=dict)
    aer_sd: dict = field(default_factory=dict)
    aer_loadings: np.ndarray | None = None   # (n_features, K)
    aer_feat_mean: np.ndarray | None = None
    aer_weights: np.ndarray | None = None    # (n_cells,) cos-lat weights used in PCA
    x_mean: np.ndarray | None = None
    x_std: np.ndarray | None = None
    gmt_coef: np.ndarray | None = None
    names: list = field(default_factory=list)

    # --- aerosol features -------------------------------------------------
    def _aer_feature_matrix(self, sf: ScenarioForcing) -> np.ndarray:
        cols = []
        for k in AER_NAMES:
            if k not in sf.aer:
                continue
            x = np.log10(np.maximum(sf.aer[k], 0) + self.aer_mu[k])  # log-scale with floor
            cols.append((x - self.aer_sd[k][0]) / self.aer_sd[k][1] * self.aer_weights[None, :])
        return np.concatenate(cols, axis=1)  # (n_years, n_species * n_cells)

    def _fit_aerosol_pca(self, scen: list[ScenarioForcing], cell_weights: np.ndarray):
        self.aer_weights = np.sqrt(cell_weights / cell_weights.sum())
        for k in AER_NAMES:
            if k not in scen[0].aer:
                continue
            allv = np.concatenate([s.aer[k][np.isin(s.years, s.own_years)] for s in scen])
            pos = allv[allv > 0]
            floor = float(np.percentile(pos, 5)) if pos.size else 1e-20
            self.aer_mu[k] = floor
            lx = np.log10(np.maximum(allv, 0) + floor)
            self.aer_sd[k] = (float(lx.mean()), float(lx.std() + 1e-12))
        F = np.concatenate([self._aer_feature_matrix(s)[np.isin(s.years, s.own_years)] for s in scen])
        self.aer_feat_mean = F.mean(axis=0)
        Fc = F - self.aer_feat_mean
        # PCA via the (years x years) Gram matrix: cheap even for 192x288x5 features
        G = Fc @ Fc.T
        w, V = np.linalg.eigh(G)
        order = np.argsort(w)[::-1][: self.n_aer_pcs]
        U = V[:, order] / np.sqrt(np.maximum(w[order], 1e-12))
        self.aer_loadings = Fc.T @ U  # (n_feat, K) unit-norm directions

    def _aer_pcs(self, sf: ScenarioForcing) -> np.ndarray:
        if self.n_aer_pcs == 0:
            return np.zeros((len(sf.years), 0))
        F = self._aer_feature_matrix(sf) - self.aer_feat_mean
        return F @ self.aer_loadings

    # --- raw (unstandardised) features -------------------------------------
    def _raw(self, sf: ScenarioForcing, tau_f: float, tau_s: float) -> np.ndarray:
        F, _ = ghg_erf(sf.ghg, self.erf_ref)
        cols = [lowpass(F, tau_f)]
        if self.two_timescales:
            cols.append(lowpass(F, tau_s))
        A = self._aer_pcs(sf)
        for k in range(A.shape[1]):
            cols.append(lowpass(A[:, k], tau_f))
        X = np.stack(cols, axis=1)
        return X[np.isin(sf.years, sf.own_years)]

    def fit(self, scen: list[ScenarioForcing], gmt: list[np.ndarray], cell_weights: np.ndarray):
        """
        scen : training scenarios (historical + SSPs with history prepended via concat_hist)
        gmt  : matching global-mean annual tas for each scenario's own years
        """
        _, self.erf_ref = ghg_erf(scen[0].ghg)
        if self.n_aer_pcs:
            self._fit_aerosol_pca(scen, cell_weights)

        # choose timescales by leave-one-scenario-out prediction of global-mean tas
        best = (np.inf, None)
        grid_s = self.tau_slow_grid if self.two_timescales else (1.0,)
        for tf in self.tau_fast_grid:
            for ts in grid_s:
                if self.two_timescales and ts <= tf * 3:
                    continue
                Xs = [np.column_stack([np.ones(len(g)), self._raw(s, tf, ts)]) for s, g in zip(scen, gmt)]
                err = 0.0
                for i in range(len(scen)):
                    if scen[i].name == "historical":
                        continue  # every SSP needs history; hold out SSPs only
                    Xtr = np.concatenate([x for j, x in enumerate(Xs) if j != i])
                    ytr = np.concatenate([g for j, g in enumerate(gmt) if j != i])
                    c, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
                    err += np.mean((Xs[i] @ c - gmt[i]) ** 2)
                if err < best[0]:
                    best = (err, (tf, ts))
        self.tau_fast, self.tau_slow = best[1]
        self.loso_gmt_mse = best[0]

        Xall = np.concatenate([self._raw(s, self.tau_fast, self.tau_slow) for s in scen])
        self.x_mean = Xall.mean(axis=0)
        self.x_std = Xall.std(axis=0) + 1e-12
        Xd = np.column_stack([np.ones(len(Xall)), (Xall - self.x_mean) / self.x_std])
        self.gmt_coef, *_ = np.linalg.lstsq(Xd, np.concatenate(gmt), rcond=None)
        self.names = (["ghg_fast"] + (["ghg_slow"] if self.two_timescales else [])
                      + [f"aer_pc{k + 1}" for k in range(self.n_aer_pcs)])
        return self

    def transform(self, sf: ScenarioForcing) -> np.ndarray:
        """Design matrix (n_own_years, 1 + p) with an intercept column first."""
        X = (self._raw(sf, self.tau_fast, self.tau_slow) - self.x_mean) / self.x_std
        return np.column_stack([np.ones(len(X)), X])

    def predict_gmt(self, sf: ScenarioForcing) -> np.ndarray:
        return self.transform(sf) @ self.gmt_coef

    def group_of_columns(self) -> np.ndarray:
        """0 = intercept, 1 = GHG, 2 = aerosol (for group-wise ridge penalties)."""
        g = [0, 1] + ([1] if self.two_timescales else []) + [2] * self.n_aer_pcs
        return np.array(g)