# D01: daily tasmax quantile curvature screen

Author: Francisco Angulo de Lafuente. ORCID: 0009-0001-1634-7063.
Preregistered 10 November 2026, before any D01 validation result.

The recovered 5 November neural families predict indices directly. They are not a permitted submission under the current [official rule](https://github.com/IPL-UV/ClimX#at-a-glance). This study instead models the conditional distribution of daily maximum temperature in Kelvin. It is a screening study, not a submission or a measured leaderboard gain.

## Fixed comparison

- Control: the recovered maverickss26 distributional pattern-scaling quantile estimator, original training-year monthly windows, 27 quantiles, seasonal transform, five linear forcing columns and penalties (intercept 0, GHG 1, aerosols 30).
- Intervention: tasmax only. Append three features (standardised fast GHG squared, fast times slow, slow squared), residualised against the original design using training data only, then standardised using training data only. All three additional ridge penalties are 100 equivalent training years. No search over this value or feature family.
- Training scenarios: historical, SSP126 and SSP585. SSP370 targets and global temperature are excluded from all fitting, normalisation, PCA, forcing-timescale selection and feature construction. Validation SSP370 forcings are transform-only. The historical baseline is fit only on historical daily temperatures.
- Evaluation: all available SSP370 days in 2015-2100, full original resolution on latitude bands 3,9,15,21,27,33,39,45 (zero based; 4 rows and 288 longitudes per band). No SSP245 test prediction or index correction.
- Score: twice the trapezoidal integral of quantile pinball loss over original probabilities 0.01 through 0.99, in Kelvin. This is truncated CRPS, lower is better. Evaluate the daily conditional quantiles, not fitted index maps. Cosine-latitude weighting across all evaluated cells; this screen is not the official AR6 land-region index metric.
- Fixed screening gate: at least 0.25% lower truncated CRPS and a positive lower endpoint of a paired 95% circular 10-year moving-block bootstrap (10,000 replicates, seed 20261110; all spatial bands kept together). Serial dependence and nonstationary trends limit this interval. Pixels and years are not counted as independent replicates. All scenarios have appeared in historical research; this is a prospective method comparison, not an untouched external confirmation.
- A screening pass permits preparation of a separate daily-realisation and official-index comparison, with another preregistered gate. It does not permit test inference, official submission, weighted index blending, or replacement of the currently selected official entry.

## Execution contract

One CPU preflight, one numerical/feasibility GPU pilot (band3,32 evenly spaced cells; no validation quality reported), then one full GPU screen only if the pilot passes and JEV recommends admission. Source, runtime and all actually used raw-band bytes are SHA256 pinned and checked before/after. Exact source-file set checked before imports. Pilot and screen each have a persistent single-attempt marker outside source. No automatic restart, resume, extension, or closed-study rerun.

Runtime byte identity covers the resolved Python executable, all Python source/shared libraries under torch, NumPy, pandas, xarray and SciPy, and installed NVIDIA CUDA runtime/NVRTC/BLAS/solver/sparse/JIT-link library packages. Version strings are checked as additional metadata. Bytecode/caches, OS kernel and unrelated system packages are outside this fingerprint. Reads of Python bytecode are redirected to a fresh empty prefix before scientific imports and writes are disabled; the prefix must still be empty at completion. The attempt marker uses atomic exclusive creation and fsync; an incomplete marker also prevents restart. The pilot wall timer starts before scientific imports and source/input checks. After the input pin artifact is added, a final CPU verify run checks the frozen source, input and runtime fingerprints plus numerical fixtures without changing that package; its report must identify the exact manifest admitted for the pilot.

GPU arithmetic must reproduce the original NumPy quantile extraction, float64 ridge solution and daily quantile interpolation on controlled fixtures. Hardware pilot checks CUDA, finite output, peak allocated VRAM <=6GiB and wall time <=300s. Source and input identities must match the CPU preflight. Screening job maximum45m, one GPU, <=8GiB requested VRAM, <=10GiB RAM, four CPU threads. Normal managed servidor-ia queue only; no lane/campaign changes.

An isolated venv under /mnt/datos/proyectos/climx-d01-20261110 inherits existing training packages read-only through its own .pth and installs xarray2026.9.0 only in that new venv. No original environment was changed. Model coefficients and intermediate outputs remain in that project folder; source and aggregate evidence will be backed up after review.
