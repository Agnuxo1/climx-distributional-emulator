# Reproduce the CPU checks and prepare another audited run

The original `source/` and its manifests are immutable execution artifacts. This guide creates a separate reproduction directory, changes only its three storage/cache/output paths, and assigns that copy a new source identity. The scientific algorithms and protocol stay unchanged. The helper runs CPU modes only; the Kaggle notebook provides the source ZIP rather than re-running the study.

## 1. Install a CPU environment and replay the numerical checks

Use Python3.11 or later in a new virtual environment. From the extracted archive root, on Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python tools/reproduce_cpu.py fixtures --work /tmp/climx-d01-reproduction
```

On a Windows computer, run these Linux commands inside WSL2 or another Linux environment; the archived runner uses the Unix `resource` module. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to select the appropriate CPU or CUDA build for that Linux environment. `requirements-cpu.txt` gives the API floor, not a claim that every version combination was tested. The exact audited versions and original six-fixture result are recorded in `CPU_VERIFICATION.json`. The helper creates its own `cpu-fixtures/PREFLIGHT.json`, including its current runtime versions and new source-manifest SHA256. It preserves the archived source. No competition observations or GPU are needed for these fixtures.

The six checks cover quantile extraction, the linear ridge solution, the quadratic ridge solution, daily interpolation, pinball integration and training-only curvature standardisation. Passing them establishes software arithmetic, not a climate-quality gain or an official score.

## 2. Build the private full-resolution store

This step is for registered ClimX participants using the organizer-provided data under the [competition rules](https://www.kaggle.com/competitions/climx-main-track/rules). The full release is approximately200GB; processing requires additional disk space and time. Keep observations, caches and predictions private. This download is separate from the small data-free CPU replay above.

Install the two additional data-access packages in the same project environment:

```bash
.venv/bin/python -m pip install netCDF4 huggingface_hub
```

Create `build_private_store.py` outside the immutable source, using the original public data-access implementation:

```python
import sys
sys.path.insert(0, 'source')
from climx_sota.data import INDEX_VARS, repack_remote

if __name__ == '__main__':
    repack_remote(
        '/your/data-drive/climx/store_full',
        '/your/data-drive/climx/raw',
        band_rows=4, variables=INDEX_VARS,
        keep_raw=True, workers=4,
    )
```

Run it from the extracted archive root with the project Python. `repack_remote` uses the organizer's full-resolution Hugging Face release and original int16 codec. It retains raw files with `keep_raw=True`. The output needs the original192x288grid,48bands, all four training scenarios, forcings, annual global temperatures and calendars. This build path is provided from the retained engine; a fresh full200GBdownload was not repeated for this audit because the complete store was already present.

## 3. Regenerate input/runtime pins and verify the final copy

For the full study, use a CUDA-capable PyTorch environment with its installed NVIDIA library packages and select the version appropriate to your GPU using the official guide. The following command still uses CPU only. It fingerprints the private store's fixed40 band slices, input files and critical runtime files, writes new `INPUTS.json` into the reproduction copy, rebuilds its source manifest, then executes the final verification against that exact identity:

```bash
.venv/bin/python tools/reproduce_cpu.py preflight \
  --work /your/data-drive/climx-d01-reproduction \
  --store /your/data-drive/climx/store_full
```

The resulting `final-cpu-verify/PREFLIGHT.json` must say `passed:true` and match the new source/input manifest hashes. It does not need to match the original machine's runtime hashes. Changing data, runtime or source requires a new reviewed reproduction identity; retain the original archive and all negative or failed results.

## 4. GPU admission and scientific result

The frozen original GPU pilot is queued, with no GPU result in this archive. For another run, use the copied source directory with the normal resource-managed GPU queue, the new manifest SHA, and the protocol's fixed bounds. The runner modes are `pilot` and, only after a passing pilot, `screen`; the respective wall caps are10m and45m. Single-attempt markers live under the copied work directory's `results/` folder. The copied runner's interpreter must be the project environment containing CUDA PyTorch. Keep the prescribed parameters, validation split and gate unchanged.

A screening pass allows preparation of a separate daily-realisation and official-index comparison. Truncated CRPS on daily tasmax quantiles does not establish a leaderboard improvement or authorise a test submission.
