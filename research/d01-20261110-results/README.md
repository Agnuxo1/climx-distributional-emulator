# D01 result: retained control, no candidate promotion

Francisco Angulo de Lafuente · [ORCID 0009-0001-1634-7063](https://orcid.org/0009-0001-1634-7063). Apache-2.0 source and aggregate evidence.

The fixed quadratic-greenhouse-gas daily-tasmax screen completed successfully on all eight preregistered latitude bands and 86 SSP370 validation years. The control is retained: the observed reduction is below the fixed promotion threshold and its paired interval includes zero. This is a completed negative screening result, not a new official score or submission.

| Quantity | Verified value |
|---|---:|
| Linear-control truncated CRPS (K; lower is better) | 1.147990597042 |
| Quadratic-GHG truncated CRPS (K) | 1.147817949760 |
| Reduction (K) | 0.000172647282 |
| Relative reduction | 0.015039% |
| Required relative reduction | at least 0.25% |
| Paired 95% 10-year block-bootstrap interval for reduction (K) | [-0.000426511220, 0.000845747464] |
| Fixed gate | both conditions failed |

The independent vectorised replay reproduces the original weighted losses, 10000-replicate bootstrap (seed 20261110) interval and gate to absolute tolerance 1e-12. Four additional software fixtures check identical predictions, a positive gain below the quarter-percent threshold, a positive gain above it and unequal spatial/time weights. Raw aggregate-loss SHA256: `92a2c27289ce5e091240b972171337d6415f3f2dcebb2602d6244591d0d2c019`.

## Measured execution and identity

The CUDA feasibility pilot passed six arithmetic checks, with 32 fixed cells and 365 days; it reported no validation-quality selection. The one full screen then completed in 205.5621 runner seconds (206.7395 native seconds), peak allocated CUDA memory 0.26576GiB and peak process RSS 4.57804GiB. All 15 source files, the pinned input/runtime identities and all eight saved coefficient hashes were checked. The [resource amendment](ADMISSION_AMENDMENT.md) tightened only the unstarted reservation's wall cap before execution; the scientific protocol remained fixed.

The complete [protocol](PROTOCOL.md), original result, aggregate per-year/band paired losses, collection checks and independent replay are included. The original [CPU source archive](../d01-20261110/README.md) and its reproduction instructions remain unchanged. The source manifest is `d4da15654dbf0116ee101a79c3ad5aa769d36d16e5e2ba84b87ff2118c6f3551`.

## Replay the reported result

With NumPy installed, from this directory in the Git repository:

```bash
python verify_statistics.py --result-dir .
```

For standalone ZIP downloads, extract the original source archive as well and pass its runner location:

```bash
python verify_statistics.py --result-dir . --source-runner /path/to/extracted-source/source/run.py
```

This reads only aggregate losses and the frozen summary function's AST. It performs no model inference, climate-data download or GPU work. The associated CPU-only Kaggle notebook also materialises both archives and independently replays these statistics.

## Interpretation and next action

This prospective comparison uses selected latitude bands, daily maximum-temperature quantiles and cosine-latitude weighting across those cells. It is distinct from the official AR6 land-region metric over 15 extreme indices. The scenarios have appeared in historical research; the 10-year resampling interval has limitations under nonstationary climate trends. A small positive sample reduction with an interval spanning zero does not establish a reliable improvement.

The existing official baseline entry 57777154 remains COMPLETE with public score 0.03070; control 57777260 is COMPLETE with 0.02852. No test prediction, official-index candidate or new submission was generated from D01. The experiment is closed and will not be repeated or have its gate relaxed. Future work needs a genuinely new preregistered hypothesis; the recovered index-only correction families remain outside the permitted daily-field candidate path.

This packet contains source/protocol and aggregate evaluation evidence. It contains no climate observations, prediction maps, trained weights, credentials or private contact details. The original forecasting engine attribution to maverickss26 and its Apache-2.0 licence remain in the original source archive.
