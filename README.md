# ClimX distributional emulator

A reproducible adaptation of the credited public distributional pattern-scaling emulator for the [ClimX main track](https://www.kaggle.com/competitions/climx-main-track), with bounded-memory member generation and an experimental longitude-smoothing option.

## Provenance and measured status

The forecasting engine comes from **maverickss26/climx-sota-main-track**. It is not represented as a newly invented model. [NOTICE](NOTICE) records that source and changes. The official [IPL-UV/ClimX](https://github.com/IPL-UV/ClimX) code computes submission indices. Our existing pixelflow/reservoir technologies are candidate hybridisations; **no reservoir contribution has yet been implemented or measured in this engine**.

| Experiment/check | Recorded outcome | Scope |
|---|---|---|
| Base, 16 members, 8 bands, SSP370 held out | approximately 0.1681 | Approximate band score without AR6 regions; not official leaderboard |
| Slope smoothing 5, matched panel | approximately 0.1680 | No demonstrated gain over base; exploratory |
| Parallel refactor, original thread mismatch |13/15 indices close; R95p differed0.019 and SDII0.004 | Technical mismatch retained |
| Corrected identical OMP=2 verification |15/15 indices difference0.0 on band0 | Numerical equivalence on one band; not model-quality gain |

The 64-member prediction is running on servidor-ia, job 1fc48030296c4f63. Earlier interrupted runs and their retained band caches are technical history, not independent quality trials. There is no completed prediction, official score or measured own-technology gain in this snapshot.

## Reproduce

Linux, Python 3.11+, sufficient RAM and disk for full climate data. Use the competition's allowed daily fields and forcing files. Do not commit them.

```bash
python -m pip install -r requirements.txt
python run_sota.py --help
python run_sota.py validate --holdout 370 --members 16
```

The inherited server driver explicitly expects `/home/ai/proyectos/climx/{data/full,store_full,cache_bands}` and completion records in `data/descarga.log`. Configure those locations before a new run; `repack` builds the band store. `predict_go3.sh` records bounded-memory/parallel settings. `verify_gen.py` compares a frozen serial and parallel band cache. Dependencies are a declared environment, not a complete frozen lockfile.

The official rules require generating daily fields first, then computing indices with official code. Index-space score manipulation is not an authorised shortcut. Read [research protocol](docs/RESEARCH_PROTOCOL.md); future hybridisation needs a fixed-control ablation and independent confirmation.

## Licence

Apache-2.0 adaptation, retaining [LICENSE](LICENSE) and [NOTICE](NOTICE). Bundled official scoring code in `ClimX/` retains its separate MIT licence. No competition data or trained weights are redistributed.
