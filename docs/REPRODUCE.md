# Reproduce experiments

There are three different objectives: verify released checkpoints on the original test split; retrain using the paper's method; reproduce all published means/standard deviations and analyses exactly. Only the first-stage binary checkpoint verification is currently complete. The method's training entry point is usable once features are supplied; exact paper reproduction has unresolved artifacts/implementation details recorded below.

## 1. Install and identify artifacts

Install with `python -m pip install -e .`; use `.[analysis]` for historical notebook analysis dependencies. Record the environment, source revision, checkpoint checksum and experiment/seed metadata.

Required historical artifacts are the original train/test tables, the corresponding features, and matched PD/MP/FP weights for each hierarchy run; baseline runs need their baseline weights. `checkpoints/manifest.json` currently inventories PD only, without established experiment/seed provenance.

## 2. Check the original data split

```bash
gehirnet audit --train-table data/metadata/train_set.csv --test-table data/metadata/test_set.csv --data-root data/features/original
```

The current workspace reports 10,807 training and 2,702 testing segments; test files exist but training files do not resolve at the configured feature root. There are 1,454 overlapping `(Dataset, ID)` keys. Determine source-specific participant IDs before calling an evaluation speaker-independent.

To compare the historical result, preserve its original split and labels. If you create a participant-disjoint split, label that as a new protocol and run participant-grouped CV using a separately implemented splitter; the extracted trainer still uses segment-stratified CV.

## 3. Prepare features, if original features are unavailable

Obtain the four source datasets according to their access terms and construct a recording list using `data/templates/recordings.example.csv`. Resolve the paper's exact inclusion/exclusion list before claiming the resulting dataset is identical. The current helper supports a new curated recording list:

```bash
gehirnet prepare --table data/metadata/train_recordings.csv --audio-root data/raw --output-root data/features/train
gehirnet prepare --table data/metadata/test_recordings.csv --audio-root data/raw --output-root data/features/test
```

Each output contains `.npy` files and `segments.csv`. Configure training `table` as `data/features/train/segments.csv`, `data_root` as `data/features/train`, and evaluation accordingly for test. Extra identity metadata is preserved. Failed preparation leaves a partial table rather than a completed CSV; fix the input and choose a new output directory.

The helper cannot recover the exact original segment split from a raw recording list. Generated filenames, manual exclusions and audio round-trip quantization differ from historical artifacts. Use original `.npy` features and canonical split lists for the strongest checkpoint comparison.

## 4. Generate the experiment configs

For original nonaugmented features at the original feature-relative paths:

```bash
python scripts/make_configs.py --experiment baseline --table data/metadata/train_set.csv --data-root data/features/original --config-dir outputs/configs/baseline --device cuda
python scripts/make_configs.py --experiment hierarchical --table data/metadata/train_set.csv --data-root data/features/original --config-dir outputs/configs/hierarchical --device cuda
```

This creates three baseline configs and nine hierarchy configs: PD/MP/FP for seeds 40, 41, 42. It does not launch training. For prepared features, replace table/root arguments with the values in step 3.

The four paper experiments map to:

| Paper | Config experiment | Tasks/data |
| --- | --- | --- |
| Exp 1 | `baseline` | Baseline on original training features |
| Exp 2 | `hierarchical` | PD, MP, FP on original training features |
| Exp 3.1 | `resampling` | PD, MP, FP on their separately balanced resampled training tables |
| Exp 3.2 | `timewarp` | PD, MP, FP on their separately balanced time-warped training tables |

For augmentation, first use the archived balancing workflows or `augmentation.py` primitives on **training audio only**. Balance PD at the sex/health target level and MP/FP across their six disease classes as described in the paper; preserve parent identities and augmentation choices. Full balancing/provenance generation is not automated by the extracted trainer. All stage tables in a generated experiment resolve from a common feature root:

```bash
python scripts/make_configs.py --experiment timewarp --pd-table data/metadata/timewarp_pd.csv --mp-table data/metadata/timewarp_mp.csv --fp-table data/metadata/timewarp_fp.csv --data-root data/features --config-dir outputs/configs/timewarp --device cuda
```

Use `--experiment resampling` and its three stage tables for Exp 3.1. The generator rejects missing explicit stage tables for augmented experiments, avoiding accidental nonaugmented runs being labeled augmented.

## 5. Train each generated config

Example for seed 42:

```bash
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_pd.json
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_mp.json
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_fp.json
```

Repeat for 40 and 41 and each experiment. Every task searches 18 combinations of epochs/batch size/learning rate, each with five validation folds: 90 fold training runs and one selected final retraining. This is GPU-scale experiment work, not part of model installation or inference. Training defaults to ImageNet initialization.

Outputs are `outputs/runs/EXPERIMENT/seedSEED/TASK/model.pth` and `training.json`. The latter includes selected parameters, fold metrics, class order and fold paths. Existing trained outputs are never overwritten. The extracted RNG sequence differs from historical notebooks; matching the method does not guarantee bit-identical training or identical published metrics.

## 6. Evaluate the same untouched test split

For a matched hierarchy, seed 42:

```bash
gehirnet evaluate --table data/metadata/test_set.csv --data-root data/features/original --pd outputs/runs/hierarchical/seed42/pd/model.pth --mp outputs/runs/hierarchical/seed42/mp/model.pth --fp outputs/runs/hierarchical/seed42/fp/model.pth --output outputs/metrics/hierarchical_seed42.json
```

For baseline:

```bash
gehirnet evaluate --table data/metadata/test_set.csv --data-root data/features/original --baseline outputs/runs/baseline/seed42/baseline/model.pth --output outputs/metrics/baseline_seed42.json
```

Repeat for each seed/experiment. Never use test scores to choose parameters or arbitrarily pair stage checkpoints. A release bundle can also be evaluated with `--model-dir`.

## 7. Aggregate and compare Table II

```bash
python scripts/summarize_metrics.py outputs/metrics/hierarchical_seed40.json outputs/metrics/hierarchical_seed41.json outputs/metrics/hierarchical_seed42.json --ddof 1 --output outputs/metrics/hierarchical_summary.json
```

The script checks that all runs have the same task, unit, sample count and test path/truth cohort. Inputs must be independent seed runs, not copies of one evaluation. It defaults to sample standard deviation (`ddof=1`); use `--ddof 0` for population standard deviation if that is the convention confirmed for the original paper aggregation. That convention has not been established here, so it must be documented when checking the published standard deviations.

Compare seven-class accuracy, weighted F1 and MCC with [Table II in the model card](../MODEL_CARD.md). Checkpoint/seed provenance, manual exclusions, original features and historical selection/RNG details must be aligned before asserting exact agreement. Current [PD binary metrics](VALIDATION.md) are a separate task.

## 8. Reproduce the remaining figures/analyses

`experiments/ablations/` contains the sex-input and extra-hidden-layer studies; `experiments/analysis/` contains Mel/UMAP/statistical/fairness analyses; `experiments/figures/` retains plots with manually entered results. Install optional analysis dependencies and configure historical paths before running them from the repository root.

The source implementation for CKA, participant aggregation, mean/median bootstrap interpretation and the fairness MacroTPR test needs reconciliation as listed in [reproducibility notes](REPRODUCIBILITY.md). Those analyses are archived, not certified as fully reproduced by the new package.
