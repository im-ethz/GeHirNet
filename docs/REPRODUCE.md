# Reproducing the GeHirNet workflow

This guide separates checkpoint verification from full retraining. The supplied test features and complete hierarchical weights support evaluation now. Full training requires the missing original training features or regenerated features from the source recordings.

## 1. Install and verify the environment

Python 3.12 is locally validated. The paper used PyTorch 2.6.0, CUDA 12.4, and an NVIDIA L4.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
python -m unittest discover -s tests -v
```

## 2. Understand the model tasks

| Task | Classes | Model file |
| --- | --- | --- |
| PD | `MC, MP, FC, FP` | `pd.pth` |
| MP | six diseases, male pathology | `mp.pth` |
| FP | six diseases, female pathology | `fp.pth` |
| Baseline | HC plus six diseases | separately trained seven-class checkpoint |

MP/FP class order is `ALS, Covid-19, Dysphonie, Laryngitis, Parkinson, Rekurrensparese`. Hierarchical inference maps healthy PD routes directly to HC and feeds the original Mel tensor to MP or FP for pathological routes.

## 3. Audit the supplied split

```bash
gehirnet audit \
  --train-table data/metadata/train_set.csv \
  --test-table data/metadata/test_set.csv \
  --data-root data/features/original
```

The local metadata contains 10,807 training and 2,702 test segments. All test feature paths resolve; the 10,807 training feature paths are currently missing. The tables share 1,454 `(Dataset, ID)` keys. Source-specific ID semantics must be resolved before calling the evaluation speaker-independent.

For direct comparison with the paper, retain the supplied historical split. For a new participant-disjoint study, define source-specific participant IDs and split before segmentation or augmentation. The extracted trainer reproduces the historical segment-stratified CV design, not participant-grouped CV.

## 4. Evaluate the supplied hierarchical weights

```bash
gehirnet evaluate \
  --table data/metadata/test_set.csv \
  --data-root data/features/original \
  --model-dir checkpoints/hierarchical \
  --output outputs/hierarchical_metrics.json
```

This is the first check for the supplied model. Compare the resulting seven-class accuracy, weighted F1, MCC, and confusion matrix with the checkpoint-specific record. Paper Table II reports the mean and standard deviation across seeds 40, 41, and 42; evaluating one final checkpoint set is not the three-seed aggregate.

## 5. Recreate features when original training features are unavailable

Construct a curated recording CSV using these required columns:

```text
Audio_Path, Dataset, ID, Sex, Pathology, Group
```

Allowed pathology labels are `HC, ALS, Covid-19, Dysphonie, Laryngitis, Parkinson, Rekurrensparese`; sex is `M/F`; group is `MC/MP/FC/FP`. Preserve participant IDs and inclusion/exclusion provenance in additional columns.

```bash
gehirnet prepare \
  --table data/metadata/train_recordings.csv \
  --audio-root data/raw \
  --output-root data/features/train
```

This writes `(1,128,98)` Mel arrays and `segments.csv`. It implements VAD, crossfade, normalization, segmentation, and Mel extraction. It does not acquire datasets, reproduce the paper's manual spectral exclusions, or reconstruct the historical split automatically. Generated features therefore constitute a new documented reconstruction unless they are verified against the original feature hashes.

## 6. Generate the three-seed training plan

For the nonaugmented hierarchy:

```bash
python scripts/make_configs.py \
  --experiment hierarchical \
  --table data/features/train/segments.csv \
  --data-root data/features/train \
  --config-dir outputs/configs/hierarchical \
  --device cuda
```

This creates PD/MP/FP configs for seeds 40, 41, and 42 without launching training. For the baseline, use `--experiment baseline`. Each classifier searches epochs `{10,20,30}`, batch sizes `{32,64}`, and learning rates `{1e-3,1e-4,1e-5}` with five-fold stratified CV, then retrains using the selected settings on the full supplied training table.

## 7. Train

Example for seed 42:

```bash
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_pd.json
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_mp.json
gehirnet train --config outputs/configs/hierarchical/hierarchical_seed42_fp.json
```

Each run writes `model.pth` and `training.json`, including class order, selected parameters, fold MCC values, and fold file paths. Existing outputs are not overwritten.

## 8. Reproduce augmentation experiments

Only training data is augmented. Prepare separate PD, MP, and FP tables because the paper balances each stage independently. Resampling uses rates from 40000–50000 Hz in 125 Hz steps. Time warping permutes five audio blocks and applies a 32-sample crossfade.

The reusable primitives are in `src/gehirnet/augmentation.py`; historical balancing scripts are in `experiments/augmentation/`. Automated end-to-end class balancing and provenance-table generation are not yet implemented. Preserve the parent recording/segment, participant identity, augmentation method, target rate or permutation, and seed.

After preparing stage-specific augmented tables:

```bash
python scripts/make_configs.py \
  --experiment timewarp \
  --pd-table data/metadata/timewarp_pd.csv \
  --mp-table data/metadata/timewarp_mp.csv \
  --fp-table data/metadata/timewarp_fp.csv \
  --data-root data/features \
  --config-dir outputs/configs/timewarp \
  --device cuda
```

Use `--experiment resampling` for Exp 3.1.

## 9. Evaluate every seed and summarize

Evaluate matched PD/MP/FP checkpoints from the same experiment and seed on the unchanged test set. Then aggregate three independent metric files:

```bash
python scripts/summarize_metrics.py \
  outputs/metrics/seed40.json \
  outputs/metrics/seed41.json \
  outputs/metrics/seed42.json \
  --ddof 1 \
  --output outputs/metrics/summary.json
```

The script rejects mismatched tasks or test cohorts. Confirm whether the paper used sample (`ddof=1`) or population (`ddof=0`) standard deviation before claiming exact agreement.

## Reported and checkpoint results

Paper Table II reports mean ± standard deviation across seeds 40, 41, and 42 on the original segment test split:

| Model | Accuracy | Weighted F1 | MCC |
| --- | ---: | ---: | ---: |
| Baseline | 0.9598 ± 0.0063 | 0.9590 ± 0.0068 | 0.9190 ± 0.0131 |
| GeHirNet | 0.9647 ± 0.0054 | 0.9644 ± 0.0058 | 0.9294 ± 0.0114 |
| + Resampling | 0.9646 ± 0.0023 | 0.9641 ± 0.0024 | 0.9289 ± 0.0047 |
| + Time warping | 0.9724 ± 0.0037 | 0.9723 ± 0.0038 | 0.9449 ± 0.0074 |

The supplied final checkpoint set was evaluated locally on all 2,702 supplied test segments:

| Checkpoint set | Accuracy | Weighted F1 | MCC |
| --- | ---: | ---: | ---: |
| `checkpoints/hierarchical/` | 0.9763 | 0.9761 | 0.9525 |

This checkpoint result is from one final model set and is separate from the paper's three-seed aggregate.

## Remaining limits for exact paper reproduction

- The original 10,807 training Mel files are absent.
- Checkpoint seed and exact experiment provenance are not encoded in the supplied filenames; the final bundle records them as `hierarchical-final` with unknown seed.
- Original manual inclusion/exclusion decisions and source participant identifiers are incomplete as public artifacts.
- End-to-end augmentation balancing is not automated in the reusable package.
- CKA source and several analysis details require reconciliation: participant versus segment aggregation, mean versus median bootstrap difference, and MacroTPR disparity implementation.
- The extracted trainer preserves the method but uses a controlled RNG sequence that may differ from the historical notebooks, so bit-identical training is not guaranteed.

The original notebooks and analysis scripts remain under `experiments/` for inspection. Keep paper-reported metrics distinct from newly measured checkpoint results.
