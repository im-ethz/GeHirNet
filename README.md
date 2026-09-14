# GeHirNet

Research code for **GeHirNet: A Sex-Aware Hierarchical Model for Fair and Accurate Voice Pathology Classification**, by Fan Wu, Kaicheng Zhao, Elgar Fleisch and Filipe Barata.

[Pipeline](docs/PIPELINE.md) · [Use a model](docs/USAGE.md) · [Reproduce experiments](docs/REPRODUCE.md) · [Release artifacts and missing items](docs/RELEASE.md)

GeHirNet classifies sustained vowel `/a/` recordings using single-channel ResNet-50 models. A first classifier predicts male/female × healthy/pathology; pathological samples are routed to male- or female-specific six-class disease classifiers. Healthy routes are merged into HC. The classifiers are trained separately.

## Release status

The reusable Python package provides recording-to-Mel preparation, WAV/Mel inference, strict CSV evaluation, five-fold training with hyperparameter selection, and portable model bundles. Scripts generate seed configs and aggregate evaluation metrics. Historical notebooks and scripts remain available for the paper experiments.

**The only checkpoint currently present locally is `checkpoints/pd/final_group_classifier.pth` (PD).** It supports four-group prediction and healthy/pathology detection. Complete seven-class pretrained inference requires matched MP and FP checkpoints, whose release artifacts and provenance have not yet been supplied. Model weights and datasets are excluded from the Python package and from Git by default.

See the [model card](MODEL_CARD.md) and [reproducibility notes](docs/REPRODUCIBILITY.md). The original CSV split contains overlapping dataset/ID keys between training and testing; published metrics describe the original segment split and do not establish unseen-speaker performance.

## Installation

Python 3.10 or newer is required. Use matching PyTorch, torchvision and torchaudio versions; the paper used PyTorch 2.6.0 and CUDA 12.4 on an NVIDIA L4.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional dependencies for historical analysis scripts:

```bash
python -m pip install -e '.[analysis]'
```

## Inference

For an extracted, separately downloaded model bundle:

```bash
gehirnet predict path/to/vowel.wav --model-dir path/to/downloaded-model --output outputs/prediction.json
```

The source clone contains no pretrained weights. A public weight URL has not yet been established; [model usage](docs/USAGE.md) explains the required artifact. Authors can assemble a local bundle with `gehirnet bundle`; see [release instructions](docs/RELEASE.md).

PD-only inference with a local test Mel:

```bash
gehirnet predict path/to/sample.npy --pd checkpoints/pd/final_group_classifier.pth
```

Complete hierarchy, after obtaining all three matched checkpoints:

```bash
gehirnet predict path/to/vowel.wav --pd path/to/pd.pth --mp path/to/mp.pth --fp path/to/fp.pth --output outputs/prediction.json
```

WAV inputs must be mono sustained-vowel recordings at 44100 Hz, or a rate from 40000 to 50000 Hz in 125 Hz steps. WAV processing applies RMS silence removal, crossfade, min-max normalization and one-second segmentation with a 0.4-second step. Use `--already-preprocessed` if VAD and normalization have already been applied. Manual outlier removal from the paper is not automated.

Mel inputs must contain finite log-Mel values with shape `(1, 128, 98)` or `(128, 98)`. Output JSON contains a prediction per segment; no participant-level aggregation is imposed. Use `--device cuda` for GPU inference; CPU is the default.

Python API:

```python
from gehirnet.inference import GeHirNet
from gehirnet.preprocessing import load_input

model = GeHirNet("checkpoints/pd/final_group_classifier.pth")
predictions = model.predict(load_input("path/to/sample.npy"))
```

## Evaluation

CSV files require `Full_Path`, `Pathology`, and `Group`. Paths may use either slash style and are resolved relative to `--data-root`. Missing samples cause an error rather than altering the evaluated cohort.

```bash
gehirnet evaluate --table data/metadata/test_set.csv --data-root data/features/original --pd checkpoints/pd/final_group_classifier.pth --output outputs/pd_metrics.json
```

Add `--mp` and `--fp` for seven-class evaluation. JSON records accuracy, weighted F1, MCC, the confusion matrix, and each sample's prediction. PD-only metrics are binary HC/pathology metrics.

Metadata audit (does not load model weights):

```bash
gehirnet audit --train-table data/metadata/train_set.csv --test-table data/metadata/test_set.csv --data-root data/features/original
```

Single-stage baseline checkpoints use a different class order, selected explicitly:

```bash
gehirnet predict path/to/sample.npy --baseline path/to/baseline.pth
gehirnet evaluate --table data/metadata/test_set.csv --data-root data/features/original --baseline path/to/baseline.pth --output outputs/baseline_metrics.json
```

## Training and paper experiments

The complete workflow, including seeds 40/41/42 and all experiment variants, is in [the reproduction guide](docs/REPRODUCE.md). Raw recording/feature schemas are in [data documentation](data/README.md).

For a curated recording CSV, the preparation entry point is:

```bash
gehirnet prepare --table data/metadata/train_recordings.csv --audio-root data/raw --output-root data/features/train
```

It writes features plus `segments.csv` and preserves recording metadata. Manual exclusions and original split reconstruction are not automated.

Edit a JSON file in `configs/` to set the training CSV, data root, device and unique output directory:

Relative configuration paths are resolved from the command's working directory. Run the supplied examples from the repository root, or use absolute paths.

```bash
gehirnet train --config configs/baseline.json
gehirnet train --config configs/pd.json
gehirnet train --config configs/mp.json
gehirnet train --config configs/fp.json
```

Defaults search epochs `{10,20,30}`, batch sizes `{32,64}` and learning rates `{1e-3,1e-4,1e-5}` using five-fold stratified segment CV. The trainer selects mean final-epoch validation MCC, retrains on the full training table and writes `model.pth` plus `training.json`. Set seeds 40, 41 and 42 in separate configs/output directories for repeated runs. Training with pretrained initialization may download ImageNet weights.

The trainer consumes prepared Mel datasets. For resampling/time-warping experiments, prepare stage-specific augmented training tables offline, retaining original sample and participant provenance. Apply augmentation to training data only. Audio primitives are in `augmentation.py`; the historical balancing workflows remain in `experiments/augmentation/`. This extraction does not claim fully automated augmentation-experiment reproduction.

Local training CSV paths currently need their corresponding training data. New speaker-disjoint experiments should split participants before segmentation/augmentation and be reported separately from the original paper protocol.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/gehirnet/` | Models, labels, preprocessing, augmentation primitives, inference, training and evaluation |
| `checkpoints/manifest.json` | Local weight checksum, class order and missing artifact status |
| `configs/` | Baseline and PD/MP/FP training configurations |
| `scripts/` | Three-seed config generation and guarded metric aggregation |
| `data/templates/` | Public recording and segment CSV schema examples |
| `tests/` | Input, routing and checkpoint compatibility checks |
| `docs/` | Paper mapping, data audit and implementation differences |
| `experiments/training/`, `experiments/evaluation/`, `experiments/ablations/` | Original paper notebooks |
| `experiments/preprocessing/`, `experiments/augmentation/` | Original preprocessing and augmentation scripts |
| `experiments/analysis/`, `experiments/figures/` | Historical statistics and paper figures |

## Validation

```bash
python -m unittest discover -s tests -v
```

Before installation, the equivalent is `PYTHONPATH=src python -m unittest discover -s tests -v`.

Local checks and PD evaluation are recorded in [validation notes](docs/VALIDATION.md).

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). The local manuscript is `docs/paper/ICHI_Final.pdf`; publication venue/year/DOI have not been inferred from its filename.

Code retains the existing [Apache-2.0 license](LICENSE.txt) and [attribution notices](NOTICES.txt). Dataset, feature and checkpoint redistribution terms must be documented separately before a public artifact release.
