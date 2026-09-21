# Using the pretrained GeHirNet model

The released hierarchical model is a directory, not one combined state dict:

```text
checkpoints/hierarchical/
  manifest.json
  README.md
  LICENSE.txt
  pd.pth   # 4 outputs: MC, MP, FC, FP
  mp.pth   # 6 male-pathology outputs
  fp.pth   # 6 female-pathology outputs
```

`pd.pth`, `mp.pth`, and `fp.pth` are all required for seven-class inference. The manifest records their label order, preprocessing contract, SHA-256 checksums, and Apache-2.0 weight license.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `--device cuda` when a compatible CUDA build is installed. CPU is the default.

## Recommended CLI call

For a raw recording:

```bash
gehirnet predict path/to/vowel.wav \
  --model-dir checkpoints/hierarchical \
  --output outputs/prediction.json
```

For a prepared Mel feature:

```bash
gehirnet predict path/to/segment.npy \
  --model-dir checkpoints/hierarchical
```

The model directory is preferred because the loader checks the manifest and all three file checksums before strict state-dict loading.

## Calling the `.pth` files directly

```bash
gehirnet predict path/to/vowel.wav \
  --pd checkpoints/hierarchical/pd.pth \
  --mp checkpoints/hierarchical/mp.pth \
  --fp checkpoints/hierarchical/fp.pth \
  --output outputs/prediction.json
```

Do not pass only `mp.pth` or `fp.pth`: the first-stage PD prediction selects which second-stage model runs. Do not exchange these roles because their output dimensions or learned populations differ.

## Python API

Recommended manifest-based loading:

```python
from gehirnet.bundle import load_bundle
from gehirnet.preprocessing import load_input

model, metadata = load_bundle("checkpoints/hierarchical", device="cpu")
mels = load_input("path/to/vowel.wav")
predictions = model.predict(mels)
print(predictions)
```

Direct checkpoint loading:

```python
from gehirnet.inference import GeHirNet
from gehirnet.preprocessing import load_input

model = GeHirNet(
    pd_checkpoint="checkpoints/hierarchical/pd.pth",
    mp_checkpoint="checkpoints/hierarchical/mp.pth",
    fp_checkpoint="checkpoints/hierarchical/fp.pth",
    device="cpu",
)
predictions = model.predict(load_input("path/to/segment.npy"))
```

## Inputs

Raw audio must be a mono sustained vowel `/a/` WAV. Supported sample rates are 44100 Hz or values from 40000 through 50000 Hz in 125 Hz steps. The default pipeline applies RMS silence removal, fade/crossfade, min-max normalization, one-second segmentation with a 0.4-second step, wrap padding, and log-Mel conversion.

Use `--already-preprocessed` only when VAD and min-max normalization were already applied to the entire recording. Segmentation and Mel extraction still run.

A `.npy` input must contain finite log-Mel values with shape `(1, 128, 98)` or `(128, 98)`.

## Outputs

The final class order is:

```text
ALS, Covid-19, Dysphonie, Laryngitis, Parkinson, Rekurrensparese, HC
```

Example schema:

```json
{
  "input": "path/to/vowel.wav",
  "mode": "hierarchical",
  "segments": [
    {"group": "MC", "label": "HC"},
    {"group": "FP", "label": "Dysphonie"}
  ]
}
```

Predictions are returned per one-second segment. The repository does not impose a whole-recording voting rule or return calibrated clinical probabilities.

## Evaluate a labeled feature set

The CSV requires at least `Full_Path`, `Pathology`, and `Group`. Paths resolve relative to `--data-root`.

```bash
gehirnet evaluate \
  --table data/metadata/test_set.csv \
  --data-root data/features/original \
  --model-dir checkpoints/hierarchical \
  --output outputs/hierarchical_metrics.json
```

The output includes segment-level accuracy, weighted F1, MCC, confusion matrix, and predictions. The model is intended for research use on sustained vowels; see `MODEL_CARD.md` for limitations.
