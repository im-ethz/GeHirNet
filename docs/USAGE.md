# Use a pretrained model

Using a model does not require downloading the training datasets or running training. It requires the source package, a suitable input and a separately obtained model artifact.

## Install

Clone the actual public repository once it is published, enter its directory, then use Python 3.12 (the locally validated version):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
gehirnet --help
```

The package constrains PyTorch/torchvision/torchaudio to the 2.6/0.21/2.6 family. CPU is the default. GPU installations must provide a matching PyTorch build for their system.

## Obtain weights

The source repository intentionally excludes `.pth` files. The author must supply a real model download location and weight terms in the release documentation. No public download is available in this initial local preparation.

A downloadable artifact has one of these layouts:

```text
pd-only/                  complete-hierarchy/
  manifest.json             manifest.json
  README.md                 README.md
  pd.pth                    pd.pth
                            mp.pth
                            fp.pth
```

The artifact type is declared by the manifest. PD alone outputs `HC`/`Pathology` plus the predicted group; complete hierarchy outputs HC or one of six diseases. A missing MP/FP pair cannot provide the complete model.

## Run inference

After extracting the artifact:

```bash
gehirnet predict path/to/vowel.wav --model-dir path/to/extracted-artifact --output outputs/prediction.json
```

Input requirements:

- Mono sustained vowel `/a/`, not unrestricted speech.
- WAV sample rate 44100 Hz, or 40000–50000 Hz in 125 Hz steps, including 48000 and 50000 Hz.
- For an already VAD-filtered and normalized recording, add `--already-preprocessed` to avoid applying those operations again. Slicing into one-second segments still occurs.
- Alternatively, supply a finite log-Mel `.npy` of shape `(1,128,98)` or `(128,98)`. No audio preprocessing is applied to `.npy` inputs.

The loader verifies manifest format, label order, preprocessing and weight checksums before strict checkpoint loading. It does not validate author-supplied experiment provenance.

PD-only JSON structure:

```json
{
  "input": "path/to/vowel.wav",
  "mode": "pd",
  "segments": [{"group": "MC", "label": "HC"}]
}
```

This example explains the output schema; it is not a prediction guaranteed for arbitrary audio. Full hierarchy uses `mode: hierarchical` and disease labels. The model infers the routing group from audio; no sex metadata is supplied at inference. Multiple segments remain separate results. No calibrated disease probability or participant-level diagnosis is returned.

## Python API

```python
from gehirnet.bundle import load_bundle
from gehirnet.preprocessing import load_input

model, metadata = load_bundle("path/to/extracted-artifact", device="cpu")
predictions = model.predict(load_input("path/to/vowel.wav"))
print(predictions)
```

Individual checkpoint arguments remain supported for author/research use:

```bash
gehirnet predict path/to/sample.npy --pd checkpoints/pd/final_group_classifier.pth
gehirnet predict path/to/sample.npy --pd pd.pth --mp mp.pth --fp fp.pth
gehirnet predict path/to/sample.npy --baseline baseline.pth
```

Use `--device cuda` and optionally `--batch-size` for GPU inference. Predictions are research outputs; the model card describes the evaluated population and limitations.

## Evaluate your labeled cohort

Create a segment CSV following [the data schema](../data/README.md):

```bash
gehirnet evaluate --table path/to/test.csv --data-root path/to/features --model-dir path/to/artifact --output outputs/metrics.json
```

This reports metrics at the segment level. Accuracy/MCC from a PD artifact are binary metrics and cannot be compared directly with the paper's seven-class Table II metrics.
