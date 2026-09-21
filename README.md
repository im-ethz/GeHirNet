# GeHirNet

Official research implementation of **GeHirNet: A Sex-Aware Hierarchical Model for Fair and Accurate Voice Pathology Classification**, by Fan Wu, Kaicheng Zhao, Elgar Fleisch, and Filipe Barata.

GeHirNet classifies sustained vowel `/a/` recordings with three independently trained ResNet-50 models:

1. `PD` predicts male/female × healthy/pathology: `MC, MP, FC, FP`.
2. Healthy predictions become `HC`.
3. `MP` or `FP` classifies pathological samples into six diseases.

The complete hierarchical checkpoint is available in `checkpoints/hierarchical/`. It contains three weight files and a validated manifest.

## Install

Python 3.12 is the locally validated version.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Use the model

Recommended:

```bash
gehirnet predict path/to/vowel.wav \
  --model-dir checkpoints/hierarchical \
  --output outputs/prediction.json
```

Using the three `.pth` files directly:

```bash
gehirnet predict path/to/vowel.wav \
  --pd checkpoints/hierarchical/pd.pth \
  --mp checkpoints/hierarchical/mp.pth \
  --fp checkpoints/hierarchical/fp.pth
```

See [docs/USAGE.md](docs/USAGE.md) for WAV/Mel requirements, Python usage, outputs, and evaluation.

## Reproduce the research workflow

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for data preparation, training, evaluation, seeds, augmentation experiments, and known reproduction limits.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/gehirnet/` | Reusable preprocessing, training, evaluation, inference, and model-bundle code |
| `checkpoints/hierarchical/` | Complete PD/MP/FP model artifact |
| `configs/`, `scripts/` | Training configurations and experiment helpers |
| `data/templates/` | Metadata schema examples |
| `experiments/` | Original notebooks, preprocessing, augmentation, ablation, and analysis materials |
| `docs/USAGE.md` | How to call the pretrained model |
| `docs/REPRODUCE.md` | How to reproduce the experimental workflow |
| `MODEL_CARD.md` | Intended use, reported results, and limitations |

## Validation

```bash
python -m unittest discover -s tests -v
```

The implementation uses the original class orders and hard routing. Model manifests verify preprocessing metadata, label order, and SHA-256 checksums before loading.

## License and citation

Code and the supplied model artifact use Apache-2.0. See [LICENSE.txt](LICENSE.txt), [NOTICES.txt](NOTICES.txt), and [CITATION.cff](CITATION.cff).
