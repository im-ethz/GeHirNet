# GeHirNet

GeHirNet turns a sustained `/a/` vowel recording into a segment-level prediction for healthy voice or one of six voice-related conditions: ALS, COVID-19, dysphonia, laryngitis, Parkinson's disease, and vocal cord paresis.

This repository is useful if you want to:

- run the pretrained model on a WAV recording or prepared Mel spectrogram;
- evaluate the released weights on a labeled dataset;
- retrain or adapt the model for voice pathology research;
- reproduce the paper's preprocessing, hierarchical training, and augmentation experiments.

Ready-to-use weights are included in `checkpoints/hierarchical/`, so inference does not require training first. The model uses a sex-aware hierarchy of three ResNet-50 classifiers to account for physiological voice differences while producing the final health or pathology label.

GeHirNet is intended for research on sustained vowels and is not a clinical diagnostic tool.

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

## Validation

```bash
python -m unittest discover -s tests -v
```

The implementation uses the original class orders and hard routing. Model manifests verify preprocessing metadata, label order, and SHA-256 checksums before loading.

## License

Code and the supplied model artifact use Apache-2.0. See [LICENSE.txt](LICENSE.txt) and [NOTICES.txt](NOTICES.txt).

## Citation

If you use this repository, please cite **GeHirNet: A Sex-Aware Hierarchical Model for Fair and Accurate Voice Pathology Classification**, by Fan Wu, Kaicheng Zhao, Elgar Fleisch, and Filipe Barata.
