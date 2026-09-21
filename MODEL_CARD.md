# GeHirNet model card

## Model and intended use

GeHirNet is a research model for classifying sustained vowel /a/ recordings into healthy control or six pathologies. The paper is **GeHirNet: A Sex-Aware Hierarchical Model for Fair and Accurate Voice Pathology Classification**, by Fan Wu, Kaicheng Zhao, Elgar Fleisch and Filipe Barata.

Three separately trained single-channel ResNet-50 classifiers implement a hierarchy: PD predicts male/female × healthy/pathology, then MP and FP classify pathological samples. Input is a log-Mel segment of shape `(1, 128, 98)`. The model's sex route is inferred from the voice, not supplied patient metadata.

This release includes a complete hierarchical artifact at `checkpoints/hierarchical/`: PD plus matched MP and FP state dicts. The artifact manifest records model roles, class order, preprocessing, SHA-256 checksums, and Apache-2.0 weight licensing. Its original training seed is not encoded in the supplied filenames and remains unknown.

## Data

The paper combines sustained vowel recordings from Coswara (COVID-19), ALS, PC-GITA (Parkinson's) and SVD (laryngeal pathologies), including healthy recordings from all four sources. Labels retain the historical dataset spelling in the API. `Rekurrensparese` denotes vocal cord paresis; `Dysphonie` denotes dysphonia.

Manual outlier removal precedes normalization and segmentation in the paper. The supplied CSVs contain 13,509 total segments and an 80/20 split. Data and metadata are not included in the Python distribution.

## Reported evaluation

Paper Table II reports mean ± standard deviation across seeds 40, 41 and 42, on the original segment test split:

| Model | Accuracy | Weighted F1 | MCC |
| --- | --- | --- | --- |
| Baseline | 0.9598 ± 0.0063 | 0.9590 ± 0.0068 | 0.9190 ± 0.0131 |
| GeHirNet | 0.9647 ± 0.0054 | 0.9644 ± 0.0058 | 0.9294 ± 0.0114 |
| + Resampling | 0.9646 ± 0.0023 | 0.9641 ± 0.0024 | 0.9289 ± 0.0047 |
| + Time warping | 0.9724 ± 0.0037 | 0.9723 ± 0.0038 | 0.9449 ± 0.0074 |

These are reported paper results. The supplied final checkpoint set was also evaluated locally on all 2,702 supplied test segments:

| Checkpoint set | Accuracy | Weighted F1 | MCC |
| --- | ---: | ---: | ---: |
| `checkpoints/hierarchical/` | 0.9763 | 0.9761 | 0.9525 |

This is one final checkpoint set, not the Table II three-seed mean. The local CSVs share 1,454 `(Dataset, ID)` keys between train and test, so speaker-independent performance has not been established. See the [reproduction guide](docs/REPRODUCE.md).

## Fairness and limitations

The paper reports smaller MacroTPR sex gaps, but EOdds differences are not statistically significant. Binary biological-sex targets reflect the source datasets; performance beyond these categories has not been evaluated.

The model was evaluated on sustained vowels, with substantial class imbalance and small rare-disease cohorts. Recording conditions, language and disease are partly associated with dataset source. Results do not establish generalization to unrestricted speech, unseen speakers, new devices or clinical populations. Predictions are research outputs, not validated clinical diagnoses.

## Licensing and availability

Code and the supplied hierarchical model artifact use Apache-2.0 and retain the project attribution notices. Source audio, metadata, and Mel-feature access or redistribution terms remain separate from the code/model license. No external model hosting location or DOI is claimed by this repository.
