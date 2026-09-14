# Paper implementation and reproducibility

This document records the initial extraction of reusable code from the original research scripts. Historical scripts and notebooks are archived under `experiments/` so their outputs and assumptions can be inspected.

## Paper mapping

| Paper section | Historical implementation | Reusable implementation |
| --- | --- | --- |
| II-B audio preprocessing | `experiments/preprocessing/` | `preprocessing.py` |
| II-C augmentation | `experiments/augmentation/` | `augmentation.py` primitives |
| II-D Mel spectrograms | `experiments/preprocessing/Melsave_Resampling.py` | `preprocessing.py` |
| II-E/F ResNet and hierarchy | training/evaluation notebooks | `models.py`, `inference.py` |
| II-G/H experiments and training | epoch 10/20/30 notebooks | `train.py`, `configs/` |
| II-I ablations | `experiments/ablations/` | historical notebooks |
| II-J acoustic analysis | `experiments/analysis/statistic_analysis.py`, `experiments/analysis/visulization.py` | historical scripts |
| II-J fairness | `experiments/analysis/fairness.py`, `experiments/analysis/Fairness.csv` | historical script |

## Preserved semantics

- Checkpoints use `base_model.*` state-dict keys and a one-channel ResNet-50.
- PD: `MC, MP, FC, FP`.
- MP/FP: `ALS, Covid-19, Dysphonie, Laryngitis, Parkinson, Rekurrensparese`.
- Hierarchical output: those six pathologies followed by `HC`.
- Baseline output: `ALS, Covid-19, Dysphonie, HC, Laryngitis, Parkinson, Rekurrensparese`.
- Routing uses PD argmax, then MP or FP argmax; healthy routes produce HC directly. The second classifier receives the original Mel input, not PD hidden features.
- RMS VAD uses 2048 samples, hop 512, threshold 0.001; voiced regions receive fade-in/out before a 512-sample crossfade, matching the historical script.
- Normalization maps amplitudes to [0, 1]; segments are one second with a 0.4-second step and wrap padding at the end.
- Mel tensors have 128 bins and 98 time frames. At 44100 Hz, FFT/hop are 1808/452; supported other rates scale 1848/462 relative to 45000 Hz.
- Training uses Adam, cross-entropy and final-epoch fold MCC. The extracted trainer selects epochs, batch size and learning rate by mean validation MCC, then retrains on the full supplied training CSV. It does not use test metrics for selection.

## Deliberate changes

- Historical artifacts were relocated to `experiments/` and paper figures/manuscript to `docs/`. The archived fairness script resolves its aggregate CSV beside itself. Core historical script contents and notebook cells were preserved. Superseded demo/readme/statistical-analysis variants, empty documentation and unrelated grant figures were subsequently removed.
- WAV processing operates in memory; historical scripts wrote intermediate audio files. Quantization/file round trips and manual selection mean new raw-WAV features are not claimed to be bit-identical to archived features.

- Inference constructs an uninitialized ResNet before loading the checkpoint; no ImageNet download is needed. Training defaults to ImageNet initialization.
- Invalid or missing Mel files raise errors rather than silently disappearing from evaluation.
- Stereo audio is rejected explicitly. Short recordings produce at least one padded segment; short VAD regions use the available crossfade overlap. Empty or constant normalized audio is rejected.
- Original preprocessing also involved manual spectrogram-based outlier removal. WAV inference cannot reproduce those manual decisions automatically.
- Training records its fold paths, class order, config and CV results. RNGs are reseeded per fold/configuration for controlled comparisons; this is not guaranteed to reproduce the historical random-number sequence or published numbers exactly.
- Augmentation functions are available, but the trainer consumes prepared Mel CSVs. Balancing datasets, generating augmented files and maintaining augmentation provenance remain offline steps. The original augmentation notebooks/scripts document the paper workflow.

## Dataset and split audit

Local CSVs contain 10,807 training and 2,702 testing segments, totaling 13,509, consistent with the paper's segmentation counts. All 2,702 test paths resolve relative to `data/features/original`; none of the 10,807 training paths resolve at that feature root.

An audit of `(Dataset, ID)` keys finds 2,511 training keys, 1,530 test keys and **1,454 shared keys**. This demonstrates recording/identity-key overlap in the supplied split. Dataset-specific ID semantics must be checked before interpreting these counts as unique people. The original cross-validation also splits segments using `StratifiedKFold`, without identity grouping.

The published results should therefore be labeled as results on the original segment split, not established performance on unseen speakers. A speaker-disjoint evaluation requires source-specific participant IDs, a split before segmentation/augmentation, and new results reported separately. No historical split has been modified during this extraction.

## Analysis discrepancies to resolve

- The paper describes participant-level Mel statistics and UMAP, while the current scripts load individual `.npy` segments without participant aggregation.
- The paper describes a median-difference bootstrap CI in the nonparametric branch; `bootstrap_ci` calculates mean differences.
- The t-test branch uses the equal-variance default, while its CI uses a Welch-style standard error and degrees of freedom.
- `experiments/analysis/fairness.py` computes average male/female TPR (`macroTPR`) for its second Friedman test, rather than the absolute sex gap described as MacroTPR disparity in the paper. EOdds uses absolute TPR/FPR differences as expected.
- The CKA figure is in the paper, but a reusable CKA computation has not been established in this extraction. Notebook outputs are not a substitute for its source implementation.

## Release artifacts still needed

The local `checkpoints/pd/final_group_classifier.pth` contains PD only. Matched MP/FP weights, experiment/seed provenance, and release locations must be supplied for a complete seven-class pretrained release. Do not combine arbitrarily chosen stage checkpoints and describe the result as the paper's best model.

Raw audio, Mel features and metadata are separate data artifacts. Their source permissions and identity fields need review before redistribution; the repository's existing code license does not establish their release status.

Local metadata/features and the original PD checkpoint were moved to `data/metadata/`, `data/features/` and `checkpoints/pd/` during final cleanup. Configurations, documentation and the two archived analysis scripts were updated to those locations; canonical CSV `Full_Path` values and all data/weight bytes were preserved.

Apache-2.0 project contributor/license headers were added to all Python files and the first nonempty code cell of each experiment notebook. Notebook outputs/metadata and executable Python syntax were preserved by this attribution-only change.
