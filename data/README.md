# Local data layout and schemas

Data artifacts are separate from the source release. The example CSVs in `templates/` contain illustrative records, not real subjects or training data.

```text
data/
  templates/               # public schema examples
  raw/                     # locally obtained mono vowel WAV files
  metadata/                # local curated recording lists and split tables
  features/                # generated .npy files and segment tables
```

`raw/`, `metadata/` and `features/` are ignored by Git. The original train/test CSV and XLSX tables are stored locally in `data/metadata/`. Original group-organized test features are in `data/features/original/`; analysis features are in `data/features/Mel_NPY_Output/`. These directories are ignored by Git.

## Recording table for preparation

Required columns: `Audio_Path, Dataset, ID, Sex, Pathology, Group`.

- `Audio_Path`: absolute path or a path relative to `--audio-root`.
- `Dataset` and `ID`: preserve source recording identity. Add `Participant_ID` if the source has a separate participant key; do not assume recording IDs are person IDs.
- `Sex`: `M` or `F`, reflecting the paper's source labels.
- `Pathology`: `HC`, `ALS`, `Covid-19`, `Dysphonie`, `Laryngitis`, `Parkinson`, or `Rekurrensparese`.
- `Group`: `MC`, `MP`, `FC`, or `FP`, consistent with sex and health status.

Prepare only recordings already selected by the paper's manual inspection or your documented inclusion rules. Do not normalize each segment separately: normalization precedes segmentation. `--already-preprocessed` means both VAD and normalization have already been applied to the input recordings.

## Segment table for training/evaluation

Minimum columns: `Full_Path, Pathology, Group`. Identity/provenance columns should also be retained. `prepare` writes `Audio_Path, Dataset, ID, Sex`, any extra input metadata, and `NPY, segment_index, segment_duration_seconds, augmentation_method`.

`Full_Path` resolves relative to the configured `data_root`. Each file is a finite log-Mel array of shape `(1, 128, 98)` or `(128, 98)`; stored output uses the former. Generated filenames are new and do not reproduce the historical filename convention.

For augmented training tables, preserve parent recording/segment IDs, original participant identity, augmentation method, target sample rate or permutation, and random seed. Stage-specific tables must resolve from a common `data_root` when using the config generator.

## Splits

Use the original train/test segment lists for comparisons with the paper's original protocol. New recording preparation does not recover its manual exclusions or exact split assignment.

For new speaker-disjoint studies, establish source-specific participant identities and split them before segmentation and augmentation. Prepare training and test recording lists separately. The extracted trainer currently retains segment-stratified CV for the historical protocol; speaker-grouped CV is not yet implemented.
