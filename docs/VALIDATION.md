# Local validation of the extracted package

Validation used Python 3.12 with PyTorch 2.6.0, torchvision 0.21.0 and torchaudio 2.6.0 on CPU. This checks the package implementation; it is not a rerun of all paper experiments.

- Thirteen unittest checks pass, covering hard routing, healthy-route bypass, baseline label/evaluation order, checkpoint key shapes, finite Mel validation, unsupported shapes, missing sample detection, short-audio behavior, and WAV-to-Mel conversion at 44100/48000/50000 Hz.
- Workflow checks additionally verify bundle corruption/label-order rejection, complete stage pairing, preparation provenance/overwrite protection, nine independent hierarchy configs and metric-task/cohort consistency.
- A local PD model bundle was built at `outputs/releases/pd-only-v0.1` and successfully used through the public `predict --model-dir` CLI. Its experiment remains `unknown` and weight license `UNSPECIFIED`; it has not been publicly released.
- The local PD checkpoint loads strictly and predicts an existing test sample. Its binary remapping matches the original demo's direct argmax/remapping, also covered by routing tests. The superseded standalone demo was removed during cleanup.
- A synthetic eight-sample PD dataset completes two-fold CV, one-epoch full-data retraining, metadata export and strict checkpoint reload. Reusing its output directory is rejected. ImageNet initialization is disabled for this smoke check; synthetic metrics have no disease-performance meaning.
- Editable installation, CLI help, dependency consistency and wheel construction pass. Wheel contents exclude audio, Mel arrays, metadata tables and model checkpoints.
- The metadata audit reports 10,807/2,702 training/test segments, 0/2,702 existing feature-relative paths and 1,454 overlapping dataset/ID keys.

Run the core checks with:

```bash
python -m unittest discover -s tests -v
gehirnet --help
gehirnet audit --train-table data/metadata/train_set.csv --test-table data/metadata/test_set.csv --data-root data/features/original
```

The configured GitHub Actions workflow has not been run remotely. Complete MP/FP pretrained inference and the full paper training grid require the missing release artifacts and training data.

## Full local PD evaluation

The local PD checkpoint was evaluated on all 2,702 supplied test segments. Binary HC/pathology accuracy is **0.979275**, weighted F1 **0.979172**, and MCC **0.951862**. The confusion matrix, with rows/columns ordered HC then Pathology, is `[[1837, 13], [43, 809]]`.

These are first-stage binary detection metrics, not the paper's seven-class Table II metrics. The [summary](validation/pd_summary.json) includes the checkpoint checksum. The temporary per-sample predictions were removed during cleanup. Re-running the documented evaluation command regenerates them; the aggregate summary is retained for reference.
