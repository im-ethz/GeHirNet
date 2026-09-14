# Prepare the public source and model release

The current workspace is an open-source repository structure with tested implementation, local PD weights and documented missing artifacts. It has not been pushed, published or given a model download URL.

## Source release

Include `src/`, `configs/`, `scripts/`, `tests/`, model usage documentation, data schemas, model/reproduction guides, citation metadata, existing license/attribution notices, and the paper experiments you intend to publish. `.github/workflows/tests.yml` runs the core tests on a clean checkout.

The existing `.gitignore` keeps local training/test metadata, raw audio, Mel features, weights, virtual environments and generated outputs out of ordinary source commits. It does not remove files from disk or remove any files already tracked in Git. The initial workspace only tracked README; the new source/artifacts remain local changes until committed.

Before publishing, inspect `git status` and the concrete files being included, run the documented tests, and verify the source package builds. Decide separately whether the manuscript and supplied figures are intended public artifacts. Add the actual repository URL, publication venue/year/DOI and model download location when known; none is invented by this packaging.

## Model release

For **PD-only**, only the existing checkpoint is necessary for local use:

```bash
gehirnet bundle --pd checkpoints/pd/final_group_classifier.pth --output-dir outputs/releases/pd-only --experiment unknown
```

For a **complete seven-class pretrained model**, supply matched PD/MP/FP from a documented experiment/seed:

```bash
gehirnet bundle --pd path/to/pd.pth --mp path/to/mp.pth --fp path/to/fp.pth --output-dir outputs/releases/gehirnet-timewarp-seed42 --experiment timewarp --seed 42 --weight-license YOUR_CONFIRMED_WEIGHT_LICENSE
```

The bundle validates checkpoint structure, preserves label order, copies weights and writes `manifest.json` plus usage instructions. It checks copied hashes and refuses existing output directories. It cannot infer experiment provenance, ownership, clinical validity or agreement with the paper. `unknown` and `UNSPECIFIED` remain unresolved metadata; resolve those before describing the model as the paper's validated released checkpoint.

Complete the artifact with the actual weight license text, reviewed model card and checkpoint-specific evaluation summary. Archive that directory, publish it in your chosen downloadable artifact location, and link the real URL from README/checkpoints documentation. Users extract it and pass `--model-dir`; no training data is needed for inference.

Verify the download/extraction in a clean source checkout, run WAV/Mel inference, and compare test metrics against the original local artifact. Do not publish the virtual environment or test-set prediction files as part of the model bundle.

## Missing-artifact inventory

| Artifact/decision | Current status | Needed for |
| --- | --- | --- |
| PD weight | Present, strict loading and binary test evaluation pass | PD-only inference |
| MP + FP weights | Not found in this repository | Full pretrained seven-class inference |
| Checkpoint experiment/seed/selected parameters | Existing PD provenance not established | Claiming a specific paper model release |
| Baseline weights and all matched stage weights across seeds/experiments | Not supplied | Checkpoint-only reproduction of all Table II runs |
| Training Mel features at CSV paths | 0 of 10,807 paths resolve locally | Retraining from the original feature split |
| Shareable split/provenance metadata or a preparation recipe | Local tables exist; public artifact/access procedure not established | Reconstructing the evaluated cohort |
| Exact recording inclusion/exclusion list and source participant IDs | Not established in public artifacts | Dataset reconstruction and speaker-disjoint study |
| Stage-specific augmented tables/parent mapping | Historical scripts exist; ready-to-run prepared artifacts not supplied | Exp 3.1 / 3.2 reproduction |
| Analysis implementation reconciliations, CKA source | Unresolved details in reproducibility notes | Full analysis/figure reproduction |
| Code license | Existing Apache-2.0 retained | Public source distribution |
| Weight/data/feature release terms | Must be documented independently | Public artifact distribution |
| Real repository/model URL and final paper citation metadata | Not supplied | Public discovery and citation |

## What is usable now

The source implementation, raw-recording feature preparation, config generation, single-classifier training, task-specific evaluation, seed aggregation and local bundle interface are available. PD-only inference is verified using the existing local weight. Full hierarchy routing is tested with controlled outputs; actual MP/FP pretrained evaluation awaits their weights. Speaker-grouped CV and automated full augmentation balancing are not implemented.

Code can be published with an honest implementation/PD-only release status. A complete pretrained GeHirNet release requires the additional weights and artifact metadata above.
