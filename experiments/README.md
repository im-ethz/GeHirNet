# Historical paper experiments

Original experiment artifacts are archived without rewriting notebook contents:

- `experiments/training/`: epoch 10, 20 and 30 notebooks.
- `experiments/evaluation/`: combinations of the two-stage classifiers.
- `experiments/ablations/`: additional layers, sex input and random-seed experiments.
- `experiments/preprocessing/`, `experiments/augmentation/`: historical preprocessing and balancing scripts.
- `experiments/figures/plot_*.py`: figures from manually entered experimental values.
- `experiments/analysis/statistic_analysis.py`, `experiments/analysis/visulization.py`, `experiments/analysis/fairness.py`: acoustic and fairness analyses.

Paths inside historical artifacts may be Windows or Colab paths. Use the `gehirnet` package for portable inference/training and read the [reproduction guide](../docs/REPRODUCE.md) before interpreting historical results. Run historical Python scripts from the repository root so their relative data paths continue to resolve. Notebook paths still require the original environment or manual configuration.

Superseded standalone demo, legacy readme, empty ablation readme and the older statistical-analysis variant were removed during release cleanup. The portable inference CLI and current statistical-analysis script provide those workflows.
