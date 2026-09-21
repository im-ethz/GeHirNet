# Contributing

Install the package with Python 3.12, then run:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Changes to preprocessing, label order or routing must describe their effect on checkpoint compatibility and historical metrics. Preserve historical notebooks as reference artifacts and put reusable functionality in `src/gehirnet/`. New results should include the task, split unit, seed, configuration and checkpoint provenance.

Do not include local data or participant metadata in ordinary source changes. Model artifacts belong under `checkpoints/` with a manifest, checksums, class order, and explicit weight license as described in `docs/USAGE.md`. Keep paper-reported values separate from newly measured checkpoint results.

All Python scripts, including tests, must retain the project contributor header and Apache-2.0 license notice. Experiment notebooks carry the same notice at the start of their first nonempty code cell. Contributor names and existing ownership attribution come from `NOTICES.txt`; update that attribution when project contributors change.
