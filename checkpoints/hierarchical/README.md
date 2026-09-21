# GeHirNet hierarchical model artifact

This directory contains the complete final hierarchical checkpoint set:

- `pd.pth`: `MC, MP, FC, FP`
- `mp.pth`: male pathology, six disease classes
- `fp.pth`: female pathology, six disease classes
- `manifest.json`: preprocessing, label order, SHA-256 checksums, and license metadata
- `LICENSE.txt`: Apache-2.0 license text

The three files form one model and must remain together. Recommended use:

```bash
gehirnet predict path/to/vowel.wav --model-dir checkpoints/hierarchical
```

Direct `.pth` use:

```bash
gehirnet predict path/to/vowel.wav \
  --pd checkpoints/hierarchical/pd.pth \
  --mp checkpoints/hierarchical/mp.pth \
  --fp checkpoints/hierarchical/fp.pth
```

Experiment label: `hierarchical-final`. Original seed: unknown. Weight license: Apache-2.0. See `docs/USAGE.md` and `docs/REPRODUCE.md` in the repository.
