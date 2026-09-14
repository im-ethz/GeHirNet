# Model artifacts

`manifest.json` inventories the checkpoint currently found in this workspace. It is not the `manifest.json` inside an inference bundle. The local `checkpoints/pd/final_group_classifier.pth` is a local PD checkpoint and is ignored by Git.

No public download URL has been established. A source-only clone will not contain weights. To use a pretrained model, users must obtain the separately published artifact once its location and terms are provided.

Build a local PD artifact:

```bash
gehirnet bundle --pd checkpoints/pd/final_group_classifier.pth --output-dir outputs/releases/pd-only --experiment unknown
```

The output includes `pd.pth`, `manifest.json`, and usage instructions. The default weight license is `UNSPECIFIED`; this supports local preparation and validation but is not a completed public weight release.

Build a full artifact once matched weights and their provenance are available:

```bash
gehirnet bundle --pd path/to/pd.pth --mp path/to/mp.pth --fp path/to/fp.pth --output-dir outputs/releases/gehirnet --experiment timewarp --seed 42 --weight-license YOUR_CONFIRMED_WEIGHT_LICENSE
```

Replace the license placeholder with the actual terms you have determined for those weights. Include the corresponding license text and updated model card in the published artifact. The bundler checks architecture and label dimensions, but matching experiment/seed provenance remains an author responsibility.

Users then run:

```bash
gehirnet predict path/to/vowel.wav --model-dir path/to/downloaded-bundle
```

See [release instructions](../docs/RELEASE.md) for publication artifacts and [model usage](../docs/USAGE.md) for inputs/outputs.

Inventory filenames are relative to the `checkpoints/` directory. Generated model bundles remain under `outputs/releases/` and are separate downloadable artifacts.
