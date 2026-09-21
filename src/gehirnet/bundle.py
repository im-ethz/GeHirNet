# Copyright 2025 ETH Zurich
# Project contributors:
#   Fan Wu (ETH Zurich)
#   Kaicheng Zhao (RWTH Aachen University)
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Portable local model artifacts with validated label order and checksums."""

import hashlib
import json
import shutil
from pathlib import Path

from .labels import BASELINE_CLASSES, DISEASE_CLASSES, GROUP_CLASSES, PATHOLOGY_CLASSES

PREPROCESSING_SPEC = {
    "input": "mono sustained vowel /a/",
    "mel_shape": [1, 128, 98],
    "vad": {"window": 2048, "hop": 512, "threshold": 0.001, "fade_and_crossfade": 512},
    "normalization": "min-max [0, 1] after VAD, before segmentation",
    "segment_seconds": 1.0,
    "step_seconds": 0.4,
    "tail_padding": "wrap",
    "mel": {"bins": 128, "scale": "power dB", "44100_fft_hop": [1808, 452], "dynamic_base_rate_fft_hop": [45000, 1848, 462]},
    "other_sample_rates": "40000 to 50000 Hz in 125 Hz steps",
    "routing": "PD argmax, then MP/FP argmax on the same Mel input",
    "aggregation": "none; predictions are per segment",
}
ROLE_CLASSES = {"pd": GROUP_CLASSES, "mp": PATHOLOGY_CLASSES, "fp": PATHOLOGY_CLASSES, "baseline": BASELINE_CLASSES}
MODE_ROLES = {"pd": {"pd"}, "hierarchical": {"pd", "mp", "fp"}, "baseline": {"baseline"}}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_bundle(output_dir, checkpoints, experiment, seed=None, weight_license="UNSPECIFIED"):
    from .models import load_model

    output = Path(output_dir)
    if output.exists():
        raise ValueError("Bundle output already exists; choose a new directory.")
    mode = next((mode for mode, roles in MODE_ROLES.items() if set(checkpoints) == roles), None)
    if mode is None:
        raise ValueError("Supply baseline alone, PD alone, or PD/MP/FP together.")
    if not experiment.strip() or not weight_license.strip():
        raise ValueError("Experiment and weight license metadata must not be empty.")
    artifacts = {}
    for role, path in checkpoints.items():
        model = load_model(path, len(ROLE_CLASSES[role]))
        del model
        artifacts[role] = {"file": f"{role}.pth", "classes": list(ROLE_CLASSES[role]), "sha256": checksum(path)}
    output.mkdir(parents=True)
    for role, path in checkpoints.items():
        destination = output / artifacts[role]["file"]
        shutil.copyfile(path, destination)
        if checksum(destination) != artifacts[role]["sha256"]:
            raise ValueError(f"Checkpoint changed while copying: {role}")
    project_license = Path(__file__).resolve().parents[2] / "LICENSE.txt"
    if project_license.is_file():
        shutil.copyfile(project_license, output / "LICENSE.txt")
    manifest = {
        "format_version": 1, "architecture": "single-channel ResNet-50", "mode": mode,
        "experiment": experiment, "seed": seed, "weight_license": weight_license,
        "classes": list(BASELINE_CLASSES if mode == "baseline" else DISEASE_CLASSES if mode == "hierarchical" else ("HC", "Pathology")),
        "preprocessing": PREPROCESSING_SPEC, "checkpoints": artifacts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "README.md").write_text(
        f"# GeHirNet {mode} model artifact\n\n"
        f"Experiment: {experiment}; seed: {seed}; weight license: {weight_license}.\n\n"
        "Install the GeHirNet source package, then run:\n\n"
        "```bash\ngehirnet predict path/to/vowel.wav --model-dir path/to/this-directory\n```\n\n"
        "See manifest.json for class order, preprocessing and checkpoint checksums. "
        + ("An UNSPECIFIED weight license is unresolved release metadata, not permission to redistribute. " if weight_license == "UNSPECIFIED" else "")
        + "This artifact does not establish agreement with published metrics or clinical validity.\n"
    )
    return manifest


def read_bundle(model_dir):
    root = Path(model_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    mode = manifest.get("mode")
    if manifest.get("format_version") != 1 or mode not in MODE_ROLES:
        raise ValueError("Unsupported model manifest format or mode.")
    expected_classes = BASELINE_CLASSES if mode == "baseline" else DISEASE_CLASSES if mode == "hierarchical" else ("HC", "Pathology")
    if manifest.get("classes") != list(expected_classes) or manifest.get("preprocessing") != PREPROCESSING_SPEC:
        raise ValueError("Bundle class order or preprocessing differs from this implementation.")
    artifacts = manifest.get("checkpoints", {})
    if set(artifacts) != MODE_ROLES[mode]:
        raise ValueError("Bundle is missing required checkpoints or contains unexpected roles.")
    paths = {}
    for role, artifact in artifacts.items():
        filename = artifact.get("file", "")
        if not filename or filename != Path(filename).name:
            raise ValueError("Checkpoint filenames must refer to files inside the bundle.")
        path = root / filename
        if artifact.get("classes") != list(ROLE_CLASSES[role]) or checksum(path) != artifact.get("sha256"):
            raise ValueError(f"Checkpoint checksum or label order mismatch: {role}")
        paths[role] = path
    return manifest, paths


def load_bundle(model_dir, device="cpu"):
    from .inference import Baseline, GeHirNet

    manifest, paths = read_bundle(model_dir)
    if manifest["mode"] == "baseline":
        model = Baseline(paths["baseline"], device)
    else:
        model = GeHirNet(paths["pd"], paths.get("mp"), paths.get("fp"), device)
    return model, manifest
