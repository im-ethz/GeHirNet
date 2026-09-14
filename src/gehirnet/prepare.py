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

"""Prepare reusable segment features from a labeled recording CSV."""

import csv
from pathlib import Path

import numpy as np

from .labels import BASELINE_CLASSES, GROUP_CLASSES
from .preprocessing import audio_to_mels


def prepare_recordings(table, audio_root, output_root, already_preprocessed=False):
    with open(table, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        required = {"Audio_Path", "Dataset", "ID", "Sex", "Pathology", "Group"}
        if not required.issubset(fields):
            raise ValueError(f"Recording CSV requires: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Recording CSV is empty.")
    for row in rows:
        if row["Pathology"] not in BASELINE_CLASSES or row["Group"] not in GROUP_CLASSES or row["Sex"] not in ("M", "F"):
            raise ValueError("Recording contains an unknown pathology, group or sex label.")
        expected_group = row["Sex"] + ("C" if row["Pathology"] == "HC" else "P")
        if row["Group"] != expected_group:
            raise ValueError("Group does not match Sex and Pathology.")
        source = Path(row["Audio_Path"].replace("\\", "/"))
        source = source if source.is_absolute() else Path(audio_root) / source
        if not source.is_file():
            raise FileNotFoundError(f"Recording not found: {source}")
    output = Path(output_root)
    if output.exists():
        raise ValueError("Feature output already exists; choose a new output_root.")
    output.mkdir(parents=True)
    extra = ["Full_Path", "NPY", "segment_index", "segment_duration_seconds", "augmentation_method"]
    fields += [field for field in extra if field not in fields]
    generated = 0
    temporary = output / "segments.csv.partial"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for recording_index, row in enumerate(rows, 1):
            source = Path(row["Audio_Path"].replace("\\", "/"))
            source = source if source.is_absolute() else Path(audio_root) / source
            mels = audio_to_mels(source, already_preprocessed)
            for segment_index, mel in enumerate(mels, 1):
                relative = Path(row["Group"]) / f"recording{recording_index:06d}_segment{segment_index:06d}.npy"
                destination = output / relative
                destination.parent.mkdir(exist_ok=True)
                np.save(destination, mel.numpy())
                writer.writerow({**row, "Full_Path": relative.as_posix(), "NPY": relative.name, "segment_index": segment_index, "segment_duration_seconds": 1.0, "augmentation_method": "original"})
                generated += 1
    temporary.rename(output / "segments.csv")
    return {"recordings": len(rows), "segments": generated, "table": str(output / "segments.csv"), "data_root": str(output), "manual_outlier_removal": "must be applied to the input recording list before this command"}
