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

"""Strict CSV loading: missing samples never silently change evaluation."""

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .labels import TASK_CLASSES
from .preprocessing import validate_mel


def read_records(table, data_root, task="baseline"):
    classes = TASK_CLASSES[task]
    with open(table, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Full_Path", "Pathology", "Group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        rows = list(reader)
    records = []
    for row in rows:
        if task in ("mp", "fp") and row["Group"] != task.upper():
            continue
        label = row["Group"] if task == "pd" else row["Pathology"]
        if task in ("mp", "fp") and label == "HC":
            continue
        if label not in classes:
            raise ValueError(f"Unknown {task} label: {label}")
        relative = Path(row["Full_Path"].replace("\\", "/"))
        path = relative if relative.is_absolute() else Path(data_root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Sample not found: {path}")
        records.append((path, classes.index(label), row))
    if not records:
        raise ValueError(f"No records for task {task}.")
    return records


class MelDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, _ = self.records[index]
        return validate_mel(np.load(path, allow_pickle=False)), label
