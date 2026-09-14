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

"""Segment-level paper metrics and exportable predictions."""

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from .datasets import MelDataset, read_records


def evaluate_model(model, table, data_root, batch_size=32):
    records = read_records(table, data_root)
    labels = model.classes
    predictions = []
    for inputs, _ in DataLoader(MelDataset(records), batch_size=batch_size):
        predictions.extend(model.predict(inputs, batch_size))
    truth = [row["Pathology"] if "Pathology" not in labels else ("HC" if row["Pathology"] == "HC" else "Pathology") for _, _, row in records]
    predicted = [p["label"] for p in predictions]
    return {
        "unit": "segment", "samples": len(records), "classes": list(labels),
        "accuracy": accuracy_score(truth, predicted),
        "weighted_f1": f1_score(truth, predicted, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(truth, predicted),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(labels)).tolist(),
        "predictions": [{"path": str(path), "truth": true, **prediction} for (path, _, _), true, prediction in zip(records, truth, predictions)],
    }
