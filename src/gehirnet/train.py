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

"""Five-fold parameter selection followed by full-training-set retraining."""

import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .datasets import MelDataset, read_records
from .labels import TASK_CLASSES
from .models import ResNet50Classifier


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit(records, num_classes, epochs, batch_size, learning_rate, device, pretrained):
    model = ResNet50Classifier(num_classes, pretrained=pretrained).to(device)
    loader = DataLoader(MelDataset(records), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def validation_mcc(model, records, batch_size, device):
    truth, predictions = [], []
    for inputs, labels in DataLoader(MelDataset(records), batch_size=batch_size):
        truth.extend(labels.tolist())
        predictions.extend(model(inputs.to(device)).argmax(dim=1).cpu().tolist())
    return float(matthews_corrcoef(truth, predictions))


def train_from_config(path):
    config = json.loads(Path(path).read_text())
    task = config["task"]
    records = read_records(config["table"], config["data_root"], task)
    classes = TASK_CLASSES[task]
    folds = config.get("folds", 5)
    epochs_grid = config.get("epochs", [10, 20, 30])
    batches = config.get("batch_sizes", [32, 64])
    rates = config.get("learning_rates", [1e-3, 1e-4, 1e-5])
    if folds < 2 or not epochs_grid or not batches or not rates or min(epochs_grid) < 1 or min(batches) < 1 or min(rates) <= 0:
        raise ValueError("Invalid cross-validation grid.")
    counts = np.bincount([r[1] for r in records], minlength=len(classes))
    if counts.min() < folds:
        raise ValueError("Every task class must have at least one sample per validation fold.")
    seed = config.get("seed", 42)
    device = config.get("device", "cpu")
    output = Path(config["output_dir"])
    if (output / "model.pth").exists() or (output / "training.json").exists():
        raise ValueError("Output already contains a trained model; choose a new output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.arange(len(records)), [r[1] for r in records]))
    results = []
    for epochs, batch_size, rate in itertools.product(epochs_grid, batches, rates):
        scores = []
        for fold, (train_indices, val_indices) in enumerate(splits):
            seed_everything(seed + fold)
            model = fit([records[i] for i in train_indices], len(classes), epochs, batch_size, rate, device, config.get("pretrained", True))
            scores.append(validation_mcc(model, [records[i] for i in val_indices], batch_size, device))
            del model
        result = {"epochs": epochs, "batch_size": batch_size, "learning_rate": rate, "fold_mcc": scores, "mean_mcc": float(np.mean(scores))}
        results.append(result)
        print(json.dumps(result), flush=True)
    best = max(results, key=lambda result: result["mean_mcc"])
    seed_everything(seed)
    model = fit(records, len(classes), best["epochs"], best["batch_size"], best["learning_rate"], device, config.get("pretrained", True))
    torch.save({key: value.cpu() for key, value in model.state_dict().items()}, output / "model.pth")
    report = {"config": config, "classes": list(classes), "selection": best, "cv_results": results, "split_unit": "segment", "folds": [{"train": [str(records[i][0]) for i in train], "validation": [str(records[i][0]) for i in validation]} for train, validation in splits]}
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
