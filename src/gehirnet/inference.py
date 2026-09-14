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

"""Hard routing reproduces the two-stage evaluation notebooks."""

import torch

from .labels import BASELINE_CLASSES, DISEASE_CLASSES, GROUP_CLASSES
from .models import load_model


@torch.inference_mode()
def route_predictions(inputs, pd_model, mp_model, fp_model):
    groups = pd_model(inputs).argmax(dim=1)
    predictions = torch.full_like(groups, DISEASE_CLASSES.index("HC"))
    for group, model in ((1, mp_model), (3, fp_model)):
        mask = groups == group
        if mask.any():
            predictions[mask] = model(inputs[mask]).argmax(dim=1)
    return groups, predictions


class GeHirNet:
    def __init__(self, pd_checkpoint, mp_checkpoint=None, fp_checkpoint=None, device="cpu"):
        if (mp_checkpoint is None) != (fp_checkpoint is None):
            raise ValueError("Supply both MP and FP checkpoints for hierarchical inference.")
        self.device = device
        self.pd = load_model(pd_checkpoint, 4, device)
        self.mp = load_model(mp_checkpoint, 6, device) if mp_checkpoint else None
        self.fp = load_model(fp_checkpoint, 6, device) if fp_checkpoint else None
        self.classes = DISEASE_CLASSES if self.mp is not None else ("HC", "Pathology")

    @torch.inference_mode()
    def predict(self, mels, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if mels.ndim != 4 or len(mels) == 0 or tuple(mels.shape[1:]) != (1, 128, 98) or not torch.isfinite(mels).all():
            raise ValueError("Expected finite input with shape (N, 1, 128, 98).")
        results = []
        for batch in mels.split(batch_size):
            batch = batch.to(self.device)
            if self.mp is None:
                groups = self.pd(batch).argmax(dim=1).cpu().tolist()
                results.extend({"group": GROUP_CLASSES[g], "label": "HC" if g in (0, 2) else "Pathology"} for g in groups)
            else:
                groups, predictions = route_predictions(batch, self.pd, self.mp, self.fp)
                results.extend({"group": GROUP_CLASSES[g], "label": DISEASE_CLASSES[p]} for g, p in zip(groups.cpu().tolist(), predictions.cpu().tolist()))
        return results


class Baseline:
    """Single-stage seven-class model with its distinct checkpoint order."""

    classes = BASELINE_CLASSES

    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        self.model = load_model(checkpoint, 7, device)

    @torch.inference_mode()
    def predict(self, mels, batch_size=32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if mels.ndim != 4 or len(mels) == 0 or tuple(mels.shape[1:]) != (1, 128, 98) or not torch.isfinite(mels).all():
            raise ValueError("Expected finite input with shape (N, 1, 128, 98).")
        results = []
        for batch in mels.split(batch_size):
            predictions = self.model(batch.to(self.device)).argmax(1).cpu().tolist()
            results.extend({"label": self.classes[p]} for p in predictions)
        return results
