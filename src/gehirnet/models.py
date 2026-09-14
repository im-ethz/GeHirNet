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

"""ResNet-50 with the state-dict layout of the paper checkpoints."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.base_model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


def load_model(path: str | Path, num_classes: int, device: str = "cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = ResNet50Classifier(num_classes)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
