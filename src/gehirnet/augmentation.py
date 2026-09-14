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

"""Audio augmentation primitives from the paper (before Mel extraction)."""

import numpy as np
import torch
import torchaudio.transforms as T

from .preprocessing import crossfade

RESAMPLING_RATES = tuple(rate for rate in range(40000, 50001, 125) if rate not in (48000, 50000))


def resample_audio(audio, original_rate, target_rate):
    if target_rate not in RESAMPLING_RATES:
        raise ValueError("Target rate must be an augmentation rate in 40000–50000 Hz.")
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    return T.Resample(original_rate, target_rate)(waveform).numpy()


def timewarp_audio(audio, order):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) // 5 < 32:
        raise ValueError("Time warping requires mono audio with at least 160 samples.")
    if sorted(order) != list(range(5)):
        raise ValueError("order must be a permutation of 0, 1, 2, 3, 4.")
    size = len(audio) // 5
    segments = [audio[i * size:(i + 1) * size] for i in order]
    return crossfade(segments, 32)
