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

"""Paper preprocessing, with explicit handling of invalid audio."""

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T


def validate_mel(mel):
    mel = np.asarray(mel, dtype=np.float32)
    if mel.shape == (128, 98):
        mel = mel[None, ...]
    if mel.shape != (1, 128, 98) or not np.isfinite(mel).all():
        raise ValueError("Expected a finite Mel array of shape (1, 128, 98) or (128, 98).")
    return torch.from_numpy(np.ascontiguousarray(mel))


def mel_parameters(sample_rate: int):
    if sample_rate == 44100:
        return 1808, 452
    if 40000 <= sample_rate <= 50000 and (sample_rate - 40000) % 125 == 0:
        return int(1848 * sample_rate / 45000), int(462 * sample_rate / 45000)
    raise ValueError("Supported sample rates: 44100 Hz or 40000–50000 Hz in 125 Hz steps.")


def crossfade(segments, samples: int):
    if not segments:
        return np.empty(0, dtype=np.float32)
    output = segments[0].copy()
    for segment in segments[1:]:
        overlap = min(samples, len(output), len(segment))
        if overlap == 0:
            output = np.concatenate((output, segment))
            continue
        alpha = np.linspace(0, 1, overlap)
        blend = (1 - alpha) * output[-overlap:] + alpha * segment[:overlap]
        output = np.concatenate((output[:-overlap], blend, segment[overlap:]))
    return output


def remove_silence(audio, threshold=1e-3):
    regions = []
    start = None
    for offset in range(0, len(audio) - 2048 + 1, 512):
        energy = np.sqrt(np.mean(audio[offset:offset + 2048] ** 2))
        if energy > threshold and start is None:
            start = offset
        elif energy < threshold and start is not None:
            regions.append(audio[start:offset])
            start = None
    if start is not None:
        regions.append(audio[start:])
    faded = []
    for region in regions:
        region = region.copy()
        length = len(region) // 2 if len(region) < 512 else 512
        if length:
            region[:length] *= np.linspace(0, 1, length)
            region[-length:] *= np.linspace(1, 0, length)
        faded.append(region)
    return crossfade(faded, 512)


def segment_audio(audio, sample_rate):
    if not len(audio):
        raise ValueError("Audio contains no voiced samples.")
    count = max(1, math.ceil((len(audio) / sample_rate - 1) / 0.4) + 1)
    segments = []
    for index in range(count):
        start = int(index * 0.4 * sample_rate)
        segment = audio[start:start + sample_rate]
        if len(segment) < sample_rate:
            segment = np.pad(segment, (0, sample_rate - len(segment)), mode="wrap")
        segments.append(segment)
    return segments


def audio_to_mels(path: str | Path, already_preprocessed=False):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError("Expected mono audio; convert channels explicitly before inference.")
    audio = audio[:, 0]
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite.")
    window, hop = mel_parameters(sample_rate)
    if not already_preprocessed:
        audio = remove_silence(audio)
        if not len(audio) or np.ptp(audio) == 0:
            raise ValueError("Audio is silent or constant after silence removal.")
        audio = (audio - audio.min()) / np.ptp(audio)
    transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=window, hop_length=hop, n_mels=128)
    to_db = T.AmplitudeToDB()
    mels = []
    for segment in segment_audio(audio, sample_rate):
        mel = to_db(transform(torch.from_numpy(segment.astype(np.float32)).unsqueeze(0)))
        mels.append(validate_mel(mel.numpy()))
    return torch.stack(mels)


def load_input(path, already_preprocessed=False):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return validate_mel(np.load(path, allow_pickle=False)).unsqueeze(0)
    if path.suffix.lower() == ".wav":
        return audio_to_mels(path, already_preprocessed)
    raise ValueError("Input must be a .npy Mel array or a .wav recording.")
