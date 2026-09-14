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

"""Meaningful compatibility checks; run with unittest (no test dependency)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gehirnet.datasets import read_records
from gehirnet.evaluate import evaluate_model
from gehirnet.inference import Baseline, route_predictions
from gehirnet.labels import BASELINE_CLASSES, DISEASE_CLASSES
from gehirnet.models import ResNet50Classifier
from gehirnet.preprocessing import audio_to_mels, crossfade, segment_audio, validate_mel


class Fixed(nn.Module):
    def __init__(self, classes, predictions):
        super().__init__()
        self.classes, self.predictions = classes, predictions
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return torch.nn.functional.one_hot(torch.tensor(self.predictions[:len(inputs)]), self.classes).float()


class CoreTests(unittest.TestCase):
    def test_baseline_checkpoint_order(self):
        model = object.__new__(Baseline)
        model.device = "cpu"
        model.model = Fixed(7, [3, 4])
        self.assertEqual(model.predict(torch.zeros(2, 1, 128, 98)), [{"label": "HC"}, {"label": "Laryngitis"}])

    def test_baseline_evaluation_order(self):
        model = object.__new__(Baseline)
        model.device = "cpu"
        model.model = Fixed(7, [3, 0])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "hc.npy", np.zeros((1, 128, 98), dtype=np.float32))
            np.save(root / "als.npy", np.zeros((1, 128, 98), dtype=np.float32))
            table = root / "test.csv"
            table.write_text("Full_Path,Pathology,Group\nhc.npy,HC,MC\nals.npy,ALS,MP\n")
            metrics = evaluate_model(model, table, root, 2)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertEqual(metrics["confusion_matrix"][3][3], 1)
            self.assertEqual(metrics["confusion_matrix"][0][0], 1)

    def test_routing_and_label_order(self):
        pd, mp, fp = Fixed(4, [0, 1, 2, 3]), Fixed(6, [4]), Fixed(6, [0])
        groups, labels = route_predictions(torch.zeros(4, 1, 128, 98), pd, mp, fp)
        self.assertEqual(groups.tolist(), [0, 1, 2, 3])
        self.assertEqual(labels.tolist(), [6, 4, 6, 0])
        self.assertEqual(BASELINE_CLASSES.index("HC"), 3)
        self.assertEqual(DISEASE_CLASSES.index("HC"), 6)
        mp, fp = Fixed(6, [0]), Fixed(6, [0])
        route_predictions(torch.zeros(2, 1, 128, 98), Fixed(4, [0, 2]), mp, fp)
        self.assertEqual((mp.calls, fp.calls), (0, 0))

    def test_mel_validation(self):
        self.assertEqual(tuple(validate_mel(np.zeros((128, 98))).shape), (1, 128, 98))
        for invalid in [np.zeros((98, 128)), np.full((1, 128, 98), np.nan)]:
            with self.assertRaises(ValueError):
                validate_mel(invalid)

    def test_short_audio_and_crossfade(self):
        self.assertEqual(len(segment_audio(np.arange(100), 44100)[0]), 44100)
        self.assertEqual(len(crossfade([np.ones(10), np.ones(5)], 512)), 10)

    def test_audio_mel_shape(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vowel.wav"
            for rate in (44100, 48000, 50000):
                audio = 0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
                sf.write(path, audio, rate, subtype="FLOAT")
                mel = audio_to_mels(path)
                self.assertEqual(tuple(mel.shape), (1, 1, 128, 98))
                self.assertTrue(torch.isfinite(mel).all())

    def test_missing_samples_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.csv"
            path.write_text("Full_Path,Pathology,Group\nmissing.npy,HC,MC\n")
            with self.assertRaises(FileNotFoundError):
                read_records(path, folder)

    def test_checkpoint_layout(self):
        model = ResNet50Classifier(4)
        self.assertEqual(tuple(model.state_dict()["base_model.conv1.weight"].shape), (64, 1, 7, 7))
        self.assertEqual(tuple(model.state_dict()["base_model.fc.weight"].shape), (4, 2048))


if __name__ == "__main__":
    unittest.main()
