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

"""Release integrity, recording provenance and repeated-run safeguards."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from gehirnet.bundle import create_bundle, read_bundle
from gehirnet.datasets import read_records
from gehirnet.prepare import prepare_recordings
from scripts.make_configs import make_configs
from scripts.summarize_metrics import summarize


class WorkflowTests(unittest.TestCase):
    def test_bundle_integrity_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.pth"
            source.write_bytes(b"test checkpoint bytes")
            with patch("gehirnet.models.load_model"):
                create_bundle(root / "bundle", {"pd": source}, "test", 42)
            manifest, paths = read_bundle(root / "bundle")
            self.assertEqual(manifest["classes"], ["HC", "Pathology"])
            paths["pd"].write_bytes(b"corrupted checkpoint")
            with self.assertRaises(ValueError):
                read_bundle(root / "bundle")
            paths["pd"].write_bytes(source.read_bytes())
            manifest["checkpoints"]["pd"]["classes"] = ["FC", "FP", "MC", "MP"]
            (root / "bundle/manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                read_bundle(root / "bundle")

    def test_bundle_requires_complete_stage_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                create_bundle(Path(directory) / "bundle", {"pd": "unused", "mp": "unused"}, "test")

    def test_preparation_preserves_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rate = 44100
            audio = 0.5 * np.sin(2 * np.pi * 220 * np.arange(rate) / rate)
            sf.write(root / "vowel.wav", audio, rate, subtype="FLOAT")
            table = root / "recordings.csv"
            table.write_text("Audio_Path,Dataset,ID,Sex,Pathology,Group\nvowel.wav,synthetic,example,M,HC,MC\n")
            result = prepare_recordings(table, root, root / "features")
            self.assertEqual(result["segments"], 1)
            records = read_records(result["table"], result["data_root"], "pd")
            self.assertEqual(records[0][2]["ID"], "example")
            self.assertEqual(records[0][2]["Dataset"], "synthetic")
            self.assertEqual(records[0][2]["Audio_Path"], "vowel.wav")
            self.assertEqual(np.load(records[0][0]).shape, (1, 128, 98))
            with self.assertRaises(ValueError):
                prepare_recordings(table, root, root / "features")

    def test_seed_plan_and_augmentation_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = make_configs("hierarchical", "train.csv", {}, "features", directory, "outputs")
            configs = [json.loads(Path(path).read_text()) for path in paths]
            self.assertEqual(len(configs), 9)
            self.assertEqual({config["seed"] for config in configs}, {40, 41, 42})
            self.assertEqual(len({config["output_dir"] for config in configs}), 9)
            with self.assertRaises(ValueError):
                make_configs("timewarp", "train.csv", {}, "features", directory, "outputs")

    def test_summary_rejects_different_tasks_and_cohorts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = {"unit": "segment", "samples": 1, "classes": ["HC", "Pathology"], "predictions": [{"path": "sample.npy", "truth": "HC"}], "accuracy": 0.8, "weighted_f1": 0.8, "mcc": 0.8}
            paths = [root / "seed40.json", root / "seed41.json"]
            paths[0].write_text(json.dumps(run))
            run["accuracy"] = 1.0
            paths[1].write_text(json.dumps(run))
            result = summarize(paths, 0)
            self.assertAlmostEqual(result["accuracy"]["mean"], 0.9)
            self.assertAlmostEqual(result["accuracy"]["std"], 0.1)
            run["classes"] = ["HC", "ALS"]
            paths[1].write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                summarize(paths)
            run["classes"] = ["HC", "Pathology"]
            run["predictions"][0]["path"] = "different.npy"
            paths[1].write_text(json.dumps(run))
            with self.assertRaises(ValueError):
                summarize(paths)


if __name__ == "__main__":
    unittest.main()
