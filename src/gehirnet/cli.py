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

"""Portable command-line entry point."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="GeHirNet inference, training and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict each Mel or audio segment")
    predict.add_argument("input", type=Path)
    checkpoints = predict.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--pd", type=Path)
    checkpoints.add_argument("--baseline", type=Path)
    checkpoints.add_argument("--model-dir", type=Path, help="Directory with manifest.json and bundled weights")
    predict.add_argument("--mp", type=Path)
    predict.add_argument("--fp", type=Path)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--already-preprocessed", action="store_true", help="WAV already had VAD and normalization")
    predict.add_argument("--output", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate a CSV of Mel samples")
    evaluate.add_argument("--table", required=True, type=Path)
    evaluate.add_argument("--data-root", required=True, type=Path)
    checkpoints = evaluate.add_mutually_exclusive_group(required=True)
    checkpoints.add_argument("--pd", type=Path)
    checkpoints.add_argument("--baseline", type=Path)
    checkpoints.add_argument("--model-dir", type=Path)
    evaluate.add_argument("--mp", type=Path)
    evaluate.add_argument("--fp", type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--output", required=True, type=Path)
    train = commands.add_parser("train", help="Cross-validate and retrain one classifier")
    train.add_argument("--config", required=True, type=Path)
    audit = commands.add_parser("audit", help="Check CSV paths and dataset/ID overlap")
    audit.add_argument("--train-table", required=True, type=Path)
    audit.add_argument("--test-table", required=True, type=Path)
    audit.add_argument("--data-root", required=True, type=Path)
    bundle = commands.add_parser("bundle", help="Validate and assemble a local downloadable model artifact")
    bundle.add_argument("--pd", type=Path)
    bundle.add_argument("--mp", type=Path)
    bundle.add_argument("--fp", type=Path)
    bundle.add_argument("--baseline", type=Path)
    bundle.add_argument("--output-dir", required=True, type=Path)
    bundle.add_argument("--experiment", required=True)
    bundle.add_argument("--seed", type=int)
    bundle.add_argument("--weight-license", default="UNSPECIFIED")
    prepare = commands.add_parser("prepare", help="Convert a recording CSV to Mel files and a segment CSV")
    prepare.add_argument("--table", required=True, type=Path)
    prepare.add_argument("--audio-root", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    prepare.add_argument("--already-preprocessed", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            from .audit import audit_tables
            print(json.dumps(audit_tables(args.train_table, args.test_table, args.data_root), indent=2))
            return
        if args.command == "train":
            from .train import train_from_config
            train_from_config(args.config)
            return
        if args.command == "bundle":
            from .bundle import create_bundle
            checkpoints = {role: getattr(args, role) for role in ("pd", "mp", "fp", "baseline") if getattr(args, role)}
            manifest = create_bundle(args.output_dir, checkpoints, args.experiment, args.seed, args.weight_license)
            print(json.dumps(manifest, indent=2))
            return
        if args.command == "prepare":
            from .prepare import prepare_recordings
            print(json.dumps(prepare_recordings(args.table, args.audio_root, args.output_root, args.already_preprocessed), indent=2))
            return
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive.")
        from .inference import Baseline, GeHirNet
        if (args.baseline or args.model_dir) and (args.mp or args.fp):
            raise ValueError("MP/FP checkpoints require --pd.")
        if args.model_dir:
            from .bundle import load_bundle
            model, manifest = load_bundle(args.model_dir, args.device)
            mode = manifest["mode"]
        else:
            model = Baseline(args.baseline, args.device) if args.baseline else GeHirNet(args.pd, args.mp, args.fp, args.device)
            mode = "baseline" if args.baseline else ("hierarchical" if args.mp else "pd")
        if args.command == "predict":
            from .preprocessing import load_input
            result = {"input": str(args.input), "mode": mode, "segments": model.predict(load_input(args.input, args.already_preprocessed), args.batch_size)}
        else:
            from .evaluate import evaluate_model
            result = evaluate_model(model, args.table, args.data_root, args.batch_size)
        payload = json.dumps(result, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n")
        print(payload)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
