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

"""Generate independent configs for paper seeds without launching training."""

import argparse
import json
from pathlib import Path


def make_configs(experiment, table, stage_tables, data_root, config_dir, output_root, device="cpu"):
    tasks = ("baseline",) if experiment == "baseline" else ("pd", "mp", "fp")
    if experiment in ("resampling", "timewarp") and any(not stage_tables.get(task) for task in tasks):
        raise ValueError("Augmented experiments require explicit PD, MP and FP training tables.")
    if not table and any(not stage_tables.get(task) for task in tasks):
        raise ValueError("Supply --table or each required task table.")
    paths = []
    for seed in (40, 41, 42):
        for task in tasks:
            config = {"task": task, "table": stage_tables.get(task) or table, "data_root": data_root, "output_dir": str(Path(output_root) / experiment / f"seed{seed}" / task), "seed": seed, "experiment": experiment, "device": device, "pretrained": True, "folds": 5, "epochs": [10, 20, 30], "batch_sizes": [32, 64], "learning_rates": [1e-3, 1e-4, 1e-5]}
            paths.append((Path(config_dir) / f"{experiment}_seed{seed}_{task}.json", config))
    if any(path.exists() for path, _ in paths):
        raise ValueError("Generated config already exists; choose a new config directory.")
    for path, config in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2) + "\n")
    return [str(path) for path, _ in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("baseline", "hierarchical", "resampling", "timewarp"))
    parser.add_argument("--table")
    for role in ("pd", "mp", "fp"):
        parser.add_argument(f"--{role}-table")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        paths = make_configs(args.experiment, args.table, {role: getattr(args, f"{role}_table") for role in ("pd", "mp", "fp")}, args.data_root, args.config_dir, args.output_root, args.device)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"configs": paths, "note": "Training has not been launched."}, indent=2))


if __name__ == "__main__":
    main()
