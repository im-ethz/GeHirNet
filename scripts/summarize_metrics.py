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

"""Summarize repeated runs of the same classification task and cohort."""

import argparse
import json
import math
from pathlib import Path


def summarize(paths, ddof=1):
    if len(paths) <= ddof or len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("Supply distinct metric files and more runs than ddof.")
    runs = [json.loads(Path(path).read_text()) for path in paths]
    reference = (runs[0]["unit"], runs[0]["samples"], set(runs[0]["classes"]))
    cohort = sorted((row["path"], row["truth"]) for row in runs[0]["predictions"])
    for run in runs:
        if (run["unit"], run["samples"], set(run["classes"])) != reference:
            raise ValueError("Cannot mix classification tasks, sample counts or evaluation units.")
        if sorted((row["path"], row["truth"]) for row in run["predictions"]) != cohort:
            raise ValueError("Metric files refer to different test cohorts.")
    result = {"runs": len(runs), "ddof": ddof, "unit": reference[0], "samples": reference[1], "classes": runs[0]["classes"], "files": [str(path) for path in paths]}
    for metric in ("accuracy", "weighted_f1", "mcc"):
        values = [float(run[metric]) for run in runs]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Metrics must be finite.")
        mean = sum(values) / len(values)
        result[metric] = {"mean": mean, "std": math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - ddof)), "values": values}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.metrics, args.ddof)
    except (ValueError, KeyError, FileNotFoundError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
