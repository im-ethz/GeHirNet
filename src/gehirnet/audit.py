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

"""Metadata audit without loading audio or neural-network dependencies."""

import csv
from collections import Counter
from pathlib import Path


def audit_tables(train_table, test_table, data_root):
    tables, identities = {}, []
    for name, path in (("train", train_table), ("test", test_table)):
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"Dataset", "ID", "Full_Path", "Group", "Pathology"}.issubset(reader.fieldnames or []):
                raise ValueError("Audit requires Dataset, ID, Full_Path, Group and Pathology columns.")
            rows = list(reader)
        keys = {(row["Dataset"], row["ID"]) for row in rows}
        identities.append(keys)
        present = 0
        for row in rows:
            relative = Path(row["Full_Path"].replace("\\", "/"))
            sample = relative if relative.is_absolute() else Path(data_root) / relative
            present += sample.is_file()
        tables[name] = {"segments": len(rows), "dataset_id_keys": len(keys), "existing_paths": present, "groups": dict(Counter(row["Group"] for row in rows)), "pathologies": dict(Counter(row["Pathology"] for row in rows))}
    tables["shared_dataset_id_keys"] = len(identities[0] & identities[1])
    tables["identity_note"] = "Check source-specific ID semantics before interpreting these keys as unique participants."
    return tables
