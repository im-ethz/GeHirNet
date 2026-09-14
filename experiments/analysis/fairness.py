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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats import friedmanchisquare


import pandas as pd

# Load your CSV
# Packaging change: resolve the archived aggregate table beside this script.
from pathlib import Path
df = pd.read_csv(Path(__file__).with_name('Fairness.csv'))

# Automatically strip spaces from all object (string) columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Function to compute EOdds
def compute_eodds(row):
    tpr_diff = abs(row['MaleTPR'] - row['FemaleTPR'])
    fpr_diff = abs(row['MaleFPR'] - row['FemaleFPR'])
    return (tpr_diff + fpr_diff) / 2

# Apply the function
df['EOdds'] = df.apply(compute_eodds, axis=1)
df['macroTPR'] = (df['MaleTPR'] + df['FemaleTPR']) / 2

disease_mapping = {'HC': 0, 'Covid-19': 1, 'Parkinson': 2, 'Dysphonie': 3, 'Rekurrensparese': 4, 'Laryngitis': 5, 'ALS': 6}  

pivot_df = df.pivot(index='Disease', columns='Model', values='EOdds')
print(pivot_df)

# Extract EOdds for each model as separate arrays
model_eodds = [pivot_df[model].values for model in sorted(pivot_df.columns)]

# Run Friedman test
stat, p_value = friedmanchisquare(*model_eodds)

print(f"Friedman statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("Significant differences between models detected (reject H0)")
else:
    print("No significant differences between models detected (fail to reject H0)")



# Pivot table: rows = Disease, columns = Model, values = macroTPR
pivot_macroTPR = df.pivot_table(index='Disease', columns='Model', values='macroTPR', aggfunc='mean')
model_arrays = [pivot_macroTPR[model].values for model in sorted(pivot_macroTPR.columns)]

# Run Friedman test
stat, p_value = friedmanchisquare(*model_arrays)

print(f"Friedman statistic: {stat:.4f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print("Significant differences between models detected")
else:
    print("No significant differences between models detected")