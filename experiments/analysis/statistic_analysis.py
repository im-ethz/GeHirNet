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
from scipy.stats import t

def load_data_from_folder(folder_path, file_table, sex_mapping, disease_mapping):
    X_list = []
    Y_sex_list = []
    Y_disease_list = []
    db_list = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            
            if file.endswith('.npy'):  

                file_path = os.path.join(root, file)

                data = np.load(file_path)[0] 
                
                # Compute mean power per subject
                mean_power_db = data.mean()
                
                X_list.append(data)
                db_list.append(mean_power_db)

                sex = file_table[file_table['NPY']==file]['Sex'].iloc[0]
                pathology = file_table[file_table['NPY']==file]['Pathology'].iloc[0]

                # Convert sex to numerical (F->0, M->1)
                Y_sex_list.append(sex_mapping[sex])
                
                # Convert disease to numerical
                Y_disease_list.append(disease_mapping[pathology])
    
    # Stack all arrays
    X = np.stack(X_list)  # (n, 128, 98)
    Y_sex = np.array(Y_sex_list).reshape(-1, 1)  # (n, 1)
    Y_disease = np.array(Y_disease_list).reshape(-1, 1)  # (n, 1)
    dbs = np.array(db_list).reshape(-1, 1)  # (n, 1)

    return X, Y_sex, Y_disease, dbs


def bootstrap_ci(x, y, n_boot=10000, alpha=0.05):
    boot_diffs = []
    for _ in range(n_boot):
        x_resample = np.random.choice(x, size=len(x), replace=True)
        y_resample = np.random.choice(y, size=len(y), replace=True)
        boot_diffs.append(np.mean(x_resample) - np.mean(y_resample))

    lower = np.percentile(boot_diffs, 100 * alpha/2)
    upper = np.percentile(boot_diffs, 100 * (1 - alpha/2))
    return lower, upper

# Path to the .npy file
folder = "data/features/Mel_NPY_Output"

file_train = pd.read_csv("data/metadata/train_set.csv")
file_test = pd.read_csv("data/metadata/test_set.csv")
file_table = pd.concat([file_train, file_test])

# Map categorical labels to numerical values
sex_mapping = {'F': 0, 'M': 1}  # Female: 0, Male: 1
disease_mapping = {'HC': 0, 'Covid-19': 1, 'Parkinson': 2, 'Dysphonie': 3, 'Rekurrensparese': 4, 'Laryngitis': 5, 'ALS': 6}  

X, y1, y2, dbs = load_data_from_folder(folder, file_table, sex_mapping, disease_mapping)

sex_labels = y1.flatten()
disease_labels = y2.flatten()

# Dictionary to store group statistics (mean and std)
group_stats = {}

# Mean mel spectrogram for each group (e.g., Male/Female × Disease_A/Disease_B).
# Initialize a dictionary to store mean spectrograms
mean_spectrograms = {}

# Iterate through each sex and disease combination
for sex in [0, 1]:
    for disease in range(7):
        mask = (sex_labels == sex) & (disease_labels == disease)
        if np.sum(mask) > 0:  # Check if group exists
            group_name = f"{'F' if sex == 0 else 'M'}_{f'D{disease}' if disease > 0 else 'HC'}"
            print(group_name)
            mean_spectrograms[group_name] = np.mean(X[mask], axis=0)  # Shape (128, 98)

            # Compute the mean and std of the mean power for the group
            group_mean = np.mean(dbs[mask])  # Mean power for the group
            group_std = np.std(dbs[mask])    # Standard deviation for the group

            # Store statistics for the group
            group_stats[group_name] = {
                'mean': group_mean,
                'std': group_std,
                'n': np.sum(mask)  # Number of subjects in the group
            }

# Print out group statistics
for group_name, stats in group_stats.items():
    print(f"Group {group_name}: {stats['mean']:.2f} \pm {stats['std']:.2f}")


# Iterate through each disease combination
for disease in range(7):
    dbs_group1 = dbs[(sex_labels == 0) & (disease_labels == disease)]  # female
    dbs_group2 = dbs[(sex_labels == 1) & (disease_labels == disease)]  # male

    dbs_group1 = np.asarray(dbs_group1).ravel()
    dbs_group2 = np.asarray(dbs_group2).ravel()
    
    # Sample sizes
    n1, n2 = len(dbs_group1), len(dbs_group2)

    # Means and variances
    mean1, mean2 = np.mean(dbs_group1), np.mean(dbs_group2)
    var1, var2 = np.var(dbs_group1, ddof=1), np.var(dbs_group2, ddof=1)

    # Mean difference
    mean_diff = mean1 - mean2

    # Shapiro-Wilk test for normality
    _, p_value1 = shapiro(dbs_group1)
    _, p_value2 = shapiro(dbs_group2)
    
    if p_value1 > 0.05 and p_value2 > 0.05:
        # Independent t-test
        stat, p_test = ttest_ind(dbs_group1, dbs_group2)

        # Standard error (pooled, since ttest_ind assumes equal variances by default)
        se = np.sqrt(var1/n1 + var2/n2)

        # Degrees of freedom (Welch is safer)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

        # 95% CI
        alpha = 0.05
        t_crit = t.ppf(1 - alpha/2, df)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se

        print(f"Mean diff = {mean_diff:.3f}, 95% CI = [{ci_low:.3f}, {ci_high:.3f}]")

        # Cohen's d (effect size)
        n1, n2 = len(dbs_group1), len(dbs_group2)               
        pooled_std = np.sqrt(((n1-1)*np.std(dbs_group1, ddof=1)**2 + (n2-1)*np.std(dbs_group2, ddof=1)**2) / (n1 + n2 - 2))
        cohen_d = (np.mean(dbs_group1) - np.mean(dbs_group2)) / pooled_std

        print(f"T-test for Disease {disease}: p-value = {p_test}, {'significant' if p_test < 0.05 else 'not'}")
    else:
        # Mann-Whitney U test for independent samples
        stat, p_test = mannwhitneyu(dbs_group1, dbs_group2)

        ci_low, ci_high = bootstrap_ci(dbs_group1, dbs_group2)

        print(f"Mean diff = {mean_diff:.3f}, 95% bootstrap CI = [{ci_low:.3f}, {ci_high:.3f}]")

        # Rank-biserial correlation (effect size)
        n1, n2 = len(dbs_group1), len(dbs_group2)
        r = 1 - (2 * stat) / (n1 * n2)  # Equivalent to Cliff's delta
        print(f"Mann-Whitney U test for Disease {disease}: p-value = {p_test}, {'significant' if p_test < 0.05 else 'not'}")




n_rows, n_cols = 2, 7
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
#plt.suptitle("Average Mel Spectrograms by Gender and Disease", fontsize=12, y=0.98)

for idx, (group_name, spectrogram) in enumerate(mean_spectrograms.items()):
    row, col = idx // n_cols, idx % n_cols
    sns.heatmap(spectrogram, ax=axes[row, col], cmap='viridis', cbar=False)
    axes[row, col].set_title(group_name)
    axes[row, col].set_xlabel("")
    axes[row, col].set_ylabel("")

plt.tight_layout()
plt.savefig('Average_Mel_Spectrograms.png', dpi=300, bbox_inches='tight')  # Save first
plt.show()  # Display after saving



