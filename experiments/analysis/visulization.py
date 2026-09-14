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
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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




# Path to the .npy file
folder = "data/features/Mel_NPY_Output"

file_train = pd.read_csv("data/metadata/train_set.csv")
file_test = pd.read_csv("data/metadata/test_set.csv")
file_table = pd.concat([file_train, file_test])

# Map categorical labels to numerical values
sex_mapping = {'F': 0, 'M': 1}  # Female: 0, Male: 1
disease_mapping = {'HC': 0, 'Covid-19': 1, 'Parkinson': 2, 'Dysphonie': 3, 'Rekurrensparese': 4, 'Laryngitis': 5, 'ALS': 6}  

sex_map = {0: 'Female', 1: 'Male'}  # Female: 0, Male: 1
disease_map = {0: 'Health Control', 1: 'COVID-19', 2: 'Parkinson', 3: 'Dysphonia', 4: 'Vocal Cord Paresis', 5: 'Laryngitis', 6: 'ALS'}  


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
            mean_spectrograms[group_name] = np.mean(X[mask], axis=0)  # Shape (128, 98)



# UMAP Clustering
# X: (n_samples, 128, 98)
n_samples = X.shape[0]
X_flat = X.reshape(n_samples, -1)  # now shape is (n_samples, 128*98)

# Combine labels into one string for coloring
y = np.array([f"{sex}_{disease}" for sex, disease in zip(y2.flatten(), y1.flatten())])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_flat)

reducer = umap.UMAP(n_components=2, random_state=42)
X_embedded = reducer.fit_transform(X_scaled)  # shape: (n_samples, 2)







# -------------------------
# Configuration
# -------------------------
sex_map = {0: "Female", 1: "Male"}
sex_markers = {0: "^", 1: "o"}

diseases = sorted(np.unique(y2))
n_d = len(diseases)


viridis = cm.get_cmap("viridis")

sex_colors = {
    0: viridis(0.25),  # Female: darker
    1: viridis(0.75),  # Male: brighter
}

# -------------------------
# Create 3 × 7 layout
# -------------------------
fig, axes = plt.subplots(
    nrows=3,
    ncols=n_d,
    figsize=(3*n_d, 8)
)

# -------------------------
# Row 1 & 2: Mean Mel spectrograms
# -------------------------
for col, disease in enumerate(diseases):
    if disease == 0:
        spec_f = mean_spectrograms[f"F_HC"]
        spec_m = mean_spectrograms[f"M_HC"]
    else:
        spec_f = mean_spectrograms[f"F_D{disease}"]
        spec_m = mean_spectrograms[f"M_D{disease}"]

    # Female
    sns.heatmap(
        spec_f,
        ax=axes[0, col],
        cmap="viridis",
        cbar=False
    )

    # Male
    sns.heatmap(
        spec_m,
        ax=axes[1, col],
        cmap="viridis",
        cbar=False
    )

    # Disease title (only once, on top row)
    axes[0, col].set_title(disease_map[disease], fontsize=14)

    # Clean axes
    for row in [0, 1]:
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        axes[row, col].set_xlabel("")
        axes[row, col].set_ylabel("")

# -------------------------
# Row 3: UMAP per disease
# -------------------------
for col, disease in enumerate(diseases):
    ax = axes[2, col]

    idx_d = y2.flatten() == disease

    for sex in np.unique(y1):
        idx = idx_d & (y1.flatten() == sex)

        ax.scatter(
            X_embedded[idx, 0],
            X_embedded[idx, 1],
            marker=sex_markers[sex],
            c=[sex_colors[sex]],
            s=14,
            alpha=0.75,
            edgecolors="none",
            label=sex_map[sex]
        )

    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend ONLY to last column of UMAP row
    if col == n_d - 1:  # last column
        ax.legend(
            fontsize=14,
            loc='lower left',
            frameon=True,        # add box
            fancybox=True,       # rounded corners
            edgecolor='black',   # box border color
            framealpha=0.9       # slightly transparent
        )
# -------------------------
# Row labels (left side)
# -------------------------
axes[0, 0].set_ylabel("Female\nMel", fontsize=14)
axes[1, 0].set_ylabel("Male\nMel", fontsize=14)
axes[2, 0].set_ylabel("UMAP", fontsize=14)

'''
# -------------------------
# Global legend (UMAP row only)
# -------------------------
handles, labels = axes[2, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False
)
'''
# -------------------------
# Final layout & save
# -------------------------
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(
    "MelSpectrograms_and_UMAP_by_Disease.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()









sys.exit()

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


# Example encoding
# 0 = female, 1 = male
sex_markers = {
    0: '^',   # Female
    1: 'o'    # Male
}

for disease in np.unique(y2):
    idx_d = y2.flatten() == disease

    plt.figure(figsize=(6, 5))
    for sex in np.unique(y1):
        idx = idx_d & (y1.flatten() == sex)
        plt.scatter(
            X_embedded[idx, 0],
            X_embedded[idx, 1],
            marker=sex_markers[sex],
            label=sex_map[sex],
            alpha=0.7
        )

    plt.title(f"{disease_map[disease]}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.tight_layout()
    plt.show()





'''
plt.figure(figsize=(10, 8))

for label in np.unique(y):
    idx = y == label
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=label, alpha=0.7)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Mel Spectrograms by Sex × Disease")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend outside plot
plt.tight_layout()
plt.show()
'''





'''
plt.figure(figsize=(10, 8))

for disease in np.unique(y2):
    for sex in np.unique(y1):
        idx = (y2.flatten() == disease) & (y1.flatten() == sex)

        plt.scatter(
            X_embedded[idx, 0],
            X_embedded[idx, 1],
            marker=sex_markers[sex],
            alpha=0.7,
            label=f"Disease {disease}, Sex {sex}"
        )

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of Mel Spectrograms\nColor = Disease, Marker = Sex")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
'''