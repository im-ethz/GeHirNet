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

import matplotlib.pyplot as plt
import numpy as np

# Order: Baseline, Ablation-1, Ablation-2, Best model
labels = [
    "Exp1 (Baseline)",
    "Sex as input feature",
    "Increase network depth",
    "Exp3.2 (GeHirNet**)",
]
metrics = ["Accuracy", "F1", "MCC"]

means = np.array([
    [0.9598, 0.9590, 0.9190],   # Baseline
    [0.9556, 0.9551, 0.9109],   # Ablation: sex as input
    [0.9600, 0.9596, 0.9201],   # Ablation: deeper net
    [0.9724, 0.9723, 0.9449],   # Best
])
# Std only for the two main experiments
stds = np.array([
    [0.0063, 0.0068, 0.0131],   # Baseline
    [0.0,    0.0,    0.0   ],   # Ablation (no std reported)
    [0.0,    0.0,    0.0   ],   # Ablation (no std reported)
    [0.0037, 0.0038, 0.0074],   # Best
])

# Cool blues for main experiments, warm tones for ablations
colors  = ["#A6CEE3", "#E78A4E", "#C44E52", "#08306B"]
hatches = ["",        "//",      "//",      ""]  # mark ablations with hatching

x = np.arange(len(metrics))
width = 0.20

fig, ax = plt.subplots(figsize=(10, 6))

for i, lab in enumerate(labels):
    offset = (i - 1.5) * width
    yerr = stds[i, :] if stds[i, :].any() else None
    bars = ax.bar(
        x + offset, means[i, :], width,
        yerr=yerr, capsize=3,
        label=lab, color=colors[i],
        edgecolor="white", linewidth=0.8,
        hatch=hatches[i],
        error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
    )
    for bar, val in zip(bars, means[i, :]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.0015,
            f"{val:.4f}",
            ha="center", va="bottom",
            fontsize=8, color="#222222",
        )

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Ablation Studies vs. Baseline and Best Model",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0.89, 1.00)
ax.legend(loc="upper right", frameon=True, fontsize=9, ncol=1)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Improvement arrows: Baseline -> Best for each metric
for j in range(len(metrics)):
    base_val = means[0, j]
    best_val = means[-1, j]
    delta = best_val - base_val
    best_x = x[j] + (3 - 1.5) * width
    arrow_x = best_x + width * 0.7
    ax.annotate(
        "",
        xy=(arrow_x, best_val),
        xytext=(arrow_x, base_val),
        arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.4),
    )
    ax.text(
        arrow_x + 0.015,
        (base_val + best_val) / 2,
        f"+{delta:.4f}",
        ha="left", va="center",
        fontsize=9, color="#C44E52", fontweight="bold",
    )

plt.tight_layout()
out = "/Users/fanwufan/Documents/PhD/Research/VoiceKaicheng/demo_code/ablation_results.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
