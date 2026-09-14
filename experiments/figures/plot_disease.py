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

diseases = [
    "Dysphonia", "Laryngitis", "Vocal Cord\nParesis",
    "COVID-19", "Parkinson's", "ALS",
]
models = ["Prior SOTA", "Baseline", "GeHirNet", "GeHirNet**"]

# rows = diseases, cols = models
acc = np.array([
    [0.990, 0.986, 0.990, 0.992],   # Dysphonia
    [0.983, 0.997, 0.997, 0.998],   # Laryngitis
    [0.941, 0.989, 0.994, 0.996],   # Vocal Cord Paresis
    [0.999, 0.979, 0.986, 0.991],   # COVID-19
    [0.997, 0.995, 0.997, 0.997],   # Parkinson's
    [0.997, 1.000, 1.000, 1.000],   # ALS
])

# Gray for Prior SOTA; blue gradient for our progression
colors = ["#999999", "#A6CEE3", "#1F78B4", "#08306B"]

x = np.arange(len(diseases))
width = 0.20

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    offset = (i - 1.5) * width
    bars = ax.bar(
        x + offset, acc[:, i], width,
        label=model, color=colors[i],
        edgecolor="white", linewidth=0.7,
    )
    for bar, val in zip(bars, acc[:, i]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.0015,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=7.5, color="#222222",
        )

ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Disease-specific Accuracy: GeHirNet vs. Prior SOTA",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(diseases, fontsize=10)
ax.set_ylim(0.92, 1.015)
ax.legend(loc="lower right", frameon=True, fontsize=9, ncol=1)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Mark each disease where GeHirNet** beats prior SOTA with a small green check above the group
for j in range(len(diseases)):
    prior = acc[j, 0]
    best = acc[j, -1]
    if best >= prior:
        ax.text(
            x[j], 1.008,
            "▲",
            ha="center", va="bottom",
            fontsize=11, color="#2E8B57", fontweight="bold",
        )
    else:
        ax.text(
            x[j], 1.008,
            "▼",
            ha="center", va="bottom",
            fontsize=11, color="#C44E52", fontweight="bold",
        )

# Legend annotation for the triangle markers
ax.text(
    0.01, 0.97,
    "▲ ours ≥ prior   ▼ ours < prior",
    transform=ax.transAxes,
    fontsize=9, color="#444444",
    va="top",
)

plt.tight_layout()
out = "/Users/fanwufan/Documents/PhD/Research/VoiceKaicheng/demo_code/disease_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
