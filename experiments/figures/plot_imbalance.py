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

# (pathology, female, male)
data = [
    ("Health Control",                696, 979),
    ("COVID-19",                      109, 174),
    ("Dysphonia",                     170,  97),
    ("Vocal Cord Paresis",            123,  68),
    ("Parkinson's Disease",            75,  74),
    ("Laryngitis",                     32,  50),
    ("ALS",                             8,   7),
]

pathologies = [d[0] for d in data]
female = np.array([d[1] for d in data])
male   = np.array([d[2] for d in data])
totals = female + male

# Plot top-to-bottom = largest-to-smallest, so we reverse for barh
order = np.arange(len(data))[::-1]
y = np.arange(len(data))

female_color = "#C44E52"
male_color   = "#1F78B4"

fig, ax = plt.subplots(figsize=(11, 6.5))

# Stacked horizontal bars
f_bars = ax.barh(
    y, female[order], color=female_color,
    edgecolor="white", linewidth=0.8, label="Female",
)
m_bars = ax.barh(
    y, male[order], left=female[order], color=male_color,
    edgecolor="white", linewidth=0.8, label="Male",
)

# In-segment counts (only when segment is wide enough)
threshold = 40
for i, idx in enumerate(order):
    if female[idx] >= threshold:
        ax.text(female[idx] / 2, y[i], f"{female[idx]}",
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    if male[idx] >= threshold:
        ax.text(female[idx] + male[idx] / 2, y[i], f"{male[idx]}",
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    # Total at end of bar
    ax.text(
        totals[idx] + 20, y[i],
        f"n = {totals[idx]}  (F {female[idx]} / M {male[idx]})",
        ha="left", va="center", fontsize=9, color="#222222",
    )

ax.set_yticks(y)
ax.set_yticklabels([pathologies[i] for i in order], fontsize=10)
ax.set_xlabel("Number of samples", fontsize=11)
ax.set_title(
    "Class imbalance across pathologies (n = {} total)".format(int(totals.sum())),
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlim(0, totals.max() * 1.32)
ax.legend(loc="lower right", frameon=True, fontsize=10)
ax.xaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Annotate the extreme imbalance ratio
ratio = totals.max() / totals.min()
ax.annotate(
    f"Health : ALS  ≈  {ratio:.0f} : 1",
    xy=(totals.max(), y[list(order).index(0)]),  # near Health bar
    xytext=(totals.max() * 0.55, len(data) - 0.4),
    fontsize=10, color="#C44E52", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2,
                    connectionstyle="arc3,rad=0.2"),
)

plt.tight_layout()
out = "/Users/fanwufan/Documents/PhD/Research/VoiceKaicheng/demo_code/class_imbalance.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
