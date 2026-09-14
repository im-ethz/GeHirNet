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

# Order matches the table; reversed for plotting so Health appears at top
diseases = [
    "Health", "COVID-19", "Parkinson's", "Dysphonia",
    "Vocal Cord Paresis", "Laryngitis", "ALS",
]
female_mean = np.array([-9.63, -22.38, -2.80, -2.69, -2.56, -3.05, -5.23])
female_std  = np.array([ 9.49,   7.22,  3.74,  2.49,  3.42,  3.04,  7.63])
male_mean   = np.array([-14.55, -20.86, -2.99, -2.96, -1.64, -2.43, -9.62])
male_std    = np.array([  9.16,   7.50,  3.14,  2.33,  4.02,  3.42,  4.55])

delta   = np.array([ 4.921, -1.523,  0.192,  0.267, -0.919, -0.618,  4.393])
ci_low  = np.array([ 4.509, -2.218, -0.218, -0.139, -1.676, -1.623,  2.295])
ci_high = np.array([ 5.335, -0.845,  0.602,  0.674, -0.162,  0.387,  6.587])
sig     = np.array([True,    True,   False,  False,  True,   False,  True])

# Reverse so the first row of the table sits on top
order = np.arange(len(diseases))[::-1]
y = np.arange(len(diseases))

female_color = "#C44E52"   # coral / red
male_color   = "#1F78B4"   # blue
pos_color    = "#C44E52"   # F > M significant
neg_color    = "#1F78B4"   # M > F significant
ns_color     = "#999999"   # not significant

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(14, 7),
    gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.35},
)

# ----- LEFT: Female vs Male means with std error bars -----
bar_h = 0.38
for plot_i, idx in enumerate(order):
    axL.barh(
        y[plot_i] + bar_h / 2, female_mean[idx], height=bar_h,
        xerr=female_std[idx],
        color=female_color, edgecolor="white", linewidth=0.6,
        error_kw={"elinewidth": 1.0, "ecolor": "#444444", "capsize": 2.5},
        label="Female" if plot_i == 0 else None,
    )
    axL.barh(
        y[plot_i] - bar_h / 2, male_mean[idx], height=bar_h,
        xerr=male_std[idx],
        color=male_color, edgecolor="white", linewidth=0.6,
        error_kw={"elinewidth": 1.0, "ecolor": "#444444", "capsize": 2.5},
        label="Male" if plot_i == 0 else None,
    )

axL.axvline(0, color="#333333", linewidth=0.8)
axL.set_yticks(y)
axL.set_yticklabels([diseases[i] for i in order], fontsize=10)
axL.set_xlabel("Mel-spectrogram power (dB)", fontsize=11)
axL.set_title("Female vs. Male — mean ± std", fontsize=12, fontweight="bold", pad=10)
axL.legend(loc="lower left", frameon=True, fontsize=9)
axL.xaxis.grid(True, linestyle="--", alpha=0.4)
axL.set_axisbelow(True)
for spine in ["top", "right"]:
    axL.spines[spine].set_visible(False)

# ----- RIGHT: Forest plot of Δ (F-M) with 95% CI -----
for plot_i, idx in enumerate(order):
    if not sig[idx]:
        color = ns_color
    else:
        color = pos_color if delta[idx] > 0 else neg_color
    # CI bar
    axR.plot(
        [ci_low[idx], ci_high[idx]], [y[plot_i], y[plot_i]],
        color=color, linewidth=2.2, solid_capstyle="round",
    )
    # CI caps
    for xv in (ci_low[idx], ci_high[idx]):
        axR.plot([xv, xv], [y[plot_i] - 0.15, y[plot_i] + 0.15],
                 color=color, linewidth=1.5)
    # Point estimate
    axR.scatter(delta[idx], y[plot_i], color=color, s=55,
                edgecolor="white", linewidth=0.8, zorder=5)
    # Value label + significance star
    star = "*" if sig[idx] else ""
    axR.text(
        ci_high[idx] + 0.25, y[plot_i],
        f"  {delta[idx]:+.3f}{star}",
        va="center", ha="left", fontsize=9,
        color=color, fontweight="bold" if sig[idx] else "normal",
    )

axR.axvline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
axR.set_yticks(y)
axR.set_yticklabels([])  # share with left panel via disease alignment
axR.set_xlabel("Δ (Female − Male), dB    [95% CI]", fontsize=11)
axR.set_title("Effect size with 95% CI", fontsize=12, fontweight="bold", pad=10)
axR.xaxis.grid(True, linestyle="--", alpha=0.4)
axR.set_axisbelow(True)
for spine in ["top", "right", "left"]:
    axR.spines[spine].set_visible(False)
axR.tick_params(left=False)

# Reference annotations on the right panel — placed inside the axes, near top
axR.text(0.02, 0.985, "← Male higher", transform=axR.transAxes,
         fontsize=9, color=neg_color, ha="left", va="top")
axR.text(0.98, 0.985, "Female higher →", transform=axR.transAxes,
         fontsize=9, color=pos_color, ha="right", va="top")

# Expand right-panel xlim a bit so labels fit
xlo = min(ci_low.min(), -3.0) - 0.5
xhi = max(ci_high.max(), 3.0) + 1.8
axR.set_xlim(xlo, xhi)

fig.suptitle(
    "Sex differences in Mel-spectrogram power across diseases",
    fontsize=14, fontweight="bold", y=0.995,
)

# Footnote
fig.text(0.5, 0.01,
         "* p < 0.05.  Bars on the left show mean ± 1 std; right panel shows Δ(F−M) with 95% CI.",
         ha="center", fontsize=9, color="#444444")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = "/Users/fanwufan/Documents/PhD/Research/VoiceKaicheng/demo_code/sex_difference.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")
