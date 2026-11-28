"""v0.19 release figure: loss gap, where it comes from, forward formats, throughput."""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1]
SURF, T1, T2, T3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a9a8a3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": T2,
    "xtick.color": T2, "ytick.color": T1, "axes.facecolor": SURF, "figure.facecolor": SURF,
})

fig = plt.figure(figsize=(14, 9.4), dpi=160)
gs = fig.add_gridspec(2, 2, left=0.205, right=0.965, top=0.845, bottom=0.115, hspace=0.5, wspace=0.6,
                      width_ratios=[1.15, 1])


def style(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=T1, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=T2, va="bottom")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bars(ax, labels, vals, colors, fmt, h=0.56):
    y = list(range(len(labels)))[::-1]
    for yi, v, c in zip(y, vals, colors):
        ax.add_patch(FancyBboxPatch((min(0, v), yi - h / 2), abs(v), h, boxstyle="round,pad=0,rounding_size=0.04",
                                    mutation_aspect=0.05, fc=c, ec=SURF, lw=2))
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    return y


# A: loss gap to BF16, three seeds
ax = fig.add_subplot(gs[0, 0])
style(ax, "Validation-loss gap to BF16 (lower is better)",
      "GPT-124M on TinyStories, 3000 steps, 3 seeds; bars = mean, dots = seeds")
rows = [("NVFP4, default recipe", [1.15, 1.42, 1.44], GRAY),
        ("+ BF16 tail, last 10% (v0.11 runs)", [0.41, 0.68, 0.77], ORANGE),
        ("+ two-term activations (new)", [0.70, 0.79, 0.62], BLUE),
        ("+ two-term + BF16 tail (new)", [0.32, 0.60, 0.48], AQUA)]
means = [sum(r[1]) / 3 for r in rows]
y = bars(ax, [r[0] for r in rows], means, [r[2] for r in rows], None)
for yi, (_, s, _), m in zip(y, rows, means):
    ax.scatter(s, [yi] * 3, s=26, color=SURF, edgecolor=T1, lw=1.1, zorder=5)
    ax.text(max(s) + 0.05, yi, f"+{m:.2f}%", va="center", fontsize=11.5, fontweight="bold", color=T1)
ax.set_xlim(0, 1.75)
ax.set_xticks([0, 0.4, 0.8, 1.2, 1.6])
ax.xaxis.set_major_formatter(lambda v, _: f"+{v:.1f}%" if v else "BF16")
ax.axvline(0, color=T2, lw=1.2)

# B: where the gap comes from
ax = fig.add_subplot(gs[0, 1])
style(ax, "Where the gap comes from", "loss removed by running one part in BF16 (seed 0)")
lab = ["forward GEMM", "fwd: activations only", "fwd: weights only", "weight-gradient GEMM", "input-gradient GEMM",
       "no stochastic rounding"]
val = [0.90, 0.66, 0.13, 0.14, 0.12, 0.02]
col = [BLUE, BLUE, GRAY, GRAY, GRAY, GRAY]
y = bars(ax, lab, val, col, None, h=0.6)
for yi, v in zip(y, val):
    ax.text(v + 0.02, yi, f"−{v:.2f}%", va="center", fontsize=10.5, color=T1)
ax.set_xlim(0, 1.15)
ax.xaxis.set_major_formatter(lambda v, _: f"−{v:.1f}%" if v else "0")
for t, w in zip(ax.get_yticklabels(), ["bold", "normal", "normal", "bold", "bold", "bold"]):
    t.set_fontweight(w)

# C: forward activation formats
ax = fig.add_subplot(gs[1, 0])
style(ax, "Fixing the activations: format tried in the forward GEMM",
      "change in val loss vs plain NVFP4 activations (emulated, 2 seeds; left is better)")
lab = ["Hadamard rotation, 16-pt", "Hadamard rotation, 128-pt", "MXFP8 (8-bit) activations",
       "two NVFP4 terms (shipped)"]
val = [0.15, 0.14, -0.65, -0.66]
col = [GRAY, GRAY, GRAY, BLUE]
y = bars(ax, lab, val, col, None)
for yi, v in zip(y, val):
    ax.text(v + (0.03 if v > 0 else -0.03), yi, f"{v:+.2f}%".replace("-", "−"), va="center",
            ha="left" if v > 0 else "right", fontsize=10.5, color=T1)
ax.axvline(0, color=T2, lw=1.2)
ax.set_xlim(-0.9, 0.4)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.1f}%".replace("-", "−") if v else "0")

# D: throughput
ax = fig.add_subplot(gs[1, 1])
style(ax, "Training throughput", "GPT-124M tokens/s, RTX 5090 Laptop (× = vs BF16)")
lab = ["BF16", "NVFP4, default", "NVFP4, two-term"]
val = [63.1, 81.9, 79.2]
y = bars(ax, lab, val, [GRAY, GRAY, BLUE], None)
for yi, v, note in zip(y, val, ["", "1.30×", "1.25×  (−3.3%)"]):
    ax.text(v + 1.5, yi, f"{v:.1f}k" + (f"  ·  {note}" if note else ""), va="center", fontsize=10.5, color=T1)
ax.set_xlim(0, 135)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}k")

fig.text(0.03, 0.955, "nvfp4-train-sm120 v0.19: 4-bit training on RTX 50 GPUs, half the loss gap",
         fontsize=19, fontweight="bold", color=T1)
fig.text(0.03, 0.915, "Most of NVFP4's loss gap is rounding of forward activations. Storing each activation as two "
         "FP4 terms (the value + its own rounding error) halves it for 3% of throughput.", fontsize=11.5, color=T2)
fig.text(0.03, 0.03, "Setup: GPT-124M, TinyStories, embeddings, LM head and last 2 of 12 blocks in BF16; "
         "RTX 5090 Laptop, CUDA 13.0, PyTorch 2.14. Data: docs/RESULTS.md §4.3", fontsize=8.5, color=T3)
fig.text(0.97, 0.03, "github.com/Shengboj0324/nvfp4-train-sm120", fontsize=10, color=T2, ha="right",
         fontweight="bold")
fig.savefig(OUT, facecolor=SURF)
print("saved", OUT)
