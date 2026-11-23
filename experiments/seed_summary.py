"""Summarise a multi-seed BF16 vs NVFP4 study (experiments/run_v05_seeds.sh).

    python experiments/seed_summary.py [DIR]   -> DIR/seeds.png, DIR/summary.json (and prints a table)

Runs are DIR/{bf16,nvfp4}_s<seed>.jsonl. Seeds change the initialisation and the data order, so the
per-seed gap (NVFP4 - BF16 with the same seed) is the paired comparison.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results", "v0.5_seeds")


def load(path):
    val, tps = [], []
    for line in open(path):
        r = json.loads(line)
        if "val_loss" in r:
            val.append((r["step"], r["val_loss"]))
        elif r.get("tok_per_s"):
            tps.append(r["tok_per_s"])
    return val, tps


runs = {}
for p in sorted(glob.glob(os.path.join(D, "*_s*.jsonl"))):
    m = re.match(r"(bf16|nvfp4)_s(\d+)\.jsonl$", os.path.basename(p))
    if m:
        runs[(m.group(1), int(m.group(2)))] = load(p)

seeds = sorted({s for (_, s) in runs if ("bf16", s) in runs and ("nvfp4", s) in runs})
rows = []
for s in seeds:
    (vb, tb), (vq, tq) = runs[("bf16", s)], runs[("nvfp4", s)]
    fb, fq = vb[-1][1], vq[-1][1]
    tail = lambda t: st.median(t[len(t) // 2:])  # noqa: E731  steady-state throughput
    rows.append({"seed": s, "bf16_val": fb, "nvfp4_val": fq, "gap": fq - fb, "rel_gap_pct": (fq / fb - 1) * 100,
                 "bf16_tok_s": tail(tb), "nvfp4_tok_s": tail(tq)})


def ms(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


summary = {"seeds": seeds, "runs": rows}
for k in ("bf16_val", "nvfp4_val", "gap", "rel_gap_pct", "bf16_tok_s", "nvfp4_tok_s"):
    mean, sd = ms([r[k] for r in rows])
    summary[k] = {"mean": mean, "std": sd}
json.dump(summary, open(os.path.join(D, "summary.json"), "w"), indent=1)

print(f"{'seed':>4} {'BF16 val':>9} {'NVFP4 val':>10} {'gap':>8} {'gap %':>7} {'BF16 tok/s':>11} {'NVFP4 tok/s':>12}")
for r in rows:
    print(f"{r['seed']:>4} {r['bf16_val']:9.4f} {r['nvfp4_val']:10.4f} {r['gap']:+8.4f} {r['rel_gap_pct']:+6.2f}% "
          f"{r['bf16_tok_s']:11.0f} {r['nvfp4_tok_s']:12.0f}")
f = summary
print(f"mean {f['bf16_val']['mean']:9.4f} {f['nvfp4_val']['mean']:10.4f} {f['gap']['mean']:+8.4f} "
      f"{f['rel_gap_pct']['mean']:+6.2f}%  (std: BF16 {f['bf16_val']['std']:.4f}, NVFP4 {f['nvfp4_val']['std']:.4f}, "
      f"gap {f['gap']['std']:.4f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for (prec, s), (val, _) in sorted(runs.items()):
    steps, losses = zip(*val)
    ax[0].plot(steps, losses, color="tab:blue" if prec == "bf16" else "tab:orange", alpha=0.8,
               label=f"{prec.upper()} seed {s}")
for s in seeds:
    vb, vq = dict(runs[("bf16", s)][0]), dict(runs[("nvfp4", s)][0])
    common = sorted(set(vb) & set(vq))[1:]
    ax[1].plot(common, [vq[c] - vb[c] for c in common], label=f"seed {s}")
ax[0].set(xlabel="step", ylabel="validation loss", title="GPT-124M: BF16 vs NVFP4, three seeds", ylim=(1.4, 2.6))
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set(xlabel="step", ylabel="NVFP4 - BF16 validation loss", title="paired gap (same seed)")
ax[0].legend(fontsize=7, ncol=2)
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(D, "seeds.png"), dpi=130)
