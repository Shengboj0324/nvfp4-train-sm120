"""Summarise and plot the runs in a results directory (default experiments/results).

    python experiments/plot.py [DIR]   -> DIR/loss.png, DIR/summary.json (and prints a table)

The run named "bf16" in DIR is the baseline for the gap panel; energy comes from DIR/train_<run>.csv.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
LABELS = {"bf16": "BF16 baseline", "nvfp4": "NVFP4 recipe", "nvfp4_all": "NVFP4 (all blocks)",
          "nvfp4_no_rht": "NVFP4 without wgrad RHT"}


def label(name, meta):
    """Legend label; the NVFP4 recipe run says how many final blocks it kept in BF16 (from its config)."""
    lab = LABELS.get(name, name)
    hp = meta["config"].get("hp_last", 0) if meta else 0
    if name == "nvfp4" and hp:
        lab += f" (last {hp} block{'s' if hp > 1 else ''} BF16)"
    return lab


def load(path):
    meta, train, val, done = None, [], [], None
    for line in open(path):
        r = json.loads(line)
        if "meta" in r:
            meta = r["meta"]
        elif "val_loss" in r:
            val.append(r)
        elif "loss" in r:
            train.append(r)
        elif "done" in r:
            done = r
    return meta, train, val, done


def energy(name):
    """Integrate GPU power from the guard log (2 s samples) -> joules, mean power, mean clock."""
    p = os.path.join(R, f"train_{name}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    pw = [float(r["gpu_power_w"]) for r in rows if r["gpu_power_w"] not in ("", "nan")]
    clk = [float(r["sm_clock_mhz"]) for r in rows if r["sm_clock_mhz"] not in ("", "nan")]
    temp = [float(r["gpu_temp_c"]) for r in rows if r["gpu_temp_c"] not in ("", "nan")]
    dt = float(rows[1]["t_s"]) - float(rows[0]["t_s"]) if len(rows) > 1 else 2.0
    return {"joules": sum(pw) * dt, "mean_w": sum(pw) / len(pw), "mean_sm_mhz": sum(clk) / len(clk),
            "max_temp_c": max(temp)}


def main():
    runs = {}
    for path in sorted(glob.glob(os.path.join(R, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        runs[name] = load(path)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    summary = {}
    order = [n for n in LABELS if n in runs] + [n for n in runs if n not in LABELS]
    for name in order:
        meta, train, val, done = runs[name]
        lab = label(name, meta)
        ax[0].plot([r["step"] for r in train], [r["loss"] for r in train], label=lab, lw=1)
        ax[1].plot([r["step"] for r in val], [r["val_loss"] for r in val], marker="o", ms=3, label=lab)
        tps = [r["tok_per_s"] for r in train if r.get("tok_per_s")]
        steady = sorted(tps[len(tps) // 5:])  # skip warm-up
        e = energy(name)
        tokens = meta["config"]["steps"] * meta["config"]["batch"] * meta["config"]["ctx"]
        summary[name] = {"final_val_loss": val[-1]["val_loss"] if val else None,
                         "median_tok_per_s": steady[len(steady) // 2] if steady else None,
                         "wall_s": done["total_s"] if done else None,
                         "max_mem_gb": max(r["max_mem_gb"] for r in train),
                         "nvfp4_linears": meta["nvfp4_linears"], "completed": done is not None}
        if e:
            summary[name].update({"gpu_joules": e["joules"], "mean_gpu_w": e["mean_w"],
                                  "mean_sm_mhz": e["mean_sm_mhz"], "max_gpu_temp_c": e["max_temp_c"],
                                  "joules_per_1k_tokens": e["joules"] / tokens * 1e3})
    m0 = runs[order[0]][0]
    c0 = m0["config"]
    model = f"GPT {m0['params'] / 1e6:.0f}M, d={c0['d']}, {c0['layers']} layers"
    ax[0].set(xlabel="step", ylabel="train loss", ylim=(1.0, 4.0), title=f"Train loss ({model}, TinyStories)")
    ax[1].set(xlabel="step", ylabel="val loss", ylim=(1.0, 4.0), title="Validation loss")
    base = {r["step"]: r["val_loss"] for r in runs["bf16"][2]} if "bf16" in runs else {}
    for name in order:
        if name == "bf16" or not base:
            continue
        val = runs[name][2]
        steps = [r["step"] for r in val if r["step"] in base and r["step"] > 0]
        ax[2].plot(steps, [next(r["val_loss"] for r in val if r["step"] == s) - base[s] for s in steps],
                   marker="o", ms=3, label=label(name, runs[name][0]))
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set(xlabel="step", ylabel="val loss - BF16 val loss", title="Gap to BF16 (same data & init)")
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "loss.png"), dpi=130)
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
    for k, v in summary.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
