"""Summarise experiments/run_ablation.sh runs: final validation loss per recipe and seed, the gap to the
BF16 run of the same seed (experiments/results/v0.5_seeds/bf16_s*.jsonl; the BF16 path is unchanged), and the
paired difference to the baseline recipe of the same seed (same data order and init, so most seed noise
cancels). Runs without a {"done": true} record (killed or still running) are skipped.

    python experiments/ablation_summary.py [DIR] [--baseline NAME]
"""
import argparse
import glob
import json
import os
import re
import statistics as st

HERE = os.path.dirname(__file__)


def final(path):
    v, done = None, False
    for line in open(path):
        r = json.loads(line)
        if "val_loss" in r:
            v = r["val_loss"]
        done = done or bool(r.get("done"))
    return v if done else None


def mean_sd(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=os.path.join(HERE, "results", "v0.10_ablate"))
    ap.add_argument("--baseline", default="baseline_sr8")
    a = ap.parse_args()
    bf16 = {int(re.search(r"_s(\d+)", p).group(1)): final(p)
            for p in glob.glob(os.path.join(HERE, "results", "v0.5_seeds", "bf16_s*.jsonl"))}
    runs = {}
    for p in sorted(glob.glob(os.path.join(a.dir, "*_s*.jsonl"))):
        m = re.match(r"(.+)_s(\d+)\.jsonl$", os.path.basename(p))
        v = final(p)
        if m and v is not None:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = v
    base = runs.get(a.baseline, {})
    print(f"{'recipe':22s} {'seeds':>5s} {'val loss':>9s} {'vs BF16':>8s} {'sd':>5s} "
          f"{'vs ' + a.baseline:>17s} {'sd':>5s}  per seed (vs BF16 | vs baseline)")
    for name, seeds in sorted(runs.items()):
        gaps = [100 * (v / bf16[s] - 1) for s, v in sorted(seeds.items()) if bf16.get(s)]
        pair = [100 * (v / base[s] - 1) for s, v in sorted(seeds.items()) if s in base]
        g, gs = mean_sd(gaps)
        line = f"{name:22s} {len(seeds):5d} {st.mean(seeds.values()):9.4f} {g:+7.2f}% {gs:5.2f} "
        if pair and name != a.baseline:
            p, ps = mean_sd(pair)
            line += f"{p:+16.2f}% {ps:5.2f}  "
        else:
            line += f"{'':17s} {'':5s}  "
        line += " ".join(f"{x:+.2f}" for x in gaps)
        if pair and name != a.baseline:
            line += " | " + " ".join(f"{x:+.2f}" for x in pair)
        print(line)


if __name__ == "__main__":
    main()
