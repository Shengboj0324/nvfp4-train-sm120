"""Quantizer throughput. Effective bandwidth counts the bf16 input read once plus the NVFP4 output
(the two-pass amax+quantize design actually reads the input twice).

    python bench/bench_quantize.py [--out bench/results/quantize.json]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from bench_gemm import timeit
from nvfp4 import ops

MODES = {
    "rowwise": {},
    "rowwise_2d": {"block_2d": True},
    "transpose": {"transpose": True},
    "transpose_rht_sr": {"transpose": True, "rht_seed": 1, "stochastic": True, "sr_seed": 3},
    "rowwise_sr": {"stochastic": True, "sr_seed": 3},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/quantize.json")
    args = ap.parse_args()
    rows = []
    for shape in [(8192, 2048), (8192, 8192), (768, 8192)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        nbytes = x.numel() * 2 + x.numel() // 2 + x.numel() // 16
        for name, kw in MODES.items():
            t = timeit(lambda: ops.quantize(x, **kw), 30)
            rows.append({"shape": list(shape), "mode": name, "us": t * 1e6, "eff_GBps": nbytes / t / 1e9})
            print(json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()}),
                  flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
