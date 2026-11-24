"""Cost of the quantizer recipe options on the dual quantizer (bf16, row + col, col RHT, col_pad 128):
round-to-nearest vs stochastic rounding (8-bit integer-domain rounder) on gradient-shaped inputs, Four Over
Six block scales on activation-shaped inputs, and MS-EDEN (64-point RHT, EDEN-corrected scales) on both.

    python bench/bench_sr_cost.py [--out bench/results/quantize_recipes.json]
"""
import argparse
import json
import os

import torch

from bench_gemm import timeit
from nvfp4 import ops


def t_us(fn):
    return timeit(fn, 50) * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/quantize_recipes.json")
    a = ap.parse_args()
    rows = []
    for shape in [(8192, 2304), (8192, 8192), (16384, 3072)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        act = lambda: ops.quantize_dual(x, rht_seed=1, col_pad=128)  # noqa: E731
        grad = lambda: ops.quantize_dual(x, row_sr_seed=2, col_sr_seed=3, rht_seed=1, col_pad=128)  # noqa: E731
        r = {"shape": list(shape), "activation_rn_us": t_us(act), "gradient_sr_us": t_us(grad)}
        with ops.four_six_scales(True):
            r["activation_four_six_us"] = t_us(act)
        ops.set_ms_eden(True)
        r["activation_ms_eden_us"] = t_us(act)
        r["gradient_ms_eden_us"] = t_us(grad)
        ops.set_ms_eden(False)
        rows.append(r)
        print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
