"""Quantizer throughput. Effective bandwidth counts the input read once plus the NVFP4 output(s)
(the two-pass amax+quantize design actually reads the input twice).

Part 2 times the fused dual quantizer against the separate calls it replaces in NVFP4Linear.

    python bench/bench_quantize.py [--out bench/results/quantize.json]
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from bench_gemm import timeit
from nvfp4 import ops

MODES = {
    "rowwise": {},
    "rowwise_2d": {"block_2d": True},
    "transpose": {"transpose": True},
    "transpose_rht_sr": {"transpose": True, "rht_seed": 1, "stochastic": True, "sr_seed": 3},
    "rowwise_sr": {"stochastic": True, "sr_seed": 3},
}


# (case, shape, dtype): the three operand roles in NVFP4Linear at GPT-2-small / 2048-wide sizes.
FUSED_CASES = [
    ("activation", (8192, 768), torch.bfloat16),
    ("activation", (8192, 3072), torch.bfloat16),
    ("gradient", (8192, 2304), torch.bfloat16),
    ("gradient", (8192, 8192), torch.bfloat16),
    ("weight", (3072, 768), torch.float32),
    ("weight", (8192, 2048), torch.float32),
]


def _fused_case(case, x):
    """(separate-calls fn, fused fn) producing the same operands NVFP4Linear needs."""
    pad = lambda t: F.pad(t, (0, 0, 0, (-t.shape[0]) % 128))  # noqa: E731
    if case == "activation":   # fprop X (rowwise) + X^T with RHT for wgrad
        return (lambda: (ops.quantize(x), ops.quantize(pad(x), transpose=True, rht_seed=1)),
                lambda: ops.quantize_dual(x, rht_seed=1, col_pad=128))
    if case == "gradient":     # dY rowwise SR (dgrad) + dY^T RHT + SR (wgrad)
        return (lambda: (ops.quantize(x, stochastic=True, sr_seed=2),
                         ops.quantize(pad(x), transpose=True, stochastic=True, sr_seed=3, rht_seed=1)),
                lambda: ops.quantize_dual(x, row_sr_seed=2, col_sr_seed=3, rht_seed=1, col_pad=128))
    return (lambda: (ops.quantize(x, block_2d=True), ops.quantize(x, transpose=True, block_2d=True)),
            lambda: ops.quantize_dual(x, block_2d=True))


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
    fused = []
    for case, shape, dtype in FUSED_CASES:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        sep, dual = _fused_case(case, x)
        ts, td = timeit(sep, 30), timeit(dual, 30)
        nbytes = x.numel() * x.element_size() + 2 * (x.numel() // 2 + x.numel() // 16)
        fused.append({"case": case, "shape": list(shape), "dtype": str(dtype).split(".")[-1],
                      "separate_us": ts * 1e6, "fused_us": td * 1e6, "speedup": ts / td,
                      "fused_eff_GBps": nbytes / td / 1e9})
        print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in fused[-1].items()}),
              flush=True)
    rows = {"single": rows, "fused": fused}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
