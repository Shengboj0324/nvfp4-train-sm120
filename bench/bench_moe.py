"""Training-step time (forward + backward) of an nvfp4.moe.MoE layer, BF16 experts vs NVFP4 experts (fused
MLPs, fused wgrad accumulation), for a few expert configurations.

    python bench/bench_moe.py [--out bench/results/moe.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_gemm import timeit  # noqa: E402

from nvfp4 import convert_linears  # noqa: E402
from nvfp4.moe import MoE  # noqa: E402

CONFIGS = [  # d, d_ff, experts, top_k, activation, tokens
    (1024, 2816, 8, 2, "swiglu", 16384),
    (2048, 5632, 8, 2, "swiglu", 16384),
    (2048, 1408, 32, 4, "swiglu", 16384),
    (768, 1536, 8, 2, "gelu_tanh", 16384),
]


def step_ms(moe, x):
    g = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.backward(g)
    step()
    return timeit(step, iters=5, warmup=2, trials=5) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/moe.json")
    a = ap.parse_args()
    rows = []
    for d, d_ff, E, k, act, T in CONFIGS:
        torch.manual_seed(0)
        bf16 = MoE(d, d_ff, E, top_k=k, activation=act).cuda()
        fp4 = copy.deepcopy(bf16)
        convert_linears(fp4, fuse_wgrad_accumulation=True)
        x = torch.randn(T, d, device="cuda", requires_grad=True)
        tb, tf = step_ms(bf16, x), step_ms(fp4, x)
        r = {"d": d, "d_ff": d_ff, "experts": E, "top_k": k, "activation": act, "tokens": T,
             "bf16_ms": tb, "nvfp4_ms": tf, "speedup": tb / tf}
        rows.append(r)
        print(json.dumps({k_: round(v, 3) if isinstance(v, float) else v for k_, v in r.items()}), flush=True)
        del bf16, fp4, x
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
