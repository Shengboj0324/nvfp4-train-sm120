"""Training-step speed of one transformer block (fwd + bwd, BF16 autocast) vs model width.

    python bench/bench_train_step.py [--out bench/results/train_step.json]

Isolates the part NVFP4 accelerates: a pre-LN GPT block (qkv, attention via SDPA, out-proj, 4x MLP)
on 8192 tokens (16 x 512). The LM head / embeddings are excluded on purpose - in a real model they
stay BF16 and their share shrinks as width and depth grow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from bench_gemm import timeit  # noqa: E402
from train_gpt import Block  # noqa: E402

from nvfp4.linear import convert_linears  # noqa: E402


def step_time(d: int, precision: str, b: int = 16, t: int = 512) -> float:
    torch.manual_seed(0)
    blk = Block(d, max(1, d // 128)).cuda()
    if precision == "nvfp4":
        convert_linears(blk)
    x = torch.randn(b, t, d, device="cuda", requires_grad=True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(x)
        y.float().sum().backward()

    return timeit(step, iters=10, warmup=3, trials=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/train_step.json")
    a = ap.parse_args()
    rows = []
    for d in (768, 1024, 2048, 3072, 4096):
        tb, tq = step_time(d, "bf16"), step_time(d, "nvfp4")
        lin_flops = 6 * 8192 * 12 * d * d  # fwd+bwd FLOPs of the four linears
        rows.append({"d": d, "bf16_ms": tb * 1e3, "nvfp4_ms": tq * 1e3, "speedup": tb / tq,
                     "linear_tflops_bf16": lin_flops / tb / 1e12, "linear_tflops_nvfp4": lin_flops / tq / 1e12})
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
