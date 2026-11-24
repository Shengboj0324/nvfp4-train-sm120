"""Training-step speed of one Llama-style block (d = 2048, 16 heads of 128, SwiGLU) vs sequence length at
16384 tokens per step: BF16; NVFP4 linears with every fusion and the weight cache, attention in BF16 SDPA
(flash attention); and the same with nvfp4.attention (INT8 scores, MXFP8 probabilities), with its default
backward (exact flash backward of the 8-bit forward) and with backward="8bit". Attention's share
of the step grows with the sequence length.

    python bench/bench_long_context.py [--out bench/results/long_context.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_gemm import timeit  # noqa: E402
from bench_llama_block import LlamaBlock, hidden_size  # noqa: E402

from nvfp4.linear import convert_linears  # noqa: E402

VARIANTS = ("bf16", "nvfp4", "nvfp4_attn8", "nvfp4_attn8_8bitbwd")


def step_ms(d: int, t: int, variant: str, tokens: int) -> float:
    torch.manual_seed(0)
    blk = LlamaBlock(d, d // 128, hidden_size(d)).cuda()
    if variant != "bf16":
        convert_linears(blk, fuse_wgrad_accumulation=True, cache_weight_quantization=True)
        blk.fused = blk.fused_res = True
        blk.attn8 = variant.startswith("nvfp4_attn8")
        blk.attn_backward = "8bit" if variant.endswith("8bitbwd") else "bf16"
    x = torch.randn(tokens // t, t, d, device="cuda", requires_grad=True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(x)
        y.float().sum().backward()

    step()
    return timeit(step, iters=5, warmup=2, trials=5) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/long_context.json")
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--lengths", default="512,1024,2048,4096,8192,16384")
    a = ap.parse_args()
    rows = []
    for t in (int(x) for x in a.lengths.split(",")):
        r = {"d": a.d, "t": t, "tokens": a.tokens}
        for order in (VARIANTS, VARIANTS[::-1]):  # best of two passes in opposite orders (laptop thermals)
            for v in order:
                r[f"{v}_ms"] = min(r.get(f"{v}_ms", float("inf")), step_ms(a.d, t, v, a.tokens))
                torch.cuda.empty_cache()
        r["speedup_nvfp4"] = r["bf16_ms"] / r["nvfp4_ms"]
        r["speedup_nvfp4_attn8"] = r["bf16_ms"] / r["nvfp4_attn8_ms"]
        r["attn8_gain"] = r["nvfp4_ms"] / r["nvfp4_attn8_ms"]
        r["attn8_8bitbwd_gain"] = r["nvfp4_ms"] / r["nvfp4_attn8_8bitbwd_ms"]
        rows.append(r)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
