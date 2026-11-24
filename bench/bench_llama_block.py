"""Training-step speed of one Llama-style block (RMSNorm, qkv, causal SDPA, o_proj, RMSNorm, SwiGLU MLP)
vs model width: BF16, NVFP4 linears, and NVFP4 with RMSNorm / SwiGLU fused into the quantizers
(nvfp4.fused.rmsnorm_linear / rmsnorm_swiglu_mlp, fused wgrad accumulation), with and without the
weight-quantization cache (a micro-batch after the first under gradient accumulation); "_res" also
fuses the residual adds and the q / k / v hand-off (chunks=3, return_residual, residual=True: v0.8).

    python bench/bench_llama_block.py [--out bench/results/llama_block.json]

8192 tokens (16 x 512) per step, BF16 autocast, gradients accumulate across steps (no zero_grad).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from bench_gemm import timeit  # noqa: E402

from nvfp4.attention import attention as attention_8bit  # noqa: E402
from nvfp4.fused import rmsnorm_linear, rmsnorm_swiglu_mlp  # noqa: E402
from nvfp4.linear import convert_linears  # noqa: E402


class LlamaBlock(torch.nn.Module):
    def __init__(self, d: int, heads: int, hidden: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = torch.nn.RMSNorm(d, eps=1e-5), torch.nn.RMSNorm(d, eps=1e-5)
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.o_proj = torch.nn.Linear(d, d, bias=False)
        self.gate = torch.nn.Linear(d, hidden, bias=False)
        self.up = torch.nn.Linear(d, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, d, bias=False)
        self.fused = False
        self.fused_res = False
        self.attn8 = False  # nvfp4.attention instead of BF16 SDPA (bench_long_context.py)

    def sdpa(self, q, k, v):
        if self.attn8:
            return attention_8bit(q, k, v, causal=True)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def attn(self, qkv):
        b, t, _ = qkv.shape
        q, k, v = qkv.view(b, t, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        return self.sdpa(q, k, v).transpose(1, 2).reshape(b, t, -1)

    def attn3(self, q, k, v):
        b, t, _ = q.shape
        q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2) for z in (q, k, v))
        return self.sdpa(q, k, v).transpose(1, 2).reshape(b, t, -1)

    def forward(self, x):
        if self.fused_res:
            q, k, v, res = rmsnorm_linear(x, self.norm1, self.qkv, chunks=3, return_residual=True)
            x = self.o_proj(self.attn3(q, k, v), residual=res)
            return rmsnorm_swiglu_mlp(x, self.norm2, self.gate, self.up, self.down, residual=True)
        if self.fused:
            x = x + self.o_proj(self.attn(rmsnorm_linear(x, self.norm1, self.qkv)))
            return x + rmsnorm_swiglu_mlp(x, self.norm2, self.gate, self.up, self.down)
        x = x + self.o_proj(self.attn(self.qkv(self.norm1(x))))
        n = self.norm2(x)
        return x + self.down(F.silu(self.gate(n)) * self.up(n))


def hidden_size(d: int) -> int:
    return (8 * d // 3 + 255) // 256 * 256  # Llama's ~8/3 d, rounded to a multiple of 256


def step_time(d: int, variant: str, b: int = 16, t: int = 512) -> tuple[float, float]:
    torch.manual_seed(0)
    blk = LlamaBlock(d, d // 128, hidden_size(d)).cuda()
    if variant != "bf16":
        convert_linears(blk, fuse_wgrad_accumulation=variant != "nvfp4",
                        cache_weight_quantization=variant.endswith("_cached"))
        blk.fused = variant.startswith("nvfp4_fused")
        blk.fused_res = variant.startswith("nvfp4_fused_res")
    x = torch.randn(b, t, d, device="cuda", requires_grad=True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(x)
        y.float().sum().backward()

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    sec = timeit(step, iters=10, warmup=3, trials=5)
    return sec, torch.cuda.max_memory_allocated() / 2**20


VARIANTS = ("bf16", "nvfp4", "nvfp4_fused", "nvfp4_fused_cached", "nvfp4_fused_res", "nvfp4_fused_res_cached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/llama_block.json")
    ap.add_argument("--widths", default="1024,2048,3072,4096")
    a = ap.parse_args()
    rows = []
    for d in (int(w) for w in a.widths.split(",")):
        r = {"d": d, "hidden": hidden_size(d)}
        # two passes in opposite orders, best of both: the power-limited laptop GPU runs hotter (and slower)
        # the longer it has been busy, which would otherwise favour whichever variant runs first
        for order in (VARIANTS, VARIANTS[::-1]):
            for v in order:
                sec, mib = step_time(d, v)
                r[f"{v}_ms"] = min(r.get(f"{v}_ms", float("inf")), sec * 1e3)
                r.setdefault(f"peak_mib_{v}", mib)
                torch.cuda.empty_cache()
        for v in VARIANTS[1:]:
            r[f"speedup_{v}"] = r["bf16_ms"] / r[f"{v}_ms"]
        rows.append(r)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
