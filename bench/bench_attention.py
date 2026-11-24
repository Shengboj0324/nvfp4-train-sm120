"""Speed of nvfp4.attention (INT8 scores + MXFP8 probabilities) vs PyTorch SDPA (flash attention, BF16):
forward, and forward + backward through autograd (quantization included), causal and full.

    python bench/bench_attention.py [--out bench/results/attention.json]
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

from nvfp4.attention import attention  # noqa: E402


def run(b, h, t, d, causal):
    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    do = torch.randn(b, h, t, d, device="cuda", dtype=torch.bfloat16)
    flops = 4 * b * h * t * t * d * (0.5 if causal else 1.0)
    r = {"b": b, "h": h, "t": t, "d": d, "causal": causal}
    for name, fn in (("sdpa", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)),
                     ("int8_fp8", lambda: attention(q, k, v, causal=causal))):
        with torch.no_grad():
            tf = timeit(fn, iters=10, warmup=3, trials=5)

        def fb():
            fn().backward(do)
        tfb = timeit(fb, iters=10, warmup=3, trials=5)
        r[f"{name}_fwd_ms"], r[f"{name}_fwdbwd_ms"] = tf * 1e3, tfb * 1e3
        r[f"{name}_fwd_tflops"], r[f"{name}_fwdbwd_tflops"] = flops / tf / 1e12, 3.5 * flops / tfb / 1e12
    r["speedup_fwd"] = r["sdpa_fwd_ms"] / r["int8_fp8_fwd_ms"]
    r["speedup_fwdbwd"] = r["sdpa_fwdbwd_ms"] / r["int8_fp8_fwdbwd_ms"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/attention.json")
    ap.add_argument("--tokens", type=int, default=16384)
    a = ap.parse_args()
    rows = []
    for d, h in ((128, 16), (64, 16)):
        for t in (512, 1024, 2048, 4096, 8192):
            for causal in (True, False):
                r = run(a.tokens // t, h, t, d, causal)
                rows.append(r)
                print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
                torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
