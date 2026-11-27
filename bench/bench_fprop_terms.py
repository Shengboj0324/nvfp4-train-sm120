"""Cost of two-term forward activations (ops.set_fprop_terms(2)): the dual quantizer with row_terms=2, the forward
GEMM at twice the depth against [W | W], and one NVFP4Linear forward + backward, each against the one-term path.
GPT-124M shapes (8 x 1024 tokens). Writes bench/results/fprop_terms.json."""
import json
import os

import torch

from nvfp4 import ops
from nvfp4.linear import NVFP4Linear

M = 8192
REPS = 50


def timed(fn, reps=REPS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / reps  # us


def main():
    torch.manual_seed(0)
    out = {"device": torch.cuda.get_device_name(), "tokens": M, "quantize": [], "gemm": [], "linear": []}
    for C in (768, 3072):
        x = torch.randn(M, C, device="cuda", dtype=torch.bfloat16)
        t = {n: timed(lambda: ops.quantize_dual(x, rht_seed=1, col_pad=128, row_terms=n)) for n in (1, 2)}
        out["quantize"].append({"shape": [M, C], "us_1": t[1], "us_2": t[2], "ratio": t[2] / t[1]})
    for N, K in ((2304, 768), (768, 768), (3072, 768), (768, 3072)):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05
        x1, _ = ops.quantize_dual(x, col=False)
        x2, _ = ops.quantize_dual(x, col=False, row_terms=2)
        wq, _ = ops.quantize_dual(w, col=False, block_2d=True)
        w2 = ops.two_term_weight(wq)
        t1 = timed(lambda: ops.gemm(x1, wq, torch.bfloat16))
        t2 = timed(lambda: ops.gemm(x2, w2, torch.bfloat16))
        fl = 2 * M * N * K
        out["gemm"].append({"N": N, "K": K, "us_1": t1, "us_2": t2, "ratio": t2 / t1,
                            "tflops_1": fl / t1 / 1e6, "tflops_2_eff": fl / t2 / 1e6})
    for K, N in ((768, 3072), (3072, 768)):
        lin = NVFP4Linear(K, N, cache_weight_quantization=True).cuda()
        x = torch.randn(M, K, device="cuda", requires_grad=True)
        g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        def step():
            y = lin(x)
            y.backward(g)

        t = {}
        for n in (1, 2):
            ops.set_fprop_terms(n)
            t[n] = timed(step, 20)
        ops.set_fprop_terms(1)
        out["linear"].append({"K": K, "N": N, "us_1": t[1], "us_2": t[2], "ratio": t[2] / t[1]})
    print(json.dumps(out, indent=1))
    os.makedirs("bench/results", exist_ok=True)
    with open("bench/results/fprop_terms.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
