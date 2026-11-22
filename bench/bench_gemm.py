"""GEMM throughput on the local GPU: BF16 cuBLAS vs this repo's NVFP4 kernel vs cuBLASLt NVFP4.

    python bench/bench_gemm.py [--out bench/results/gemm.json]

Every NVFP4 timing is preceded by a correctness check against the fp64 reference, and the cuBLASLt
path is only timed if its output agrees with ours. Times are medians over --iters launches measured
with CUDA events after warm-up. FLOPs = 2*M*N*K.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time

import torch

from nvfp4 import ops
from nvfp4 import reference as ref

SHAPES = [
    # square
    (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
    # transformer-training shaped (tokens x hidden): fprop / dgrad / wgrad of a 2048->8192 MLP
    (8192, 8192, 2048), (8192, 2048, 8192), (2048, 8192, 8192),
    # smaller model (768 hidden, 3072 MLP), 8192 tokens
    (8192, 3072, 768), (8192, 768, 3072), (768, 3072, 8192),
]


def timeit(fn, iters: int, warmup: int = 5, trials: int = 7) -> float:
    """Median over `trials` of the mean time of `iters` back-to-back launches (seconds).

    Launches are batched between two events: syncing after every launch leaves the GPU idle between
    iterations, and this laptop GPU drops its clocks within microseconds of idling, which made
    multi-kernel workloads look up to 7x slower than they run in a busy training stream.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e-3 / iters)
    times.sort()
    return times[len(times) // 2]


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """cuBLASLt block-scale layout (128x4 tiles, 32x16 swizzle), as used by torchao."""
    rows, cols = scales.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    p = torch.zeros(pr, pc, dtype=scales.dtype, device=scales.device)
    p[:rows, :cols] = scales
    blocks = p.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def cublaslt_nvfp4(qa: ops.NVFP4Tensor, qb: ops.NVFP4Tensor):
    """Return a callable running torch._scaled_mm on the same NVFP4 operands, or (None, reason)."""
    if not hasattr(torch, "float4_e2m1fn_x2"):
        return None, "torch has no float4_e2m1fn_x2"
    try:
        a = qa.data.view(torch.float4_e2m1fn_x2)
        b = qb.data.view(torch.float4_e2m1fn_x2)
        sa = to_blocked(qa.scales).view(torch.float8_e4m3fn)
        sb = to_blocked(qb.scales).view(torch.float8_e4m3fn)
        alpha = (qa.gdec * qb.gdec).reshape(())

        def run():
            return torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16) * alpha

        run()
        return run, None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"


def gpu_info() -> dict:
    q = "name,driver_version,power.limit,clocks.max.sm,clocks.sm,temperature.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    return {"nvidia_smi": out, "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
            "host": platform.node()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="bench/results/gemm.json")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds idle between shapes (laptop thermals)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    info = gpu_info()
    print(info)
    rows = []
    for (m, n, k) in SHAPES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        qa, qb = ops.quantize(a), ops.quantize(b)

        # correctness gate (sampled rows to keep the fp64 reference cheap)
        d = ops.gemm(qa, qb, out_dtype=torch.float32)
        rs = slice(0, min(m, 256))
        r = ref.gemm((qa.data[rs], qa.scales[rs], qa.gdec.reshape(())), qb.as_tuple())
        rel = ((d[rs] - r).abs().max() / r.abs().max()).item()
        assert rel < 1e-5, f"NVFP4 GEMM mismatch {rel} at {(m, n, k)}"

        flops = 2.0 * m * n * k
        t_bf16 = timeit(lambda: a @ b.T, args.iters)
        t_fp4 = timeit(lambda: ops.gemm(qa, qb), args.iters)
        t_fp4_q = timeit(lambda: ops.gemm(ops.quantize(a), ops.quantize(b)), args.iters)
        row = {"M": m, "N": n, "K": k,
               "bf16_cublas_tflops": flops / t_bf16 / 1e12,
               "nvfp4_ours_tflops": flops / t_fp4 / 1e12,
               "nvfp4_ours_incl_quant_tflops": flops / t_fp4_q / 1e12,
               "check_rel_err": rel}
        run, why = cublaslt_nvfp4(qa, qb)
        if run is not None:
            dc = run().float()
            agree = ((dc - d).abs().max() / d.abs().max()).item()
            row["cublaslt_vs_ours_rel_diff"] = agree
            if agree < 1e-2:
                row["nvfp4_cublaslt_tflops"] = flops / timeit(run, args.iters) / 1e12
            else:
                row["cublaslt_note"] = "output disagrees with reference; not timed"
        else:
            row["cublaslt_note"] = why
        clk = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        row["after_shape_gpu_state"] = clk
        rows.append(row)
        print(json.dumps({k_: (round(v, 2) if isinstance(v, float) and k_.endswith("tflops") else v)
                          for k_, v in row.items()}), flush=True)
        del a, b, qa, qb, d
        torch.cuda.empty_cache()
        time.sleep(args.cooldown)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"info": info, "results": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
