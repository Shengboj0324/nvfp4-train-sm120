"""The persistent NVFP4 GEMM while another kernel holds some SMs (what NCCL's all-to-all CTAs do during
overlapped expert parallelism), with the fixed tile assignment and with dynamic tile claiming:

    python bench/bench_residency.py [--out FILE]

A spin kernel of k CTAs, each with 64 KB of shared memory (so no GEMM CTA fits beside it), runs on a second
stream for longer than the timed GEMMs. With the fixed assignment the GEMM's k CTAs that cannot start run
their share of the tiles after the others finish (a second wave: ~2x); with dynamic claiming the resident CTAs
take that work (~SMs / (SMs - k)). Single GPU; no NCCL needed.
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.cpp_extension import load_inline

from nvfp4 import ops

SRC = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void hold_kernel(long long cycles) {
  extern __shared__ char smem[];
  const long long t0 = clock64();
  while (clock64() - t0 < cycles) {
    if (threadIdx.x == 1024) smem[0] = 0;  // never true: keeps the shared memory allocation
  }
}
void hold(int64_t ctas, int64_t cycles) {
  const int bytes = 64 * 1024;
  cudaFuncSetAttribute(hold_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  hold_kernel<<<ctas, 32, bytes, c10::cuda::getCurrentCUDAStream()>>>(cycles);
}
"""


def timed(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=16384)
    ap.add_argument("--n", type=int, default=2816)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ext = load_inline("nvfp4_hold", cpp_sources="void hold(int64_t ctas, int64_t cycles);", cuda_sources=SRC,
                      functions=["hold"], extra_cuda_cflags=["-O2"], verbose=False)
    qa, _ = ops.quantize_dual(torch.randn(a.m, a.k, device="cuda"), col=False)
    qb, _ = ops.quantize_dual(torch.randn(a.n, a.k, device="cuda"), col=False)
    gemm = lambda: ops.gemm(qa, qb, impl="tma")  # noqa: E731
    props = torch.cuda.get_device_properties(0)
    clock_khz = getattr(props, "clock_rate", 0) or 2_500_000
    side = torch.cuda.Stream()
    res = {"sms": props.multi_processor_count, "shape": [a.m, a.n, a.k], "device": props.name}
    for f in range(3):
        gemm()
    ext.hold(1, 1000)  # load the spin kernel before timing (lazy module loading delayed the first hold)
    torch.cuda.synchronize()
    for mode in ("static", "dynamic"):
        ops.set_dynamic_tiles(mode == "dynamic")
        alone = timed(gemm, a.iters)
        res[f"{mode}_alone_ms"] = alone
        for k in (1, 2, 4, 8, 16):
            cycles = int(alone * a.iters * 3 * clock_khz)  # ~3x the timed batch (ms * kHz = cycles)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ext.hold(k, cycles)
            torch.cuda._sleep(1_000_000)  # let the hold kernel take its SMs first
            t = timed(gemm, a.iters)
            torch.cuda.synchronize()
            res[f"{mode}_held{k}_ms"] = t
    ops.set_dynamic_tiles(False)
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))
    for mode in ("static", "dynamic"):
        al = res[f"{mode}_alone_ms"]
        print(f"{mode:8s} alone {al:.3f} ms; with k SMs held: " +
              ", ".join(f"k={k} {res[f'{mode}_held{k}_ms'] / al:.2f}x" for k in (1, 2, 4, 8, 16)))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
