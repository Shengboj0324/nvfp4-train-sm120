"""How much a GEMM slows down while an all-to-all runs beside it (expert parallelism's chunked overlap):

    torchrun --standalone --nproc_per_node W bench/bench_overlap.py [--out FILE]   (NCCL_MAX_NCHANNELS etc. from env)

Per rank: an NVFP4 GEMM (the persistent TMA kernel, M x N x K) and a cuBLAS BF16 GEMM of the same shape, timed
alone and while a long all-to-all (``--mib`` per rank, repeated to outlast the GEMMs) runs on NCCL's stream;
the NVFP4 GEMM also with SM margins (ops.set_sm_margin). The NCCL kernel's grid (CTAs) comes from a profiler
trace. Rank 0 prints ms per GEMM and the slowdown factors.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import torch
import torch.distributed as dist

from nvfp4 import ops


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
    ap.add_argument("--mib", type=int, default=64, help="all-to-all size per rank")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    dev = "cuda"
    qa, _ = ops.quantize_dual(torch.randn(a.m, a.k, device=dev), col=False)
    qb, _ = ops.quantize_dual(torch.randn(a.n, a.k, device=dev), col=False)
    xa = torch.randn(a.m, a.k, device=dev, dtype=torch.bfloat16)
    xb = torch.randn(a.n, a.k, device=dev, dtype=torch.bfloat16)
    nv = lambda: ops.gemm(qa, qb, impl="tma")  # noqa: E731
    bf = lambda: xa @ xb.T  # noqa: E731
    buf = torch.empty(a.mib * 2**20 // 2, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(buf)
    for f in (nv, bf):
        f()
    torch.cuda.synchronize()
    a2a_ms = timed(lambda: dist.all_to_all_single(out, buf), 5)
    res = {"a2a_ms": a2a_ms, "a2a_GiB_s_per_gpu": a.mib / 1024 * (W - 1) / W / a2a_ms * 1e3}

    def beside(fn, iters):
        """fn's time while all-to-alls run: enough of them queued on NCCL's stream to outlast the GEMMs."""
        alone = timed(fn, 3)
        reps = max(2, int(alone * iters / a2a_ms) + 2)
        dist.barrier()
        torch.cuda.synchronize()
        works = [dist.all_to_all_single(out, buf, async_op=True) for _ in range(reps)]
        t = timed(fn, iters)
        for w in works:
            w.wait()
        torch.cuda.synchronize()
        return t

    res["nvfp4_alone_ms"] = timed(nv, a.iters)
    res["bf16_alone_ms"] = timed(bf, a.iters)
    res["bf16_beside_ms"] = beside(bf, a.iters)
    for margin in (0, 8, 16, 24, 32, 48):
        ops.set_sm_margin(margin)
        res[f"nvfp4_alone_margin{margin}_ms"] = timed(nv, a.iters)
        res[f"nvfp4_beside_margin{margin}_ms"] = beside(nv, a.iters)
    ops.set_sm_margin(0)
    # the NCCL kernel's grid, from a trace of one all-to-all
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        dist.all_to_all_single(out, buf)
        torch.cuda.synchronize()
    grids = []
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.json")
        prof.export_chrome_trace(p)
        for ev in json.load(open(p)).get("traceEvents", []):
            if ev.get("cat") == "kernel" and "nccl" in ev.get("name", "").lower():
                grids.append(ev.get("args", {}).get("grid"))
    res["nccl_grid"] = grids[:4]
    res["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
    res["env"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    if rank == 0:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
        nb = res["nvfp4_alone_ms"]
        print(f"nvfp4 GEMM {a.m}x{a.n}x{a.k}: alone {nb:.3f} ms; beside the all-to-all " +
              ", ".join(f"margin {m}: {res[f'nvfp4_beside_margin{m}_ms'] / nb:.2f}x"
                        for m in (0, 8, 16, 24, 32, 48)) +
              f"; bf16 cuBLAS {res['bf16_beside_ms'] / res['bf16_alone_ms']:.2f}x; NCCL grid {res['nccl_grid']}",
              flush=True)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
