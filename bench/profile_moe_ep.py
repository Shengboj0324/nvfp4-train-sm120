"""Where an expert-parallel MoE training step spends its time (torch.profiler over a few steps):

    torchrun --standalone --nproc_per_node W bench/profile_moe_ep.py [--out DIR]

Per variant (rank 0 prints): wall time per step; GPU time per step of the compute kernels and of the NCCL
kernels (summed over kernels, so overlapped time counts twice); the compute stream's idle time (wall minus the
union of its kernel intervals); the host time the step spends in the layer's Python / launches before the GPU
catches up (the time blocked in the counts read ``.tolist()``); and the top CUDA kernels. Rank 0's Chrome traces go to
DIR/trace_<variant>_rank0.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from nvfp4 import convert_linears, ops
from nvfp4.moe import MoE

VARIANTS = {  # name: (constructor kwargs, attributes, SM margin)
    "dropless": ({}, {}, 0),
    "chunks2": ({}, {"ep_chunks": 2}, 0),
    "chunks2_margin8": ({}, {"ep_chunks": 2}, 8),
    "wire_nvfp4": ({}, {"ep_wire": "nvfp4"}, 0),
    "cap1.25": ({"capacity_factor": 1.25}, {}, 0),
}


def union_ms(intervals):
    tot, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e <= end:
            continue
        tot += e - max(s, end)
        end = e
    return tot / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/profile_moe_ep")
    ap.add_argument("--d", type=int, default=2048)
    ap.add_argument("--d-ff", type=int, default=1408)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--shared", type=int, default=2816)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(1 + rank)
    x = torch.randn(a.tokens, a.d, device="cuda", requires_grad=True)
    g = torch.randn(a.tokens, a.d, device="cuda", dtype=torch.bfloat16)
    summary = {}
    for name, (ctor, attrs, margin) in VARIANTS.items():
        torch.manual_seed(0)
        m = MoE(a.d, a.d_ff, a.experts, top_k=a.top_k, shared_d_ff=a.shared, ep_group=dist.group.WORLD, **ctor).cuda()
        convert_linears(m)
        for k, v in attrs.items():
            setattr(m, k, v)
        ops.set_sm_margin(margin)

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = m(x)
            y.backward(g)
            m.zero_grad(set_to_none=True)
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        dist.barrier()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(a.steps):
                step()
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1e3 / a.steps
        ops.set_sm_margin(0)
        if rank == 0:
            prof.export_chrome_trace(os.path.join(a.out, f"trace_{name}_rank0.json"))
        evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        nccl = [e for e in evs if "nccl" in e.name.lower()]
        comp = [e for e in evs if "nccl" not in e.name.lower()]
        comp_ms = sum(e.device_time for e in comp) / 1e3 / a.steps
        nccl_ms = sum(e.device_time for e in nccl) / 1e3 / a.steps
        busy = union_ms([(e.time_range.start, e.time_range.end) for e in comp]) / a.steps
        host = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU]
        tolist_ms = sum(e.cpu_time_total for e in host if e.name in ("aten::item", "aten::_local_scalar_dense")) \
            / 1e3 / a.steps
        top = {}
        for e in comp:
            top[e.name[:70]] = top.get(e.name[:70], 0.0) + e.device_time / 1e3 / a.steps
        top = dict(sorted(top.items(), key=lambda kv: -kv[1])[:12])
        n_kernels = len(comp) / a.steps
        summary[name] = {"wall_ms": wall, "compute_kernels_ms": comp_ms, "nccl_kernels_ms": nccl_ms,
                         "compute_busy_ms": busy, "compute_idle_ms": wall - busy, "host_sync_ms": tolist_ms,
                         "kernels_per_step": n_kernels, "top_kernels_ms": top}
        if rank == 0:
            print(f"{name:16s} wall {wall:7.2f}  compute {comp_ms:7.2f} (busy {busy:7.2f}, idle {wall - busy:6.2f})  "
                  f"nccl {nccl_ms:7.2f}  host sync {tolist_ms:6.2f}  kernels/step {n_kernels:.0f}", flush=True)
        del m, prof
        torch.cuda.empty_cache()
    if rank == 0:
        json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
        for name, s in summary.items():
            print(name, json.dumps({k: round(v, 2) for k, v in s["top_kernels_ms"].items()}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
