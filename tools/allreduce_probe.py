"""NCCL all-reduce bandwidth between the visible GPUs (run with torchrun), to tell a slow multi-GPU host
(no peer-to-peer, narrow PCIe links) from slow training code.

    torchrun --standalone --nproc_per_node 2 tools/allreduce_probe.py

Reports, per message size, the time per all-reduce and the bus bandwidth (2 (n-1)/n * bytes / time, the
figure nccl-tests prints). A GPT-124M step all-reduces ~0.5 GB of fp32 gradients.
"""
import os
import time

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    for mb in (8, 64, 256, 512):
        x = torch.ones(mb * 2**20 // 4, device="cuda")
        for _ in range(3):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        n = 10
        t0 = time.perf_counter()
        for _ in range(n):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        if rank == 0:
            bus = 2 * (world - 1) / world * x.numel() * 4 / dt / 1e9
            print(f"all_reduce {mb:4d} MB fp32: {dt * 1e3:8.2f} ms, bus bandwidth {bus:6.1f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
