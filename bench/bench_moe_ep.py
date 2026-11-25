"""Expert parallelism across GPUs: training-step time (forward + backward) of an nvfp4.moe.MoE layer whose
experts are split over W GPUs, vs every GPU holding all experts (no exchange; data parallelism would add an
all-reduce of the expert gradients, timed separately), and the all-to-all alone.

    torchrun --standalone --nproc_per_node 2 bench/bench_moe_ep.py [--out bench/results/moe_ep.json]

Each rank routes its own T tokens. Rows per configuration and expert precision (BF16 grouped / NVFP4):
  local          all E experts on every GPU, no communication
  ep cap 1.25    E / W experts per GPU, fixed-size exchanges (capacity factor 1.25)
  ep cap serial  the same with the shared expert run after the exchange instead of during it
  ep dropless    E / W experts per GPU, variable-size exchanges (one host read of the row counts)
  a2a            one all-to-all of the capacity-1.25 dispatch buffer (bf16), forward direction only
  allreduce      all-reduce of the expert gradients in bf16 (what data parallelism pays instead)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))
from bench_gemm import timeit  # noqa: E402

from nvfp4 import convert_linears  # noqa: E402
from nvfp4.moe import MoE  # noqa: E402

CONFIGS = [  # d, d_ff, experts, top_k, tokens per GPU, shared_d_ff
    (1024, 2816, 8, 2, 16384, 0),
    (2048, 1408, 32, 4, 16384, 2816),  # DeepSeek-style: many small experts and a shared expert
]


def step_ms(moe, x):
    g = torch.randn(x.shape, device="cuda", dtype=torch.bfloat16)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.backward(g)
        moe.zero_grad(set_to_none=True)
    step()
    dist.barrier()
    return timeit(step, iters=5, warmup=2, trials=5) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/results/moe_ep.json")
    ap.add_argument("--smoke", action="store_true", help="tiny shapes over gloo, every rank on GPU 0")
    a = ap.parse_args()
    rank, W = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if a.smoke:
        torch.cuda.set_device(0)
        dist.init_process_group("gloo")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
        dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    rows = []
    for d, d_ff, E, k, T, shared in ([(256, 256, 4, 2, 512, 256)] if a.smoke else CONFIGS):
        torch.manual_seed(0)
        local = MoE(d, d_ff, E, top_k=k, shared_d_ff=shared).cuda()
        torch.manual_seed(1 + rank)  # every rank routes its own tokens
        x = torch.randn(T, d, device="cuda", requires_grad=True)
        for prec in ("bf16", "nvfp4"):
            def make(**kw):
                m = MoE(d, d_ff, E, top_k=k, shared_d_ff=shared, **kw).cuda()
                if "ep_group" in kw:
                    m.load_state_dict({n: (v[rank * m.n_local:(rank + 1) * m.n_local]
                                           if n.startswith(("w_", "b_")) else v)
                                       for n, v in local.state_dict().items()})
                else:
                    m.load_state_dict(local.state_dict())
                m.bf16_impl = "grouped"
                if prec == "nvfp4":
                    convert_linears(m, fuse_wgrad_accumulation=False)
                return m
            res = {"local": step_ms(make(), x)}
            ep = make(ep_group=dist.group.WORLD, capacity_factor=1.25)
            res["ep cap 1.25"] = step_ms(ep, x)
            if shared:
                ep.ep_overlap = False
                res["ep cap serial"] = step_ms(ep, x)
            res["ep dropless"] = step_ms(make(ep_group=dist.group.WORLD), x)
            rows.append({"d": d, "d_ff": d_ff, "E": E, "k": k, "T": T, "shared": shared, "W": W,
                         "precision": prec, **res})
            if rank == 0:
                print(f"d={d} d_ff={d_ff} E={E} k={k} T={T} shared={shared} {prec:5s} " +
                      "  ".join(f"{n} {v:.2f} ms" for n, v in res.items()), flush=True)
        # the communication alone
        C = math.ceil(1.25 * T * k / E)
        buf = torch.randn(E * C, d, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(buf)
        a2a = timeit(lambda: dist.all_to_all_single(out, buf), iters=5, warmup=2, trials=5) * 1e3
        n_params = sum(p.numel() for n, p in local.named_parameters() if n.startswith("w_"))
        grads = torch.randn(n_params, device="cuda", dtype=torch.bfloat16)
        ar = timeit(lambda: dist.all_reduce(grads), iters=3, warmup=1, trials=3) * 1e3
        mb = buf.numel() * 2 / 2**20
        rows.append({"d": d, "E": E, "k": k, "T": T, "W": W, "a2a_ms": a2a, "a2a_MiB": mb,
                     "allreduce_ms": ar, "allreduce_MiB": grads.numel() * 2 / 2**20})
        if rank == 0:
            print(f"  a2a of {mb:.0f} MiB: {a2a:.2f} ms ({mb / 1024 * (W - 1) / W / a2a * 1e3:.1f} GiB/s sent per "
                  f"GPU); all-reduce of {grads.numel() * 2 / 2**20:.0f} MiB expert gradients: {ar:.2f} ms", flush=True)
        del local, x, buf, out, grads
        torch.cuda.empty_cache()
    if rank == 0:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        meta = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "torch": torch.__version__}
        json.dump({"meta": meta, "rows": rows}, open(a.out, "w"), indent=1)
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
