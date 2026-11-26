"""Expert parallelism across GPUs: training-step time (forward + backward) of an nvfp4.moe.MoE layer whose
experts are split over W GPUs, vs every GPU holding all experts (no exchange; data parallelism would add an
all-reduce of the expert gradients, timed separately), and the all-to-all alone.

    torchrun --standalone --nproc_per_node W bench/bench_moe_ep.py [--node-size L] [--out bench/results/moe_ep.json]

Each rank routes its own T tokens. Rows per configuration and expert precision (BF16 grouped / NVFP4):
  local              all E experts on every GPU, no communication
  ep cap 1.25        E / W experts per GPU, fixed-size per-slot exchanges (capacity factor 1.25)
  ep cap serial      the same with the shared expert run after the exchange instead of during it
  ep dropless        variable-size exchanges, each token once per rank holding one of its experts
  ... chunks n       the tokens in n chunks, exchanges overlapping the experts (wavefront)
  ... wire nvfp4     NVFP4 activation messages (+grad: NVFP4 gradient messages too; NVFP4 experts only)
  ... nodes L        (--node-size L) two-stage exchanges over nodes of L GPUs
  a2a                one all-to-all of the capacity-1.25 dispatch buffer (bf16), forward direction only
  allreduce          all-reduce of the expert gradients in bf16 (what data parallelism pays instead)
Also recorded: rows each rank sends to other ranks per step (forward dispatch; ``last_ep_rows``).
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

from nvfp4 import convert_linears, ops  # noqa: E402
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
    ap.add_argument("--node-size", type=int, default=1, help="also time two-stage exchanges over nodes of this size")
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
            res, sent = {"local": step_ms(make(), x)}, {}
            variants = [("ep cap 1.25", {"capacity_factor": 1.25}, {})]
            m8 = {"margin": 8}  # SMs the persistent GEMMs leave to the all-to-alls (ops.set_sm_margin)
            if shared:
                variants.append(("ep cap serial", {"capacity_factor": 1.25}, {"ep_overlap": False}))
            variants += [("ep dropless", {}, {}), ("ep dropless chunks 2", {}, {"ep_chunks": 2}),
                         ("ep dropless chunks 4", {}, {"ep_chunks": 4}),
                         ("ep dropless chunks 2 margin 8", {}, {"ep_chunks": 2, **m8}),
                         ("ep dropless chunks 4 margin 8", {}, {"ep_chunks": 4, **m8})]
            if prec == "nvfp4":
                variants += [("ep wire nvfp4", {}, {"ep_wire": "nvfp4"}),
                             ("ep wire nvfp4 chunks 2", {}, {"ep_wire": "nvfp4", "ep_chunks": 2}),
                             ("ep wire nvfp4+grad chunks 2", {}, {"ep_wire": "nvfp4", "ep_wire_grad": "nvfp4",
                                                                  "ep_chunks": 2}),
                             ("ep wire nvfp4 chunks 2 margin 8", {}, {"ep_wire": "nvfp4", "ep_chunks": 2, **m8}),
                             ("ep wire nvfp4 chunks 4 margin 8", {}, {"ep_wire": "nvfp4", "ep_chunks": 4, **m8}),
                             ("ep wire nvfp4 + fp8 return", {}, {"ep_wire": "nvfp4", "ep_wire_return": "fp8"}),
                             ("ep wire nvfp4 + fp8 return chunks 2", {}, {"ep_wire": "nvfp4", "ep_wire_return": "fp8",
                                                                         "ep_chunks": 2})]
            else:
                variants.append(("ep fp8 return", {}, {"ep_wire_return": "fp8"}))
            if a.node_size > 1:
                L = a.node_size
                variants += [(f"ep nodes {L}", {"ep_node_size": L}, {}),
                             (f"ep nodes {L} chunks 2", {"ep_node_size": L}, {"ep_chunks": 2, "ep_serial": False}),
                             (f"ep nodes {L} chunks 2 serial", {"ep_node_size": L}, {"ep_chunks": 2, "ep_serial": True}),
                             (f"ep nodes {L} chunks 4 serial", {"ep_node_size": L}, {"ep_chunks": 4, "ep_serial": True}),
                             (f"ep nodes {L} fp8 return", {"ep_node_size": L}, {"ep_wire_return": "fp8"})]
                if prec == "nvfp4":
                    variants += [(f"ep nodes {L} wire nvfp4", {"ep_node_size": L}, {"ep_wire": "nvfp4"}),
                                 (f"ep nodes {L} wire nvfp4 + fp8 return chunks 2 serial", {"ep_node_size": L},
                                  {"ep_wire": "nvfp4", "ep_wire_return": "fp8", "ep_chunks": 2, "ep_serial": True})]
            for name, ctor, attrs in variants:
                m = make(ep_group=dist.group.WORLD, **ctor)
                attrs = dict(attrs)
                ops.set_sm_margin(attrs.pop("margin", 0))
                for k_, v_ in attrs.items():
                    setattr(m, k_, v_)
                res[name] = step_ms(m, x)
                ops.set_sm_margin(0)
                if m.last_ep_rows is not None:
                    sent[name] = list(m.last_ep_rows)
                del m
            rows.append({"d": d, "d_ff": d_ff, "E": E, "k": k, "T": T, "shared": shared, "W": W,
                         "precision": prec, **res, "rows_sent": sent})
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
