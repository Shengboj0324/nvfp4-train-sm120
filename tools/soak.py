"""Soak test: train pre-LN GPT blocks in NVFP4 (fused and unfused paths) and hammer the GEMM shapes
of a d=2048 model for a fixed wall-clock budget, checking continuously that

  * every loss, gradient and GEMM output is finite,
  * replaying a step with the same seed reproduces its gradients bit for bit (no races / stale state),
  * a block trained through a captured CUDA graph (fresh GPU-drawn seeds every replay) stays finite,
  * GEMM results stay bit-identical to their first computation (no drift, no hangs: the guard and the
    kernels' bounded waits turn a hang into an error).

Run it under the resource guard:
    tools/run.sh soak tools/soak.py --minutes 5
Exit code 0 means every check passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from train_gpt import Block, fuse_producers  # noqa: E402

from nvfp4 import ops  # noqa: E402
from nvfp4.linear import convert_linears  # noqa: E402


def make_block(d: int, fused: bool, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    blk = Block(d, d // 64).cuda()
    convert_linears(blk, fuse_wgrad_accumulation=fused)
    if fused:
        fuse_producers([blk])
    return blk


def step(blk, x, seed):
    blk.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = blk(x).float().pow(2).mean()
    loss.backward()
    return loss.detach(), [p.grad.detach().clone() for p in blk.parameters()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--d", type=int, default=1024)
    a = ap.parse_args()
    deadline = time.time() + 60 * a.minutes
    # PyTorch's flash attention backward accumulates dQ with atomics by default and is not
    # bit-reproducible (a pure BF16 block fails the replay check with it). Deterministic mode switches
    # it to a reproducible backward at little cost, so any replay mismatch can only come from this
    # repository's kernels. (It also fills fresh allocations with NaN, so a kernel that read
    # uninitialized output memory would fail the finiteness checks.)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

    blocks = {"unfused": make_block(a.d, False, 0), "fused": make_block(a.d, True, 0)}
    opts = {k: torch.optim.AdamW(b.parameters(), lr=1e-4) for k, b in blocks.items()}
    T = 4096
    shapes = [(T, 6144, 2048), (T, 2048, 8192), (6144, 2048, T), (2048, 8192, T), (768, 768, 8192)]
    gemms = []
    for (m, n, k) in shapes:
        qa = ops.quantize(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
        qb = ops.quantize(torch.randn(n, k, device="cuda", dtype=torch.bfloat16))
        gemms.append((qa, qb, ops.gemm(qa, qb, torch.float32)))

    # A third block trains through a captured CUDA graph (forward, backward, optimizer step): seeds are
    # drawn on the GPU, so every replay uses fresh ones; checked for finite loss / weights.
    gblk = make_block(a.d, False, 1)
    gopt = torch.optim.AdamW(gblk.parameters(), lr=1e-4, capturable=True)
    gx = torch.randn(8, 512, a.d, device="cuda")

    def gstep():
        gopt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = gblk(gx).float().pow(2).mean()
        loss.backward()
        gopt.step()
        return loss.detach()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            gstep()
    torch.cuda.current_stream().wait_stream(side)
    gopt.zero_grad(set_to_none=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gloss = gstep()

    it, checks = 0, 0
    while time.time() < deadline:
        x = torch.randn(8, 512, a.d, device="cuda")
        gx.copy_(x)
        graph.replay()
        if it % 10 == 0 and not (torch.isfinite(gloss) and all(torch.isfinite(p).all() for p in gblk.parameters())):
            print(f"FAIL non-finite in the CUDA-graph step at iter {it}", flush=True)
            return 1
        for name, blk in blocks.items():
            seed = 1000 + it
            loss, grads = step(blk, x, seed)
            if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in grads):
                print(f"FAIL non-finite at iter {it} ({name})", flush=True)
                return 1
            if it % 10 == 0:  # replay: same weights, same seed -> identical gradients
                _, again = step(blk, x, seed)
                if not all(torch.equal(g, h) for g, h in zip(grads, again)):
                    print(f"FAIL replay mismatch at iter {it} ({name})", flush=True)
                    return 1
                checks += 1
            opts[name].step()
        for qa, qb, ref in gemms:
            out = ops.gemm(qa, qb, torch.float32)
            if it % 10 == 0 and not torch.equal(out, ref):
                print(f"FAIL GEMM drift at iter {it} for {tuple(qa.shape)} x {tuple(qb.shape)}", flush=True)
                return 1
        if it % 25 == 0:
            torch.cuda.synchronize()
            print(f"iter {it}: losses " + ", ".join(f"{n} {s:.4f}" for n, s in
                  ((n, step(b, x, 7)[0].item()) for n, b in blocks.items())), flush=True)
        it += 1
    torch.cuda.synchronize()
    print(f"SOAK_OK {it} iterations, {checks} replay checks, {it * len(gemms)} extra GEMMs, "
          f"{it} CUDA-graph training steps, {a.minutes:.1f} min", flush=True)
    del blocks, opts, gemms, graph, gblk, gopt
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
