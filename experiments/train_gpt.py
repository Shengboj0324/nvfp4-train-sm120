"""Pretrain a small GPT on TinyStories in BF16 or NVFP4 and log loss / throughput.

    python experiments/train_gpt.py --precision bf16  --out experiments/results/bf16.jsonl
    python experiments/train_gpt.py --precision nvfp4 --out experiments/results/nvfp4.jsonl

Both runs use the same seed, initialisation and data order. In NVFP4 mode every transformer-block
linear (qkv, attn-out, fc1, fc2) runs fprop/dgrad/wgrad on NVFP4 tensor cores, except the last
--hp-last blocks; embeddings, LM head, LayerNorms and attention (SDPA) stay BF16, as in NVIDIA's
recipe. The optimizer keeps fp32 master weights in both modes (autocast BF16).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4.fused import layernorm_linear, layernorm_mlp
from nvfp4.linear import NVFP4Linear, convert_linears


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
        self.fused = False  # set by fuse_producers(): LayerNorm / GELU run inside the NVFP4 quantizers

    def forward(self, x):
        b, t, d = x.shape
        if self.fused:
            q, k, v = layernorm_linear(x, self.ln1, self.qkv).split(d, dim=2)
        else:
            q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        if self.fused:
            return x + layernorm_mlp(x, self.ln2, self.fc1, self.fc2)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


def fuse_producers(blocks) -> int:
    """Run LayerNorm and GELU inside the NVFP4 quantizers in every block whose qkv / fc1 / fc2 are
    NVFP4Linear (nvfp4.fused). Returns the number of fused blocks."""
    n = 0
    for blk in blocks:
        if all(isinstance(m, NVFP4Linear) and m.backend == "cuda" for m in (blk.qkv, blk.fc1, blk.fc2)):
            blk.fused = True
            n += 1
    return n


class GPT(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int, heads: int, ctx: int):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.wte.weight  # tied
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "fc2.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                                                            targets.view(-1))
        return logits, loss


def batches(data: np.memmap, b: int, t: int, gen: torch.Generator, device="cuda"):
    while True:
        ix = torch.randint(len(data) - t - 1, (b,), generator=gen)
        x = torch.stack([torch.from_numpy(data[i:i + t].astype(np.int64)) for i in ix.tolist()])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + t].astype(np.int64)) for i in ix.tolist()])
        yield x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, val, b, t, n_batches=20):
    model.eval()
    gen = torch.Generator().manual_seed(1234)  # identical val batches every time and in every run
    it = batches(val, b, t, gen)
    tot = 0.0
    for _ in range(n_batches):
        x, y = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += model(x, y)[1].item()
    model.train()
    return tot / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["bf16", "nvfp4"], required=True)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32, help="sequences per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=16, help="sequences per forward/backward")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--hp-last", type=int, default=2, help="keep the last N blocks in BF16 (NVFP4 mode)")
    ap.add_argument("--no-rht", action="store_true", help="ablation: disable the wgrad Hadamard transform")
    ap.add_argument("--fuse-wgrad", action="store_true",
                    help="accumulate NVFP4 weight grads across micro-batches inside the wgrad GEMM")
    ap.add_argument("--fuse-producers", action="store_true",
                    help="run LayerNorm / GELU (and its backward) inside the NVFP4 quantizers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=os.environ.get("NVFP4_DATA", os.path.expanduser("~/nvfp4_data")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # Data parallel when launched with torchrun (NCCL, one process per GPU): the global batch stays
    # a.batch sequences per optimizer step, split over ranks; one process keeps the single-GPU path.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world > 1
    rank = 0
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(os.environ.get("NVFP4_DIST_BACKEND", "nccl"))  # gloo: smoke tests on one GPU
        rank = dist.get_rank()
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())

    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    model = GPT(50304, a.d, a.layers, a.heads, a.ctx).cuda()
    converted = []
    if a.precision == "nvfp4":
        skip = tuple(f"blocks.{i}." for i in range(a.layers - a.hp_last, a.layers))
        converted = convert_linears(model.blocks, skip=tuple(s[len("blocks."):] for s in skip),
                                    wgrad_rht=not a.no_rht, fuse_wgrad_accumulation=a.fuse_wgrad)
        if a.fuse_producers:
            fuse_producers(model.blocks)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    fwd = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        fwd = DDP(model, device_ids=[torch.cuda.current_device()])
        torch.cuda.manual_seed(a.seed * 7919 + rank + 1)  # independent rounding seeds per rank
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))  # noqa: E731
    assert a.batch % (a.micro_batch * world) == 0
    accum = a.batch // (a.micro_batch * world)
    gen = torch.Generator().manual_seed(a.seed if not ddp else a.seed * 1000 + rank)
    it = batches(train, a.micro_batch, a.ctx, gen)
    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    log = open(a.out if rank == 0 else os.devnull, "w", buffering=1)
    meta = {"config": vars(a), "params": n_params, "nvfp4_linears": len(converted), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(), "world_size": world}
    log.write(json.dumps({"meta": meta}) + "\n")
    if rank == 0:
        print(meta, flush=True)

    tok_per_step = a.batch * a.ctx
    t_last, tokens_window = time.perf_counter(), 0
    t_start = t_last
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            vl = evaluate(model, val, a.micro_batch, a.ctx, n_batches=40)
            rec = {"step": step, "val_loss": vl, "elapsed_s": time.perf_counter() - t_start}
            log.write(json.dumps(rec) + "\n")
            if rank == 0:
                print(rec, flush=True)
            t_last, tokens_window = time.perf_counter(), 0  # exclude eval from throughput
        if step == a.steps:
            break
        for g in opt.param_groups:
            g["lr"] = a.lr * sched(step)
        loss_acc = 0.0
        for i in range(accum):
            x, y = next(it)
            sync = not ddp or i == accum - 1  # all-reduce gradients once per optimizer step
            with (contextlib.nullcontext() if sync else fwd.no_sync()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = fwd(x, y)
                (loss / accum).backward()
            loss_acc += loss.detach() / accum
        loss = loss_acc
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_window += tok_per_step
        if step % 25 == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            rec = {"step": step, "loss": loss.item(), "grad_norm": gn.item(), "lr": a.lr * sched(step),
                   "tok_per_s": tokens_window / (now - t_last) if step else None,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            log.write(json.dumps(rec) + "\n")
            if step % 100 == 0 and rank == 0:
                print(rec, flush=True)
            t_last, tokens_window = now, 0
            if not math.isfinite(rec["loss"]):
                print("non-finite loss, stopping", flush=True)
                break
    log.write(json.dumps({"done": True, "total_s": time.perf_counter() - t_start}) + "\n")
    log.close()
    # Wind down explicitly (finish queued work, release cached memory) rather than tearing down a busy
    # CUDA context at interpreter exit; on this laptop a run once ended in a display-driver crash.
    torch.cuda.synchronize()
    del model, opt, fwd
    torch.cuda.empty_cache()
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
