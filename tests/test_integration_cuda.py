"""Training-stack integration: GPU-resident seeds (no host syncs), torch.compile without graph breaks,
CUDA-graph capture of whole training steps, fused gradient accumulation that keeps autograd hooks,
and real two-process DDP / FSDP2 (gloo; both ranks share the one GPU)."""
import contextlib
import os
import socket

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref
from nvfp4.fused import layernorm_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
DEV = "cuda"


def _q_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.as_storage(), b.as_storage()))


# --- seeds as device tensors -------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["plain", "layernorm", "gelu", "gelu_bwd"])
def test_tensor_seeds_match_int_seeds(op):
    torch.manual_seed(0)
    x = torch.randn(200, 256, device=DEV, dtype=torch.bfloat16)
    h = torch.randn(200, 256, device=DEV, dtype=torch.bfloat16)
    w, b = torch.randn(256, device=DEV), torch.randn(256, device=DEV)
    ints = (123456789012345, 987654321, 2**61 + 7)
    tens = torch.tensor(ints, device=DEV, dtype=torch.int64)
    kw_int = dict(row_sr_seed=ints[0], col_sr_seed=ints[1], rht_seed=ints[2], col_pad=128)
    kw_ten = dict(row_sr_seed=tens[0], col_sr_seed=tens[1], rht_seed=tens[2], col_pad=128)
    call = {"plain": lambda **kw: ops.quantize_dual(x, **kw),
            "layernorm": lambda **kw: ops.layernorm_quantize_dual(x, w, b, 1e-5, **kw)[2:],
            "gelu": lambda **kw: ops.gelu_quantize_dual(h, **kw),
            "gelu_bwd": lambda **kw: ops.gelu_backward_quantize_dual(x, h, **kw)}[op]
    a, b_ = call(**kw_int), call(**kw_ten)
    assert _q_equal(a[0], b_[0]) and _q_equal(a[1], b_[1])
    if op == "plain":  # and both equal the executable spec
        r = ref.quantize_dual(x, **kw_int)
        assert torch.equal(a[1].data, r[1][0]) and torch.equal(a[0].scales, r[0][1])


def test_seed_tensor_validation():
    x = torch.randn(64, 64, device=DEV)
    with pytest.raises(RuntimeError, match="one-element int64"):
        ops.quantize_dual(x, row_sr_seed=torch.tensor([1, 2], device=DEV))
    with pytest.raises(RuntimeError, match="one-element int64"):
        ops.quantize_dual(x, row_sr_seed=torch.tensor(1, dtype=torch.int32, device=DEV))
    with pytest.raises(RuntimeError, match="one-element int64"):
        ops.quantize_dual(x, row_sr_seed=torch.tensor(1))  # CPU seed tensor


def _block(d=256):
    ln = torch.nn.LayerNorm(d).to(DEV)
    fc1, fc2 = NVFP4Linear(d, 4 * d, bias=False).to(DEV), NVFP4Linear(4 * d, d, bias=False).to(DEV)
    qkv = NVFP4Linear(d, d, bias=True).to(DEV)
    return ln, fc1, fc2, qkv


def test_training_step_has_no_host_sync():
    ln, fc1, fc2, qkv = _block()
    x = torch.randn(4, 128, 256, device=DEV, requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any host synchronization raises
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = layernorm_mlp(x, ln, fc1, fc2) + qkv(x)
        y.float().square().mean().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert all(torch.isfinite(p.grad).all() for p in (fc1.weight, fc2.weight, qkv.weight, x))


# --- torch.compile -----------------------------------------------------------------------------------

def test_compile_fullgraph_matches_eager_bitwise():
    """No graph breaks (fullgraph=True raises on one), and with inductor drawing random numbers like
    eager (fallback_random), the compiled step is bit-identical to eager: same seeds, same kernels."""
    import torch._inductor.config as inductor_config
    torch.manual_seed(0)
    m = torch.nn.Sequential(NVFP4Linear(512, 1024, bias=False), NVFP4Linear(1024, 512, bias=False)).to(DEV)
    x = torch.randn(256, 512, device=DEV)
    dy = torch.randn(256, 512, device=DEV)

    def run(mod):
        xg = x.clone().requires_grad_()
        for p in m.parameters():
            p.grad = None
        torch.manual_seed(7)
        y = mod(xg)
        y.backward(dy)
        return [y.detach(), xg.grad] + [p.grad.clone() for p in m.parameters()]

    eager = run(m)
    torch._dynamo.reset()
    with inductor_config.patch(fallback_random=True):
        compiled = run(torch.compile(m, fullgraph=True))
    assert all(torch.equal(a, b) for a, b in zip(eager, compiled))


def test_compile_fullgraph_fused_block():
    ln, fc1, fc2, qkv = _block()
    x = torch.randn(2, 128, 256, device=DEV)

    def f(t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (layernorm_mlp(t, ln, fc1, fc2) + qkv(t)).float()

    torch._dynamo.reset()
    cf = torch.compile(f, fullgraph=True)
    xe, xc = x.clone().requires_grad_(), x.clone().requires_grad_()
    torch.manual_seed(1)
    ye = f(xe)
    ye.square().mean().backward()
    ge = [p.grad.clone() for p in (fc1.weight, fc2.weight, qkv.weight)] + [xe.grad]
    for p in (fc1.weight, fc2.weight, qkv.weight, ln.weight, ln.bias, qkv.bias):
        p.grad = None
    torch.manual_seed(1)
    yc = cf(xc)
    yc.square().mean().backward()
    gc = [p.grad for p in (fc1.weight, fc2.weight, qkv.weight)] + [xc.grad]
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    assert rel(yc, ye) < 1e-2  # inductor's own random stream / fused elementwise ops: rounding-level only
    assert all(rel(a, b) < 0.25 for a, b in zip(gc, ge)) and all(torch.isfinite(a).all() for a in gc)


# --- CUDA graphs: whole training steps -----------------------------------------------------------------

def _mlp():
    return torch.nn.Sequential(NVFP4Linear(256, 512, bias=False), torch.nn.GELU(),
                               NVFP4Linear(512, 256, bias=False)).to(DEV)


def _warmup(fn, n=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n):
            fn()
    torch.cuda.current_stream().wait_stream(s)


def test_cuda_graph_backward_draws_fresh_seeds_each_replay():
    torch.manual_seed(2)
    m = _mlp()
    ref_m = torch.nn.Sequential(torch.nn.Linear(256, 512, bias=False), torch.nn.GELU(),
                                torch.nn.Linear(512, 256, bias=False)).to(DEV)
    ref_m.load_state_dict(m.state_dict())
    x, dy = torch.randn(512, 256, device=DEV), torch.randn(512, 256, device=DEV)
    ref_m(x).backward(dy)
    exact = ref_m[0].weight.grad

    def step():
        m.zero_grad(set_to_none=True)
        m(x).backward(dy)

    _warmup(step)
    m.zero_grad(set_to_none=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        m(x).backward(dy)
    grads = []
    for _ in range(32):
        g.replay()
        grads.append(m[0].weight.grad.clone())
    assert not torch.equal(grads[0], grads[1])  # new SR / RHT seeds on every replay
    e1 = (grads[0] - exact).norm() / exact.norm()
    e32 = (sum(grads) / len(grads) - exact).norm() / exact.norm()
    assert e32 < 0.6 * e1  # the replays' rounding noise is independent and zero-mean


def test_cuda_graph_full_training_step_learns():
    torch.manual_seed(3)
    m = _mlp()
    teacher = torch.nn.Linear(256, 256, bias=False).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, capturable=True)
    x = torch.randn(512, 256, device=DEV)
    with torch.no_grad():
        target = teacher(x)
    static_loss = torch.zeros((), device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(m(x).float(), target)
        loss.backward()
        opt.step()
        return loss

    _warmup(step)
    opt.zero_grad(set_to_none=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = step()
    losses = []
    for _ in range(200):
        g.replay()
        losses.append(static_loss.item())
    assert all(map(lambda v: v == v, losses))
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


# --- fused wgrad accumulation keeps autograd's contract ----------------------------------------------------

def test_fused_accumulation_in_place_and_hooks_fire():
    torch.manual_seed(4)
    lin = NVFP4Linear(256, 384, bias=False, fuse_wgrad_accumulation=True).to(DEV)
    plain = NVFP4Linear(256, 384, bias=False).to(DEV)
    plain.load_state_dict(lin.state_dict())
    fired = []
    lin.weight.register_post_accumulate_grad_hook(lambda p: fired.append(p.grad.data_ptr()))
    xs = [torch.randn(200, 256, device=DEV) for _ in range(3)]
    dys = [torch.randn(200, 384, device=DEV) for _ in range(3)]
    for layer in (lin, plain):
        torch.manual_seed(11)
        for x, dy in zip(xs, dys):
            layer(x).backward(dy)
    assert len(fired) == 3                      # the hook fires on every micro-batch (DDP / FSDP rely on it)
    assert fired[1] == fired[2] == lin.weight.grad.data_ptr()  # accumulated in place, never copied
    assert torch.equal(lin.weight.grad, plain.weight.grad)


def test_fused_accumulation_with_autograd_grad():
    """torch.autograd.grad does not accumulate: it returns this call's gradient and leaves .grad alone."""
    torch.manual_seed(5)
    lin = NVFP4Linear(256, 256, bias=False, fuse_wgrad_accumulation=True).to(DEV)
    plain = NVFP4Linear(256, 256, bias=False).to(DEV)
    plain.load_state_dict(lin.state_dict())
    lin.weight.grad = torch.ones_like(lin.weight)
    x = torch.randn(128, 256, device=DEV)
    torch.manual_seed(1)
    (g,) = torch.autograd.grad(lin(x).float().sum(), lin.weight)
    torch.manual_seed(1)
    (gp,) = torch.autograd.grad(plain(x).float().sum(), plain.weight)
    assert torch.equal(g, gp)
    assert torch.equal(lin.weight.grad, torch.ones_like(lin.weight))


def test_outputs_are_not_views():
    """Like nn.Linear: FSDP2 warns about (and in-place ops on) module outputs that are views."""
    ln, fc1, fc2, qkv = _block()
    x = torch.randn(2, 64, 256, device=DEV, requires_grad=True)
    lin = NVFP4Linear(256, 256, bias=False).to(DEV)
    for y in (lin(x), qkv(x), layernorm_mlp(x, ln, fc1, fc2)):
        assert y.shape == (2, 64, 256) and not y._is_view()
    y = lin(x)
    y += 1  # in-place on the output is fine and keeps the graph
    y.float().sum().backward()
    assert x.grad is not None


# --- multi-process DDP / FSDP2 -----------------------------------------------------------------------

N_MB = 3  # micro-batches per step; the first N_MB - 1 skip gradient synchronization


def _data(r, mb):
    g = torch.Generator(device=DEV).manual_seed(100 * r + mb)
    return (torch.randn(192, 256, device=DEV, generator=g), torch.randn(192, 256, device=DEV, generator=g))


def _model(fuse):
    torch.manual_seed(0)  # identical initialization on every rank
    return torch.nn.Sequential(NVFP4Linear(256, 512, bias=False, fuse_wgrad_accumulation=fuse), torch.nn.GELU(),
                               NVFP4Linear(512, 256, bias=False, fuse_wgrad_accumulation=fuse)).to(DEV)


def _expected(world):
    """Average over ranks of each rank's locally accumulated gradient (plain autograd, no fusion)."""
    per_rank = []
    for r in range(world):
        m = _model(False)
        for mb in range(N_MB):
            x, dy = _data(r, mb)
            torch.manual_seed(1000 + mb)
            m(x).backward(dy)
        per_rank.append([p.grad for p in m.parameters()])
    return [sum(g[i] for g in per_rank) / world for i in range(len(per_rank[0]))]


def _count_fusions():
    import nvfp4.linear as L
    hits = [0]
    orig = L._accumulation_target

    def counting(ctx, param):
        t = orig(ctx, param)
        hits[0] += t is not None
        return t
    L._accumulation_target = counting
    return hits


def _check_ddp(rank, world, bucket_view):
    from torch.nn.parallel import DistributedDataParallel as DDP
    expected = _expected(world)
    hits = _count_fusions()
    for fuse in (False, True):
        hits[0] = 0
        m = _model(fuse)
        ddp = DDP(m, gradient_as_bucket_view=bucket_view)
        for _step in range(2):
            m.zero_grad(set_to_none=True)
            for mb in range(N_MB):
                x, dy = _data(rank, mb)
                torch.manual_seed(1000 + mb)
                with ddp.no_sync() if mb < N_MB - 1 else contextlib.nullcontext():
                    ddp(x).backward(dy)
            got = [p.grad for p in m.parameters()]
            assert all(torch.equal(a, b) for a, b in zip(got, expected)), (rank, fuse, bucket_view)
        assert hits[0] == (2 * (N_MB - 1) * 2 if fuse else 0), hits[0]  # 2 layers x 2 micro-batches x 2 steps


def _gather_dim0(shard):
    """Concatenate the ranks' dim-0 shards. (DTensor.full_tensor() over gloo with CUDA tensors crashes
    in PyTorch 2.14 even for plain nn.Linear, so gather through CPU tensors.)"""
    import torch.distributed as dist
    parts = [torch.empty_like(shard, device="cpu") for _ in range(dist.get_world_size())]
    dist.all_gather(parts, shard.cpu())
    return torch.cat(parts).to(shard.device)


def _check_fsdp2(rank, world):
    from torch.distributed.fsdp import fully_shard
    expected = _expected(world)
    for fuse in (False, True):
        m = _model(fuse)
        names = [n for n, _ in m.named_parameters()]
        for layer in (m[0], m[2]):
            fully_shard(layer)
        fully_shard(m)
        for mb in range(N_MB):
            m.set_requires_gradient_sync(mb == N_MB - 1)
            x, dy = _data(rank, mb)
            torch.manual_seed(1000 + mb)
            m(x).backward(dy)
        got = [_gather_dim0(p.grad.to_local()) for p in m.parameters()]
        assert names == [n for n, _ in m.named_parameters()]
        for a, b in zip(got, expected):
            assert torch.equal(a, b), (rank, fuse, (a - b).abs().max().item())


def _dist_worker(rank, world, port, mode):
    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.set_num_threads(2)  # gloo reduces on the CPU; a full OpenMP pool per rank only adds load
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        if mode == "fsdp2":
            _check_fsdp2(rank, world)
        else:
            _check_ddp(rank, world, bucket_view=mode == "ddp_bucket_view")
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("mode", ["ddp", "ddp_bucket_view", "fsdp2"])
def test_two_process_data_parallel(mode):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    os.environ.setdefault("OMP_NUM_THREADS", "2")  # inherited by the spawned ranks
    mp.spawn(_dist_worker, args=(2, _free_port(), mode), nprocs=2, join=True)
