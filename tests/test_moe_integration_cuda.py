"""nvfp4.moe.MoE with NVFP4 experts in the training stack: no host synchronization in a forward + backward,
CUDA-graph capture (forward replays bit-identical to eager under changing routing; whole training steps with
a capturable optimizer learn), torch.compile with fullgraph=True (no graph breaks; bit-identical to eager),
activation checkpointing (bit-identical gradients), and real two-process DDP / FSDP1 / FSDP2 over gloo
(gradients bit-identical to the average of the ranks' single-process gradients, with and without fused
accumulation)."""
import contextlib
import copy
import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4 import convert_linears
from nvfp4.moe import MoE

pytestmark = requires_sm120
DEV = "cuda"
ACTS = ["swiglu", "gelu_tanh"]


def _moe(activation="swiglu", fuse=False, E=4, k=2, d=256, d_ff=256, seed=0):
    torch.manual_seed(seed)
    m = MoE(d, d_ff, E, top_k=k, activation=activation).to(DEV)
    convert_linears(m, fuse_wgrad_accumulation=fuse)
    return m


def _grads(m):
    return [p.grad.clone() for p in m.parameters()]


@pytest.mark.parametrize("activation", ACTS)
def test_forward_backward_has_no_host_sync(activation):
    m = _moe(activation)
    x = torch.randn(300, 256, device=DEV, requires_grad=True)
    m(x)  # warm-up (kernel attributes, tensor-map encoder lookup)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")  # any host synchronization raises
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
        (y.float().square().mean() + 0.01 * m.aux_loss).backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert all(torch.isfinite(p.grad).all() for p in m.parameters()) and torch.isfinite(x.grad).all()


def _warmup(fn, n=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n):
            fn()
    torch.cuda.current_stream().wait_stream(s)


@pytest.mark.parametrize("activation", ACTS)
def test_cuda_graph_forward_replays_match_eager_under_changing_routing(activation):
    """The captured forward serves any routing of the same number of tokens: each replay reads new inputs
    (a different split over the experts, including experts without tokens) and equals eager bit for bit
    (the forward has no stochastic rounding)."""
    m = _moe(activation, E=8)
    static_x = torch.randn(512, 256, device=DEV)

    def fwd():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return m(static_x)

    _warmup(fwd)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_y = fwd()
    counts = []
    for i in range(6):
        torch.manual_seed(100 + i)
        x = torch.randn(512, 256, device=DEV)
        if i % 2:  # skew the routing: every token to experts i and i + 1, the other six get none
            w = m.router.weight.detach()
            x = 0.1 * x + 50 * (w[i] + w[i + 1])
        static_x.copy_(x)
        g.replay()
        want = fwd()
        assert torch.equal(static_y, want), i
        counts.append(m.last_counts)
    assert len({tuple(c) for c in counts}) == len(counts)  # every replay routed differently
    assert any(0 in c for c in counts)


@pytest.mark.parametrize("activation", ACTS)
def test_cuda_graph_full_training_step_learns(activation):
    m = _moe(activation, fuse=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, capturable=True)
    torch.manual_seed(1)
    teacher = torch.nn.Linear(256, 256, bias=False).to(DEV)
    x = torch.randn(512, 256, device=DEV)
    with torch.no_grad():
        target = teacher(x)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
        loss = F.mse_loss(y.float(), target) + 0.01 * m.aux_loss
        loss.backward()
        opt.step()
        return loss

    _warmup(step)
    opt.zero_grad(set_to_none=True)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_loss = step()
    losses = []
    for _ in range(150):
        g.replay()
        losses.append(static_loss.item())
    assert all(v == v for v in losses)
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


@pytest.mark.parametrize("activation", ACTS)
def test_compile_fullgraph_matches_eager(activation):
    """No graph breaks (fullgraph=True raises on one). With inductor drawing random numbers like eager
    (fallback_random) and rounding at every bf16 cast like eager (emulate_precision_casts: by default its fused
    kernels keep fp32 intermediates, and a slightly different bf16 gradient moves the NVFP4 tensor scale and with
    it every rounding downstream), the output and the expert weight gradients are bit-identical to eager; the
    load-balancing loss and the input gradient differ by fp32 summation order only (inductor's reductions)."""
    import torch._inductor.config as inductor_config
    m = _moe(activation)
    x = torch.randn(300, 256, device=DEV)

    def f(mod, xg):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = mod(xg)
        return y, mod.aux_loss

    def run(fn):
        xg = x.clone().requires_grad_()
        m.zero_grad(set_to_none=True)
        torch.manual_seed(7)
        y, aux = fn(m, xg)
        (y.float().square().sum() + aux).backward()
        return [y.detach(), aux.detach(), xg.grad] + _grads(m)

    eager = run(f)
    torch._dynamo.reset()
    with inductor_config.patch(fallback_random=True, emulate_precision_casts=True):
        compiled = run(torch.compile(f, fullgraph=True, backend=COMPILE_BACKEND))
    for i, (a, b) in enumerate(zip(eager, compiled)):
        if i in (1, 2):  # aux loss, input gradient (a few elements one bf16 step apart after a near-tie)
            assert ((a - b).norm() / a.norm()).item() < 1e-5, i
        else:
            assert torch.equal(a, b), i


@pytest.mark.parametrize("activation", ACTS)
def test_activation_checkpointing_gives_identical_gradients(activation):
    """torch.utils.checkpoint recomputes the layer in the backward with the forward's RNG state, so the
    rounding seeds, the routing and therefore every gradient match the plain run bit for bit."""
    m = _moe(activation)
    x = torch.randn(300, 256, device=DEV)

    def run(ckpt):
        xg = x.clone().requires_grad_()
        m.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = checkpoint(m, xg, use_reentrant=False) if ckpt else m(xg)
        y.float().square().sum().backward()
        return [y.detach(), xg.grad] + _grads(m)

    plain, ck = run(False), run(True)
    for i, (a, b) in enumerate(zip(plain, ck)):
        assert torch.equal(a, b), i


# --- two-process DDP / FSDP (gloo; both ranks on the one GPU) ----------------------------------------

N_MB = 3  # micro-batches per step; the first N_MB - 1 skip gradient synchronization


def _data(r, mb):
    g = torch.Generator(device=DEV).manual_seed(100 * r + mb)
    return torch.randn(192, 256, device=DEV, generator=g), torch.randn(192, 256, device=DEV, generator=g)


def _step(net, rank, mb, act_ctx=contextlib.nullcontext):
    x, dy = _data(rank, mb)
    torch.manual_seed(1000 + mb)
    with act_ctx():
        net(x).backward(dy)


def _expected(world, activation):
    """Average over ranks of each rank's locally accumulated gradient (single process, no fusion)."""
    per_rank = []
    for r in range(world):
        m = _moe(activation)
        for mb in range(N_MB):
            _step(m, r, mb)
        per_rank.append(_grads(m))
    return [sum(g[i] for g in per_rank) / world for i in range(len(per_rank[0]))]


def _count_fusions():
    import nvfp4.moe as M
    hits = [0]
    orig = M._accumulation_target

    def counting(ctx, param):
        t = orig(ctx, param)
        hits[0] += t is not None
        return t
    M._accumulation_target = counting
    return hits


def _check_ddp(rank, world, activation):
    from torch.nn.parallel import DistributedDataParallel as DDP
    expected = _expected(world, activation)
    hits = _count_fusions()
    for fuse in (False, True):
        hits[0] = 0
        m = _moe(activation, fuse)
        ddp = DDP(m)
        for _ in range(2):
            m.zero_grad(set_to_none=True)
            for mb in range(N_MB):
                _step(ddp, rank, mb, ddp.no_sync if mb < N_MB - 1 else contextlib.nullcontext)
            for i, (a, b) in enumerate(zip(_grads(m), expected)):
                assert torch.equal(a, b), (rank, activation, fuse, i, (a - b).abs().max().item())
        n_w = len(m.experts_weights())
        assert hits[0] == (2 * (N_MB - 1) * n_w if fuse else 0), hits[0]  # x micro-batches after the first x steps


def _check_fsdp2(rank, world, activation):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard
    from test_integration_cuda import _fsdp2_reduction_scale, _gather_dim0
    mesh = DeviceMesh.from_group(dist.group.WORLD, "cuda")
    scale = _fsdp2_reduction_scale(mesh, world)
    expected = [e * scale for e in _expected(world, activation)]
    for fuse in (False, True):
        m = _moe(activation, fuse)
        fully_shard(m, mesh=mesh)  # stacked experts and the router sharded along dim 0
        for mb in range(N_MB):
            m.set_requires_gradient_sync(mb == N_MB - 1)
            _step(m, rank, mb)
        got = [_gather_dim0(p.grad.to_local()) for p in m.parameters()]
        for i, (a, b) in enumerate(zip(got, expected)):
            assert torch.equal(a, b), (rank, activation, fuse, i, (a - b).abs().max().item())


def _check_fsdp1(rank, world, activation):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    expected = _expected(world, activation)
    names = [n for n, _ in _moe(activation).named_parameters()]
    for fuse in (False, True):
        m = _moe(activation, fuse)
        fsdp = FSDP(m, device_id=torch.cuda.current_device(), use_orig_params=True)
        for mb in range(N_MB):
            _step(fsdp, rank, mb, fsdp.no_sync if mb < N_MB - 1 else contextlib.nullcontext)
        with FSDP.summon_full_params(fsdp, with_grads=True):
            got = {n.replace("_fsdp_wrapped_module.", ""): p.grad.clone() for n, p in fsdp.named_parameters()}
        for n, b in zip(names, expected):
            assert torch.equal(got[n], b), (rank, activation, fuse, n, (got[n] - b).abs().max().item())


def _moe_dist_worker(rank, world, port, mode, activation):
    import datetime

    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    dist.init_process_group("gloo", rank=rank, world_size=world, timeout=datetime.timedelta(minutes=3))
    try:
        {"ddp": _check_ddp, "fsdp1": _check_fsdp1, "fsdp2": _check_fsdp2}[mode](rank, world, activation)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["ddp", "fsdp1", "fsdp2"])
@pytest.mark.parametrize("activation", ACTS)
def test_two_process_data_parallel(mode, activation):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from test_integration_cuda import _free_port
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    mp.spawn(_moe_dist_worker, args=(2, _free_port(), mode, activation), nprocs=2, join=True)


def test_deepcopy_and_state_dict_roundtrip():
    """Stacked parameters survive copy.deepcopy and a state-dict round trip (the NVFP4 flag is module
    state, not a parameter)."""
    import pickle
    m = _moe()
    m(torch.randn(64, 256, device=DEV, requires_grad=True)).sum().backward()  # aux_loss: a non-leaf tensor
    c = copy.deepcopy(m)
    assert c.nvfp4 and all(torch.equal(a, b) for a, b in zip(m.parameters(), c.parameters()))
    assert c.aux_loss is None and m.aux_loss is not None and c.last_counts is None
    assert pickle.loads(pickle.dumps(m)).nvfp4
    n = MoE(256, 256, 4, top_k=2).to(DEV)
    n.load_state_dict(m.state_dict())
    x = torch.randn(64, 256, device=DEV)
    convert_linears(n)
    assert torch.equal(n(x), m(x))
