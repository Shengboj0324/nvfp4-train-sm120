"""nvfp4.moe.MoE options of v0.15 / v0.16: capacity-limited routing (token dropping), the shared expert, and
expert parallelism over a process group, with a capacity or dropless (two ranks over gloo on one GPU), against
per-token references."""
import copy
import os

import pytest
import torch
import torch.nn.functional as F

from conftest import HAS_TRITON, requires_sm120
from nvfp4 import convert_linears
from nvfp4.moe import MoE

pytestmark = requires_sm120
DEV = "cuda"


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _mlp(moe, ws, bs, e, x):
    b = [None if t is None else t[e] for t in bs]
    if moe.activation == "swiglu":
        return F.linear(F.silu(F.linear(x, ws[0][e], b[0])) * F.linear(x, ws[1][e], b[1]), ws[2][e], b[2])
    approx = "tanh" if moe.activation == "gelu_tanh" else "none"
    return F.linear(F.gelu(F.linear(x, ws[0][e], b[0]), approximate=approx), ws[1][e], b[1])


def _reference(moe, x2):
    """Per-token reference: slot (t, j) is kept if it is among its expert's first C slots in (t, j) order."""
    w, idx, _ = moe.route(x2)
    T, k = idx.shape
    C = moe.capacity(T)
    seen = [0] * moe.n_experts
    out = torch.zeros_like(x2)
    dropped = 0
    for t in range(T):
        for j in range(k):
            e = int(idx[t, j])
            seen[e] += 1
            if C is None or seen[e] <= C:
                out[t] += w[t, j] * _mlp(moe, moe.experts_weights(), moe.experts_biases(), e, x2[t:t + 1])[0]
            else:
                dropped += 1
    if moe.shared_d_ff:
        out += _mlp(moe, moe.shared_weights(), moe.shared_biases(), 0, x2)
    return out, dropped


def _moe(activation="swiglu", E=4, k=2, d=256, d_ff=256, seed=0, **kw):
    torch.manual_seed(seed)
    return MoE(d, d_ff, E, top_k=k, activation=activation, **kw).to(DEV)


@pytest.mark.parametrize("impl", ["loop", "grouped"])
@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
@pytest.mark.parametrize("cf,shared,bias", [(0.5, 0, False), (1.0, 384, True), (None, 128, False)])
def test_capacity_and_shared_match_per_token_reference(impl, activation, cf, shared, bias):
    if impl == "grouped" and not HAS_TRITON:
        pytest.skip("needs Triton")
    """fp32 experts: the capacity rule (first C slots per expert in token order; dropped slots add nothing)
    and the shared expert (a dense MLP added with weight 1), both high-precision paths."""
    moe = _moe(activation, capacity_factor=cf, shared_d_ff=shared, bias=bias)
    moe.bf16_impl = impl
    x = torch.randn(150, 256, device=DEV)
    with torch.no_grad():  # skew the routing so that capacity matters
        moe.router.weight[0] += 0.3 * moe.router.weight[1]
    y = moe(x)
    want, dropped = _reference(moe, x)
    torch.testing.assert_close(y, want, rtol=1e-4, atol=1e-5)
    assert moe.last_dropped == dropped and (cf != 0.5 or dropped > 0)
    if cf is not None:
        C = moe.capacity(150)
        assert dropped == sum(max(0, c - C) for c in moe.last_counts)


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_nvfp4_capacity_and_shared(activation):
    """NVFP4 experts with capacity and a shared expert track the BF16 ones; a capacity no slot reaches
    gives the dropless forward bit for bit (the forward has no stochastic rounding); gradients reach every
    parameter, and a token whose every slot was dropped gets only the shared expert's output."""
    ref = _moe(activation, E=4, d=512, d_ff=512, capacity_factor=0.75, shared_d_ff=512, bias=True)
    ref.bf16_impl = "loop"
    moe = copy.deepcopy(ref)
    convert_linears(moe)
    x = torch.randn(400, 512, device=DEV, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y, y_ref = moe(x), ref(x2)
    assert moe.last_dropped == ref.last_dropped > 0
    assert _cos(y, y_ref) > 0.96
    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert _cos(x.grad, x2.grad) > 0.95
    for (n, p), (_, q) in zip(moe.named_parameters(), ref.named_parameters()):
        assert p.grad is not None and _cos(p.grad, q.grad) > 0.9, n
    # every slot of token 0 dropped: route all tokens to expert 0 and 1 with capacity 1 -> only token 0 kept
    big, small = _moe(activation, capacity_factor=8.0), _moe(activation)  # C = 8 T k / E >= T k: no drops
    small.load_state_dict(big.state_dict())
    for m in (big, small):
        convert_linears(m)
    xs = torch.randn(256, 256, device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(big(xs), small(xs)) and big.last_dropped == 0
    sh = _moe(activation, capacity_factor=1 / 128, shared_d_ff=256)  # C = ceil(256 * 2 / 4 / 128) = 1
    convert_linears(sh)
    with torch.no_grad():
        sh.router.weight.zero_()  # ties -> experts 0, 1 for every token: only token 0's slots are kept
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = sh(xs)
            only_shared = sh._shared(xs).to(out.dtype)
    assert sh.capacity(256) == 1 and sh.last_dropped == 2 * 256 - 2
    assert torch.equal(out[1:], only_shared[1:]) and not torch.equal(out[0], only_shared[0])


def test_capacity_validation_and_repr():
    with pytest.raises(ValueError):
        MoE(256, 256, 4, capacity_factor=0.0)
    with pytest.raises(ValueError):
        MoE(256, 256, 4, shared_d_ff=-1)
    with pytest.raises(ValueError):
        MoE(256, 256, 4, shared_d_ff=200).to_nvfp4()
    m = MoE(256, 256, 4, capacity_factor=1.25, shared_d_ff=128, bias=True)
    assert m.capacity(100) == 63 and "capacity_factor=1.25" in repr(m) and "shared_d_ff=128" in repr(m)
    assert [n for n, _ in m.named_parameters()][-7:] == ["shared_w_gate", "shared_b_gate", "shared_w_up",
                                                         "shared_b_up", "shared_w_down", "shared_b_down",
                                                         "router.weight"]
    assert m.shared_w_down.shape == (1, 256, 128) and len(m.shared_weights()) == 3
    assert MoE(256, 256, 4).last_dropped is None


# --- expert parallelism: two ranks over gloo on the one GPU --------------------------------------------

def _ep_pair(world, rank, group, fp4, cf, seed=3):
    """(reference MoE with all experts, the EP MoE of this rank holding its slice of the same weights; the
    routing skewed toward expert 0, so the ranks receive different numbers of rows)."""
    kw = dict(d=256, d_ff=256, E=4, k=2, seed=seed, capacity_factor=cf, shared_d_ff=256, bias=True)
    ref = _moe("swiglu", **kw)
    with torch.no_grad():
        ref.router.weight[0] += 0.3 * ref.router.weight[1]
    torch.manual_seed(seed)
    ep = MoE(256, 256, 4, top_k=2, activation="swiglu", bias=True, capacity_factor=cf, shared_d_ff=256,
             ep_group=group).to(DEV)
    El = 4 // world
    with torch.no_grad():
        for n, p in ep.named_parameters():
            q = dict(ref.named_parameters())[n]
            p.copy_(q[rank * El:(rank + 1) * El] if n.startswith(("w_", "b_")) else q)
    for m in (ref, ep):
        if fp4:
            convert_linears(m)
    return ref, ep


def _ep_data(r):
    g = torch.Generator(device=DEV).manual_seed(50 + r)
    return torch.randn(200, 256, device=DEV, generator=g), torch.randn(200, 256, device=DEV, generator=g)


def _check_ep(rank, world, group, cf, overlap=True):
    import torch.distributed as dist
    # fp32 experts: the EP output on each rank equals the single-process layer on that rank's tokens; the
    # local experts' gradients are the sum over both ranks' tokens (the shared expert's backward runs while
    # the output gradients travel: its gradients and the router's are checked too)
    ref, ep = _ep_pair(world, rank, group, fp4=False, cf=cf)
    ep.ep_overlap = overlap
    El = 4 // world
    ref_grads = {n: torch.zeros_like(p) for n, p in ref.named_parameters()}
    for r in range(world):
        x, g = _ep_data(r)
        ref.zero_grad(set_to_none=True)
        xr = x.clone().requires_grad_()
        yr = ref(xr)
        yr.backward(g)
        for n, p in ref.named_parameters():
            ref_grads[n] += p.grad
        if r == rank:
            y_want, dx_want, router_want, dropped = yr.detach(), xr.grad, ref.router.weight.grad.clone(), \
                ref.last_dropped
            ref_local = {n: p.grad.clone() for n, p in ref.named_parameters()}
    x, g = _ep_data(rank)
    xe = x.clone().requires_grad_()
    ye = ep(xe)
    ye.backward(g)
    assert ep.last_dropped == dropped and (dropped > 0) == (cf is not None)
    if cf is None:  # skewed routing: the two ranks' experts get different numbers of rows
        c = torch.tensor(ep.last_counts, device=DEV)
        both = torch.empty(world * c.numel(), device=DEV, dtype=c.dtype)
        dist.all_to_all_single(both, c.repeat(world), group=group)  # (all_reduce of CUDA tensors crashes
        c = both.view(world, -1).sum(0)                              # gloo on PyTorch 2.11 Windows)
        assert c[:2].sum() != c[2:].sum()
    torch.testing.assert_close(ye, y_want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(xe.grad, dx_want, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ep.router.weight.grad, router_want, rtol=1e-4, atol=1e-5)
    for n, p in ep.named_parameters():
        if n.startswith(("w_", "b_")):  # local experts: every rank's tokens
            torch.testing.assert_close(p.grad, ref_grads[n][rank * El:(rank + 1) * El], rtol=1e-4, atol=1e-5)
        elif n.startswith("shared_"):  # replicated: this rank's tokens
            torch.testing.assert_close(p.grad, ref_local[n], rtol=1e-4, atol=1e-5)
    # NVFP4 experts: the same exchange (tensor scales now span both ranks' tokens: compared by cosine)
    ref, ep = _ep_pair(world, rank, group, fp4=True, cf=cf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = ref(x)
        y = ep(xe.detach())
    assert _cos(y, y_ref) > 0.97, _cos(y, y_ref)
    dist.barrier(group)


def _ep_worker(rank, world, port, backend="gloo"):
    import datetime

    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.set_num_threads(2)
    torch.cuda.set_device(rank if backend == "nccl" else 0)
    dist.init_process_group(backend, rank=rank, world_size=world, timeout=datetime.timedelta(minutes=3),
                            device_id=torch.device("cuda", rank) if backend == "nccl" else None)
    try:
        for cf in (0.75, None):
            _check_ep(rank, world, dist.group.WORLD, cf)
        # without the overlap: the same results
        _check_ep(rank, world, dist.group.WORLD, 0.75, overlap=False)
    finally:
        dist.destroy_process_group()


def test_expert_parallel_two_ranks():
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from test_integration_cuda import _free_port
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    mp.spawn(_ep_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_multi_gpu_nccl_expert_parallel():
    """The same checks with one process per GPU over NCCL (needs two sm_120 GPUs)."""
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from test_integration_cuda import _free_port

    from nvfp4 import ops as _ops
    if torch.cuda.device_count() < 2 or not all(_ops.available(i) for i in range(2)):
        pytest.skip("needs two sm_120 GPUs")
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("NCCL not available")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    mp.spawn(_ep_worker, args=(2, _free_port(), "nccl"), nprocs=2, join=True)


def test_expert_parallel_validation():
    class _G:  # a stand-in group: only the size and rank are read before the check
        pass
    import torch.distributed as dist
    orig = dist.get_world_size, dist.get_rank
    dist.get_world_size, dist.get_rank = (lambda g=None: 2), (lambda g=None: 0)
    try:
        with pytest.raises(ValueError):
            MoE(256, 256, 3, capacity_factor=1.0, ep_group=_G())
        assert MoE(256, 256, 4, ep_group=_G()).n_local == 2  # dropless expert parallelism
        m = MoE(256, 256, 4, capacity_factor=1.0, ep_group=_G())
        assert m.w_gate.shape[0] == 2 and m.router.weight.shape[0] == 4 and "expert_parallel=2" in repr(m)
    finally:
        dist.get_world_size, dist.get_rank = orig


@pytest.mark.parametrize("cf", [1.0, None])
def test_expert_parallel_overlap_order(monkeypatch, cf):
    """The shared expert's forward runs between the start and the end of the outputs' exchange, and its
    backward between the start and the end of the reverse exchange (one rank over gloo: the order of the
    calls, not the timing)."""
    import torch.distributed as dist
    from test_integration_cuda import _free_port

    from nvfp4 import moe as moe_mod
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("gloo not available")
    events = []
    for cls, tag in ((moe_mod._ExchangeStart, "start"), (moe_mod._ExchangeWait, "wait")):
        for fn in ("forward", "backward"):
            orig = getattr(cls, fn)

            def wrapped(ctx, *a, _orig=orig, _name=f"{tag}.{fn}"):
                events.append(_name)
                return _orig(ctx, *a)
            monkeypatch.setattr(cls, fn, staticmethod(wrapped))
    store = dist.TCPStore("127.0.0.1", _free_port(), 1, True, use_libuv=False)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        m = MoE(256, 256, 4, capacity_factor=cf, shared_d_ff=256, ep_group=dist.group.WORLD).to(DEV)
        orig_shared = m._shared

        def shared(x):
            events.append("shared.forward")
            return orig_shared(x)
        m._shared = shared
        m.shared_w_gate.register_hook(lambda g: events.append("shared.backward"))
        m(torch.randn(64, 256, device=DEV, requires_grad=True)).sum().backward()
    finally:
        dist.destroy_process_group()
    assert events == ["start.forward", "wait.forward",  # dispatch
                      "start.forward", "shared.forward", "wait.forward",  # outputs back, shared expert
                      "wait.backward", "shared.backward", "start.backward",  # output gradients out
                      "wait.backward", "start.backward"], events  # input gradients back
