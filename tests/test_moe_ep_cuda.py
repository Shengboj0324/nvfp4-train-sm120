"""Expert parallelism with variable-size exchanges (v0.17): the NVFP4 wire format, the router, chunked
(wavefront) exchanges and nodes (two-stage exchanges), checked against the single-process layer."""
import os

import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops, reference
from nvfp4.moe import MoE, _decode, _encode, _route
from nvfp4 import convert_linears

pytestmark = requires_sm120
DEV = "cuda"


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


# --- kernels ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 37, 300])
def test_dequantize_matches_reference(dtype, rows):
    g = torch.Generator(device=DEV).manual_seed(rows)
    x = torch.randn(rows, 512, device=DEV, generator=g) * 3
    x[:, :16] = 0  # an all-zero block
    q = ops.quantize(x)
    want = reference.dequantize(q.data, q.scales, q.gdec.reshape(())).to(dtype)
    got = ops.dequantize(q.data, q.scales, q.gdec, dtype)
    assert got.dtype == dtype and torch.equal(got, want)


def test_dequantize_strided_rows_and_per_row_scales():
    """Rows inside a wider message, each with its own tensor decode scale."""
    xs = [torch.randn(5, 256, device=DEV) * s for s in (1.0, 100.0)]
    qs = [ops.quantize(x) for x in xs]
    tail = [q.gdec.view(1, 1).expand(5, 1) for q in qs]
    msg = torch.cat([torch.cat([q.data, q.scales, t.contiguous().view(torch.uint8), torch.zeros(5, 4, device=DEV,
                                dtype=torch.uint8)], 1) for q, t in zip(qs, tail)])
    o = 128 + 16
    got = ops.dequantize(msg[:, :128], msg[:, 128:o], msg[:, o:o + 4].view(torch.float32)[:, 0])
    want = torch.cat([reference.dequantize(q.data, q.scales, q.gdec.reshape(())).bfloat16() for q in qs])
    assert torch.equal(got, want)


def test_amax_floor_sets_the_tensor_scale():
    x = torch.randn(64, 256, device=DEV)
    floor = (x.abs().amax().reshape(1) * 4.3).bfloat16().float()  # bf16-representable: compared via a bf16 row
    row, _ = ops.quantize_dual(x.bfloat16(), col=False, amax_floor=floor)
    # the same as quantizing x with an extra row that holds the floor
    ext = torch.cat([x, torch.zeros(1, 256, device=DEV)])
    ext[-1, 0] = floor.item()
    want, _ = ops.quantize_dual(ext.bfloat16(), col=False)
    assert torch.equal(row.gdec, want.gdec)
    assert torch.equal(row.data, want.data[:64]) and torch.equal(row.scales, want.scales[:64])
    # a floor below the amax changes nothing; with bias_grad (the parts path) too
    plain, _ = ops.quantize_dual(x.bfloat16(), col=False)
    low, _ = ops.quantize_dual(x.bfloat16(), col=False, amax_floor=floor / 100)
    assert torch.equal(plain.data, low.data) and torch.equal(plain.gdec, low.gdec)
    parts = ops.quantize_dual(x.bfloat16(), col=False, amax_floor=floor, bias_grad=True)
    assert torch.equal(parts[0].data, row.data) and torch.equal(parts[0].gdec, row.gdec)


@pytest.mark.parametrize("four_six", [False, True])
def test_wire_round_trip_reproduces_the_codes(four_six):
    """Rows sent as NVFP4 with the group's amax, dequantized to bf16 and quantized again with that amax as the
    floor give the codes and scales of quantizing the original rows directly: the experts' forward is the same
    as with bf16 messages."""
    g = torch.Generator(device=DEV).manual_seed(4)
    x = (torch.randn(300, 512, device=DEV, generator=g) * torch.rand(300, 1, device=DEV, generator=g) * 5).bfloat16()
    amax = x.abs().amax().float().reshape(1) * 1.7  # another rank's larger amax
    with ops.four_six_scales(four_six):
        msg = _encode(x, "nvfp4", amax)
        assert msg.shape == (300, 512 * 9 // 16 + 8)
        y, got_amax, o = _decode(msg, 0, 512, torch.bfloat16, "nvfp4")
        assert o == msg.shape[1] and torch.equal(got_amax, amax)
        direct, _ = ops.quantize_dual(x, col=False, amax_floor=amax)
        again, _ = ops.quantize_dual(y, col=False, amax_floor=amax)
    assert torch.equal(direct.data, again.data) and torch.equal(direct.scales, again.scales)
    assert torch.equal(direct.gdec, again.gdec)


def test_sm_margin_keeps_gemm_results():
    """Fewer persistent CTAs (SMs left to concurrent communication kernels): the same bits."""
    a = ops.quantize(torch.randn(2048, 1024, device=DEV))
    b = ops.quantize(torch.randn(1536, 1024, device=DEV))
    tiles = torch.tensor([0] * 8 + [1] * 7 + [-1], device=DEV, dtype=torch.int32)
    bg = ops.quantize(torch.randn(2 * 768, 1024, device=DEV))
    want, want_g = ops.gemm(a, b), ops.gemm_grouped(a, bg, "m", tiles, group_rows=768)
    try:
        for n in (8, 150):
            ops.set_sm_margin(n)
            assert ops.sm_margin() == n
            assert torch.equal(ops.gemm(a, b), want)
            assert torch.equal(ops.gemm_grouped(a, bg, "m", tiles, group_rows=768), want_g)
    finally:
        ops.set_sm_margin(0)
    with pytest.raises(ValueError):
        ops.set_sm_margin(-1)


def test_route_once_per_member():
    ids = torch.tensor([[0, 5], [1, 1], [-1, 6], [7, 4], [-1, -1]], device=DEV)
    r = _route(ids, members=2, per=4, n=5)  # experts 0-3 on member 0, 4-7 on member 1
    # (row, member) pairs: (0,0) (1,0) | (0,1) (2,1) (3,1); row 1's experts 1, 1 and row 3's 7, 4 go once
    assert r.rows.tolist() == [0, 1, 0, 2, 3]
    assert r.src.tolist() == [0, 1, 0, 2, 3, 5]
    assert r.ids.tolist() == [[0, -1], [1, 1], [-1, 1], [-1, 2], [3, 0]]
    assert r.mask.tolist() == [[True, False], [True, True], [False, True], [False, True], [True, True]]
    assert r.pos.tolist() == [[0, 2], [1, 5], [5, 3], [4, 5], [5, 5]]


# --- expert parallelism across processes (gloo, one GPU) ------------------------------------------------------

E, K = 8, 3


def _pair(world, rank, group, fp4, cf, node=1, seed=3):
    """(the single-process reference with all experts, this rank's expert-parallel layer with its slice)."""
    from test_moe_features_cuda import _moe
    ref = _moe("swiglu", d=256, d_ff=256, E=E, k=K, seed=seed, capacity_factor=cf, shared_d_ff=256, bias=True)
    with torch.no_grad():
        ref.router.weight[0] += 0.3 * ref.router.weight[1]  # skewed: the ranks get different numbers of rows
    torch.manual_seed(seed)
    ep = MoE(256, 256, E, top_k=K, activation="swiglu", bias=True, capacity_factor=cf, shared_d_ff=256,
             ep_group=group, ep_node_size=node).to(DEV)
    El = E // world
    with torch.no_grad():
        for n, p in ep.named_parameters():
            q = dict(ref.named_parameters())[n]
            p.copy_(q[rank * El:(rank + 1) * El] if n.startswith(("w_", "b_")) else q)
    for m in (ref, ep):
        if fp4:
            convert_linears(m)
    return ref, ep


def _data(r):
    g = torch.Generator(device=DEV).manual_seed(50 + r)
    return torch.randn(150, 256, device=DEV, generator=g), torch.randn(150, 256, device=DEV, generator=g)


def _check_fp32(rank, world, group, cf, node, chunks):
    """fp32 experts: outputs and every gradient equal the single-process layer's (up to summation order)."""
    ref, ep = _pair(world, rank, group, False, cf, node)
    ep.ep_dispatch, ep.ep_chunks = "variable", chunks
    El = E // world
    ref_grads = {n: torch.zeros_like(p) for n, p in ref.named_parameters()}
    for r in range(world):
        x, g = _data(r)
        ref.zero_grad(set_to_none=True)
        xr = x.clone().requires_grad_()
        yr = ref(xr)
        yr.backward(g)
        for n, p in ref.named_parameters():
            ref_grads[n] += p.grad
        if r == rank:
            y_want, dx_want, local = yr.detach(), xr.grad, {n: p.grad.clone() for n, p in ref.named_parameters()}
    x, g = _data(rank)
    xe = x.clone().requires_grad_()
    ye = ep(xe)
    ye.backward(g)
    tol = dict(rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ye, y_want, **tol)
    torch.testing.assert_close(xe.grad, dx_want, **tol)
    for n, p in ep.named_parameters():
        want = ref_grads[n][rank * El:(rank + 1) * El] if n.startswith(("w_", "b_")) else local[n]
        torch.testing.assert_close(p.grad, want, **tol, msg=lambda m: f"{n}: {m}")


def _run_nvfp4(ep, x, g, seed=11):
    torch.manual_seed(seed)
    xe = x.clone().requires_grad_()
    ep.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = ep(xe)
    y.backward(g.bfloat16())
    return y.detach(), xe.grad, {n: p.grad.clone() for n, p in ep.named_parameters()}


def _check_nvfp4(rank, world, group, node):
    """NVFP4 experts: with NVFP4 activation messages the forward and the input gradient are bit-identical to
    bf16 messages (the receiver re-quantizes the dequantized rows to the same codes; only the weight gradients'
    token-wise quantization sees the dequantized rows); NVFP4 gradient messages, other chunkings (each chunk's
    buffer has its own hidden-activation tensor scale) and the single-process layer are compared by cosine."""
    ref, ep = _pair(world, rank, group, True, None, node)
    ep.ep_dispatch = "variable"
    x, g = _data(rank)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = ref(x)
    first = None
    for chunks in (1, 3):
        ep.ep_chunks, ep.ep_wire, ep.ep_wire_grad = chunks, "bf16", "bf16"
        base = _run_nvfp4(ep, x, g)
        assert _cos(base[0], y_ref) > 0.97
        if first is None:
            first = base
        else:
            cy, cx = _cos(base[0], first[0]), _cos(base[1], first[1])
            assert cy > 0.99 and cx > 0.95, (cy, cx)  # dx: other stochastic-rounding draws
        for wire_grad in ("bf16", "nvfp4"):
            ep.ep_wire, ep.ep_wire_grad = "nvfp4", wire_grad
            y, dx, grads = _run_nvfp4(ep, x, g)
            assert torch.equal(y, base[0]), (chunks, wire_grad)
            if wire_grad == "bf16":
                assert torch.equal(dx, base[1]), chunks
            else:
                assert _cos(dx, base[1]) > 0.98, (chunks, _cos(dx, base[1]))
            for n, gr in grads.items():
                assert _cos(gr, base[2][n]) > 0.95, (n, chunks, wire_grad, _cos(gr, base[2][n]))


def _worker(rank, world, port, configs, backend="gloo"):
    import datetime

    import torch.distributed as dist
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(port)
    torch.set_num_threads(2)
    torch.cuda.set_device(rank if backend == "nccl" else 0)
    dist.init_process_group(backend, rank=rank, world_size=world, timeout=datetime.timedelta(minutes=4),
                            device_id=torch.device("cuda", rank) if backend == "nccl" else None)
    try:
        for kind, *args in configs:
            if kind == "fp32":
                _check_fp32(rank, world, dist.group.WORLD, *args)
            else:
                _check_nvfp4(rank, world, dist.group.WORLD, *args)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(world, configs, backend="gloo"):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from test_integration_cuda import _free_port
    if not dist.is_available() or not (dist.is_gloo_available() if backend == "gloo" else dist.is_nccl_available()):
        pytest.skip(f"{backend} not available")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    mp.spawn(_worker, args=(world, _free_port(), configs, backend), nprocs=world, join=True)


def test_variable_ep_two_ranks():
    _spawn(2, [("fp32", None, 1, 1), ("fp32", 0.75, 1, 3), ("fp32", None, 2, 2), ("fp32", None, 1, 7),
               ("nvfp4", 1), ("nvfp4", 2)])


def test_variable_ep_two_nodes_of_two():
    _spawn(4, [("fp32", None, 2, 1), ("fp32", 0.75, 2, 2), ("fp32", None, 1, 2), ("nvfp4", 2)])


def test_multi_gpu_nccl_variable_ep():
    """The same checks with one process per GPU over NCCL: two GPUs, and with four also two nodes of two."""
    n = torch.cuda.device_count()
    if n < 2 or not all(ops.available(i) for i in range(n)):
        pytest.skip("needs two sm_120 GPUs")
    if n >= 4:
        _spawn(4, [("fp32", None, 2, 1), ("fp32", None, 1, 3), ("fp32", 0.75, 2, 2), ("nvfp4", 2), ("nvfp4", 1)],
               "nccl")
    else:
        _spawn(2, [("fp32", None, 1, 1), ("fp32", 0.75, 1, 3), ("fp32", None, 2, 2), ("nvfp4", 1), ("nvfp4", 2)],
               "nccl")
