"""nvfp4.moe.MoE: routing against a per-token reference loop (BF16 experts); NVFP4 experts (grouped GEMMs)
against BF16 ones (NVFP4 error level, forward and gradients) and against the same computation with a per-group
fp64 reference in place of the grouped GEMM (checks the padding, tile map, segments and the stacked transposed
weights, forward and backward); experts without tokens; fused gradient accumulation; the weight cache; the
BF16 tail; the load-balancing loss."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import convert_linears, ops
from nvfp4 import moe as moe_mod
from nvfp4.moe import MoE

pytestmark = requires_sm120


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _moe(activation="swiglu", E=4, k=2, d=256, d_ff=512, seed=0):
    torch.manual_seed(seed)
    return MoE(d, d_ff, E, top_k=k, activation=activation).cuda()


def _expert(moe, e, x):
    ws = moe.experts_weights()
    if moe.activation == "swiglu":
        return F.linear(F.silu(F.linear(x, ws[0][e])) * F.linear(x, ws[1][e]), ws[2][e])
    approx = "tanh" if moe.activation == "gelu_tanh" else "none"
    return F.linear(F.gelu(F.linear(x, ws[0][e]), approximate=approx), ws[1][e])


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh", "gelu"])
@pytest.mark.parametrize("k", [1, 2])
def test_routing_matches_per_token_loop(activation, k):
    moe = _moe(activation, k=k)
    x = torch.randn(3, 50, 256, device="cuda")
    y = moe(x)
    x2 = x.reshape(-1, 256)
    w, idx, _ = moe.route(x2)
    want = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        for j in range(k):
            want[t] += w[t, j] * _expert(moe, int(idx[t, j]), x2[t:t + 1])[0]
    torch.testing.assert_close(y.reshape(-1, 256), want, rtol=1e-4, atol=1e-5)
    assert sum(moe.last_counts) == x2.shape[0] * k


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_nvfp4_experts_track_bf16(activation):
    ref = _moe(activation, E=4, k=2, d=512, d_ff=1024)
    moe = copy.deepcopy(ref)
    assert convert_linears(moe) == [""] and moe.nvfp4
    assert type(moe.router) is torch.nn.Linear  # the router stays BF16
    x = torch.randn(4, 256, 512, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = moe(x)
        y_ref = ref(x2)
    assert _cos(y, y_ref) > 0.96  # two NVFP4 GEMMs in series (one NVFP4Linear: > 0.98, test_linear.py)
    torch.manual_seed(1)
    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert _cos(x.grad, x2.grad) > 0.95
    for (n, p), (_, q) in zip(moe.named_parameters(), ref.named_parameters()):
        assert p.grad is not None and _cos(p.grad, q.grad) > 0.9, n


def _gemm_grouped_reference(a, b, mode, meta, *, group_rows=0, out_dtype=torch.bfloat16, out=None,
                            accumulate=False):
    """ops.gemm_grouped computed per group in fp64 from the dequantized operands."""
    A, B = a.dequantize().double(), b.dequantize().double()
    if mode == "m":
        res = torch.zeros(A.shape[0], group_rows, dtype=torch.float64, device=A.device)
        for t, g in enumerate(meta.tolist()):
            rows = slice(128 * t, 128 * (t + 1))
            res[rows] = A[rows] @ B[g * group_rows:(g + 1) * group_rows].T
        res = res.to(out_dtype)
        if out is None:
            return res
        out.copy_(out + res if accumulate else res)
        return out
    for k0, n, r0 in meta.tolist():
        blk = A[:, k0:k0 + n] @ B[:, k0:k0 + n].T
        rows = slice(r0, r0 + A.shape[0])
        out[rows] = (out[rows].double() + blk if accumulate else blk).to(out.dtype)
    return out


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_grouped_path_matches_per_group_reference(activation, monkeypatch):
    def run(patch):
        moe = _moe(activation, E=5, k=2, d=256, d_ff=384, seed=3)
        convert_linears(moe)
        if patch:
            monkeypatch.setattr(moe_mod.ops, "gemm_grouped", _gemm_grouped_reference)
        torch.manual_seed(7)  # the same rounding seeds in both runs
        x = torch.randn(300, 256, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.float().square().sum().backward()
        monkeypatch.undo()
        return y.float(), x.grad, [p.grad for p in moe.experts_weights()], moe.last_counts
    y, dx, gw, counts = run(False)
    y_r, dx_r, gw_r, counts_r = run(True)
    assert counts == counts_r and min(counts) > 0
    # same quantized operands: only fp32 accumulation order (and bf16 outputs) differ; quantizers downstream
    # of a bf16 output can round an element differently, hence not bit-exact
    for a, b in [(y, y_r), (dx, dx_r), *zip(gw, gw_r)]:
        assert (a - b).norm() / b.norm() < 2e-2


def test_experts_without_tokens_and_fused_accumulation():
    moe = _moe(E=4, k=1)
    convert_linears(moe, fuse_wgrad_accumulation=True)
    with torch.no_grad():  # route everything to expert 2
        moe.router.weight.zero_()
        moe.router.weight[2] = 1.0
    x = torch.rand(64, 256, device="cuda") + 0.1
    for it in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.float().sum().backward()
        if it == 0:
            first = moe.w_gate.grad.clone()
    assert moe.last_counts == [0, 0, 64, 0]
    g = moe.w_gate.grad
    assert g.dtype == torch.float32 and torch.count_nonzero(g[2]) > 0
    assert torch.count_nonzero(g[[0, 1, 3]]) == 0
    # the second backward added (in place) a gradient of the same size as the first
    assert (g - 2 * first).norm() / first.norm() < 0.2
    assert moe.aux_loss.item() == pytest.approx(4 * 1.0 * moe.route(x)[2].mean(0)[2].item(), rel=1e-5)


def test_weight_cache_and_bf16_tail():
    moe = _moe(E=4, k=2)
    ref = copy.deepcopy(moe)
    convert_linears(moe, cache_weight_quantization=True)
    x = torch.randn(128, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y1 = moe(x)
        y2 = moe(x)
    assert torch.equal(y1, y2) and moe._caches[0].hits >= 1
    try:
        ops.set_high_precision(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            assert torch.equal(moe(x), ref(x))
    finally:
        ops.set_high_precision(False)


def test_balanced_router_aux_loss_is_one():
    moe = _moe(E=4, k=2)
    with torch.no_grad():
        moe.router.weight.zero_()  # uniform probabilities; top-k ties broken by index -> experts 0, 1
    moe(torch.randn(32, 256, device="cuda"))
    # f = (0.5, 0.5, 0, 0), P = 1/4 each: E * sum f P = 4 * (0.5 * 0.25 * 2) = 1
    assert moe.aux_loss.item() == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError):
        MoE(256, 512, 4, top_k=5)
    with pytest.raises(ValueError):
        MoE(256, 512, 4, activation="relu")
    with pytest.raises(ValueError):
        MoE(256, 200, 4).to_nvfp4()
