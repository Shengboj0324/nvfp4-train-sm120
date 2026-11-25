"""nvfp4.moe.MoE: routing against a per-token reference loop (BF16 experts); NVFP4 experts (grouped GEMMs)
against BF16 ones (NVFP4 error level, forward and gradients) and against the same computation with a per-group
fp64 reference in place of the grouped GEMM (checks the padding, tile map, segments and the stacked transposed
weights, forward and backward); experts without tokens; fused gradient accumulation; the weight cache; the
BF16 tail; the load-balancing loss; expert biases (both paths; bias gradients against the routed output
gradient); the grouped high-precision path (Triton) against the per-expert loop."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import HAS_TRITON, requires_sm120, requires_triton
from nvfp4 import convert_linears, ops
from nvfp4 import moe as moe_mod
from nvfp4.moe import MoE

pytestmark = requires_sm120


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _moe(activation="swiglu", E=4, k=2, d=256, d_ff=512, seed=0, bias=False):
    torch.manual_seed(seed)
    return MoE(d, d_ff, E, top_k=k, activation=activation, bias=bias).cuda()


def _expert(moe, e, x):
    ws = moe.experts_weights()
    bs = [None if b is None else b[e] for b in moe.experts_biases()]
    if moe.activation == "swiglu":
        return F.linear(F.silu(F.linear(x, ws[0][e], bs[0])) * F.linear(x, ws[1][e], bs[1]), ws[2][e], bs[2])
    approx = "tanh" if moe.activation == "gelu_tanh" else "none"
    return F.linear(F.gelu(F.linear(x, ws[0][e], bs[0]), approximate=approx), ws[1][e], bs[1])


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh", "gelu"])
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("impl,bias", [("loop", False), ("grouped", False), ("grouped", True), ("loop", True)])
def test_routing_matches_per_token_loop(activation, k, impl, bias):
    if impl == "grouped" and not HAS_TRITON:
        pytest.skip("needs Triton")
    """fp32 experts (the grouped path: Triton with IEEE fp32 products)."""
    moe = _moe(activation, k=k, bias=bias)
    moe.bf16_impl = impl
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
@pytest.mark.parametrize("bias", [False, True])
def test_nvfp4_experts_track_bf16(activation, bias):
    ref = _moe(activation, E=4, k=2, d=512, d_ff=1024, bias=bias)
    ref.bf16_impl = "loop"
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
                            accumulate=False, bias=None, tile_rows=None):
    """ops.gemm_grouped computed per group in fp64 from the dequantized operands."""
    A, B = a.dequantize().double(), b.dequantize().double()
    if mode == "m":
        res = torch.zeros(A.shape[0], group_rows, dtype=torch.float64, device=A.device)
        for t, g in enumerate(meta.tolist()):
            if g >= 0:  # -1: an unused tile, zeros
                rows = slice(128 * t, 128 * (t + 1))
                res[rows] = A[rows] @ B[g * group_rows:(g + 1) * group_rows].T
        res = res.to(out_dtype)
        if bias is not None:  # the real rows of each used tile
            for t, (g, n) in enumerate(zip(meta.tolist(), tile_rows.tolist())):
                if g >= 0:
                    res[128 * t:128 * t + n] += bias[g].to(out_dtype)
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
@pytest.mark.parametrize("bias", [False, True])
def test_grouped_path_matches_per_group_reference(activation, bias, monkeypatch):
    def run(patch):
        moe = _moe(activation, E=5, k=2, d=256, d_ff=384, seed=3, bias=bias)
        convert_linears(moe)
        if patch:
            monkeypatch.setattr(moe_mod.ops, "gemm_grouped", _gemm_grouped_reference)
        torch.manual_seed(7)  # the same rounding seeds in both runs
        x = torch.randn(300, 256, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.float().square().sum().backward()
        monkeypatch.undo()
        params = [p for p in moe.parameters() if p is not moe.router.weight]
        return y.float(), x.grad, [p.grad for p in params], moe.last_counts
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
    m = _moe()
    m.bf16_impl = "fast"
    with pytest.raises(ValueError):
        m(torch.randn(8, 256, device="cuda"))


@requires_triton
@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
@pytest.mark.parametrize("bias", [False, True])
def test_bf16_grouped_matches_loop(activation, bias):
    """The grouped high-precision experts (Triton, on the padded buffer) against the per-expert loop under
    bf16 autocast: forward and every gradient at bf16 rounding level (different fp32 summation orders), and
    deterministic (two runs bit-identical)."""
    def run(impl):
        moe = _moe(activation, E=6, k=2, d=256, d_ff=384, seed=4, bias=bias)
        moe.bf16_impl = impl
        torch.manual_seed(9)
        x = torch.randn(500, 256, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        (y.float().square().sum() + moe.aux_loss).backward()
        return [y.float(), x.grad] + [p.grad for p in moe.parameters()]
    loop, grouped, again = run("loop"), run("grouped"), run("grouped")
    for i, (a, b, c) in enumerate(zip(loop, grouped, again)):
        assert torch.equal(b, c), i
        assert ((a - b).norm() / a.norm()).item() < 1e-2, i


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_nvfp4_bias_gradients_are_routed_gradient_sums(activation):
    """The output bias gradient of every expert is the sum of the gradient rows routed to it: the combine
    weight times the upstream gradient, over the expert's (token, choice) slots (fp64 here; the layer's dy is
    bf16). An expert without tokens gets zero."""
    moe = _moe(activation, E=5, k=2, bias=True)
    convert_linears(moe)
    x = torch.rand(200, 256, device="cuda") + 0.1
    with torch.no_grad():
        moe.router.weight[4] = -10.0  # positive inputs: expert 4 gets a large negative logit, no tokens
        w, idx, _ = moe.route(x)
    torch.manual_seed(3)
    g = torch.randn(200, 256, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = moe(x)
    y.backward(g)
    want = torch.zeros(5, 256, dtype=torch.float64, device="cuda")
    for j in range(2):
        want.index_add_(0, idx[:, j], w[:, j, None].double() * g.double())
    got = moe.experts_biases()[-1].grad
    assert moe.last_counts[4] == 0 and torch.count_nonzero(got[4]) == 0
    assert ((got.double() - want).norm() / want.norm()).item() < 1e-2
    for b in moe.experts_biases():
        assert b.grad is not None and torch.isfinite(b.grad).all() and torch.count_nonzero(b.grad[4]) == 0


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_nvfp4_biases_only_trainable(activation):
    """Frozen expert weights and no input gradient: the bias gradients still flow (the quantizers run for their
    column sums alone) and equal those of the fully trainable layer bit for bit (same seeds, same operands)."""
    def run(freeze):
        moe = _moe(activation, E=4, k=2, bias=True, seed=2)
        convert_linears(moe)
        for w in moe.experts_weights():
            w.requires_grad_(not freeze)
        torch.manual_seed(11)
        x = torch.randn(256, 256, device="cuda", requires_grad=not freeze)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe(x)
        y.float().square().sum().backward()
        assert all((w.grad is None) == freeze for w in moe.experts_weights())
        return [b.grad for b in moe.experts_biases()]
    for a, b in zip(run(True), run(False)):
        assert torch.equal(a, b)


def test_bias_option_parameters_and_state_dict():
    plain, biased = _moe(), _moe(bias=True)
    assert all(b is None for b in plain.experts_biases())
    assert [n for n, _ in biased.named_parameters()] == ["w_gate", "b_gate", "w_up", "b_up", "w_down", "b_down",
                                                         "router.weight"]
    assert set(plain.state_dict()) == {"router.weight", "w_gate", "w_up", "w_down"}  # unchanged without bias
    assert biased.b_down.shape == (4, 256) and "bias=True" in repr(biased)
