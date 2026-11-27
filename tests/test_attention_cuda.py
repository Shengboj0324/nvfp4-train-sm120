"""nvfp4.attention: 8-bit flash attention (INT8 scores, MXFP8 probabilities / gradients) for sm_120.

The operand quantizer is checked bit-exactly against nvfp4.reference (both layouts, the dequantized values,
mean subtraction, NaN); the forward against the reference emulation of the same quantization (which differs
only in where the probabilities are rounded) and against exact attention, and its two-term-P variant against
exact attention of the quantized inputs; the gradients of both backwards against exact attention at the
error levels 8-bit operands imply, and those of backward="bf16" against the exact gradient of the quantized
forward; plus determinism, gradient layouts, GQA, ragged lengths, fallback and compile."""
import math
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4 import _library as lib, reference as ref
from nvfp4.attention import attention, supported

pytestmark = requires_sm120
DEV = "cuda"


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _qkv(B, H, Hk, N, D, dtype=torch.bfloat16, k_offset=0.5, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, D, device=DEV).to(dtype)
    k = (torch.randn(B, Hk, N, D, device=DEV) + k_offset).to(dtype)  # keys with a channel offset (smoothing)
    v = torch.randn(B, Hk, N, D, device=DEV).to(dtype)
    return q, k, v


def _exact(q, k, v, causal, scale):
    rep = q.shape[1] // k.shape[1]
    return F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(rep, 1),
                                          v.float().repeat_interleave(rep, 1), is_causal=causal, scale=scale)


@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("N", [1, 64, 200, 1000])
@pytest.mark.parametrize("with_mean", [False, True])
def test_quantizer_is_exact(D, N, with_mean):
    x = torch.randn(2, 3, N, D, device=DEV, dtype=torch.bfloat16) * 3
    mean = x.mean(dim=2, dtype=torch.float32) if with_mean else None
    row, rs, col, cs, _, raw, _, _ = lib.attn_quant(x, mean, None, True, True, True)
    xs = x.float() - mean.unsqueeze(2) if with_mean else x.float()
    codes, sc = ref.int8_token_quantize(xs)
    Np = row.shape[2]
    assert Np % 128 == 0 and Np >= N
    assert torch.equal(row[:, :, :N].view(torch.int8), codes) and torch.equal(rs[:, :, :N], sc)
    assert not row[:, :, N:].any() and not rs[:, :, N:].any()  # zero padding
    assert torch.equal(raw[:, :, :N], xs.bfloat16()) and not raw[:, :, N:].any()
    xp = ref._pad_seq(xs)
    sig = torch.tensor(ref.ATTN_SIGMA, device=DEV)
    perm = (torch.arange(Np // 32, device=DEV)[:, None] * 32 + sig[None]).reshape(-1)
    ccodes, ce = ref.mxfp8_quantize(xp[:, :, perm], dim=-2)
    ce = ce.reshape(2, 3, D // 8, 8, Np // 64, 2).permute(0, 1, 4, 3, 2, 5).reshape(cs.shape)
    assert torch.equal(col.view(torch.uint8), ccodes.view(torch.uint8)) and torch.equal(cs.int(), ce)
    # dequantized values (backward="bf16"): of the row layout when it is requested, else of the col layout
    deq_row = lib.attn_quant(x, mean, None, True, False, False, True)[6]
    assert deq_row.dtype == x.dtype and deq_row.shape == row.shape
    assert torch.equal(deq_row[:, :, :N], ref.int8_token_fake(xs).to(x.dtype))
    deq_col = lib.attn_quant(x, mean, None, False, True, False, True)[6]
    assert torch.equal(deq_col, ref.mxfp8_fake(xp, dim=-2).to(x.dtype))


def test_quantizer_delta_and_strides():
    # dO as a transposed view (the layout autograd hands over), delta = rowsum(dO * O)
    g = torch.randn(2, 300, 4, 64, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    o = torch.randn(2, 300, 4, 64, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    row, rs, col, cs, delta, _, _, _ = lib.attn_quant(g, None, o, True, True, False)
    row2, rs2, col2, cs2, _, _, _, _ = lib.attn_quant(g.contiguous(), None, None, True, True, False)
    assert torch.equal(row, row2) and torch.equal(rs, rs2) and torch.equal(col, col2) and torch.equal(cs, cs2)
    want = (g.float() * o.float()).sum(-1)
    torch.testing.assert_close(delta[:, :, :300], want, rtol=1e-5, atol=1e-4)
    assert not delta[:, :, 300:].any()


def test_nan_propagates():
    q, k, v = _qkv(1, 2, 2, 256, 128)
    q[0, 1, 17, 5] = float("nan")
    o = attention(q, k, v, causal=True)
    assert torch.isnan(o[0, 1, 17]).all()
    assert torch.isfinite(o[0, 0]).all() and torch.isfinite(o[0, 1, :17]).all()


CASES = [  # B, H, Hk, N, D, causal
    (2, 4, 4, 512, 128, True),
    (2, 4, 4, 512, 128, False),
    (1, 8, 2, 1000, 64, True),
    (2, 4, 4, 300, 64, False),
    (1, 4, 1, 2048, 128, True),
    (1, 2, 2, 1, 64, True),
    (3, 2, 1, 77, 128, False),
]


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
def test_forward_matches_emulation_and_exact(B, H, Hk, N, D, causal):
    q, k, v = _qkv(B, H, Hk, N, D)
    scale = 2 * D ** -0.5
    o = attention(q, k, v, causal=causal, scale=scale)
    assert o.shape == q.shape and o.dtype == q.dtype
    emu = ref.attention_8bit(q, k, v, causal=causal, scale=scale)
    exact = _exact(q, k, v, causal, scale)
    assert _rel(o, emu) < 0.03  # the same quantization; P rounded relative to a different maximum
    assert _rel(o, exact) < 0.05  # 8-bit error level (BF16 flash attention: ~0.002)


def _quantized_inputs(q, k, v):
    """The operands as the 8-bit kernels see them (fp32): int8 q / centred k per token, MXFP8 v."""
    N = q.shape[2]
    return (ref.int8_token_fake(q.float()), ref.int8_token_fake(k.float() - k.float().mean(2, keepdim=True)),
            ref.mxfp8_fake(ref._pad_seq(v.float()), dim=-2)[..., :N, :])


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
def test_two_term_forward_is_exact_attention_of_quantized_inputs(B, H, Hk, N, D, causal, monkeypatch):
    # backward="bf16" runs the forward with P V = e4m3(P) V + e4m3(residual) V: the output is then exact
    # attention of the quantized operands up to ~2^-8 in P (BF16 flash attention's own level), which keeps
    # Delta = rowsum(dO O) consistent with the exact backward
    A = sys.modules["nvfp4.attention"]  # (nvfp4.attention the attribute is the function)

    q, k, v = _qkv(B, H, Hk, N, D)
    want = _exact(*(t.to(q.dtype) for t in _quantized_inputs(q, k, v)), causal, None)
    with torch.no_grad():
        o2 = attention(q, k, v, causal=causal, backward="bf16")
        monkeypatch.setattr(A, "_FWD_P2_8BIT", False)  # the one-term forward
        o1 = attention(q, k, v, causal=causal, backward="8bit")
    assert _rel(o2, want) < 0.006
    if N > 1:
        assert _rel(o2, want) < 0.5 * _rel(o1, want)


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_bf16_backward_is_exact_gradient_of_quantized_forward(B, H, Hk, N, D, causal, dtype):
    q, k, v = _qkv(B, H, Hk, N, D, dtype=dtype)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    torch.manual_seed(1)
    do = torch.randn_like(q)
    attention(q, k, v, causal=causal, backward="bf16").backward(do)
    got = [t.grad for t in (q, k, v)]
    # straight-through: the gradient of exact attention at the quantized operands (k's mean shift changes no
    # gradient)
    qq, kq, vq = (t.detach().to(dtype).float().requires_grad_() for t in _quantized_inputs(q, k, v))
    _exact(qq, kq, vq, causal, D ** -0.5).backward(do.float())
    for name, a, b in zip("qkv", got, (qq.grad, kq.grad, vq.grad)):
        assert a.dtype == dtype and a.shape == b.shape and a.isfinite().all()
        if N == 1 and name != "v":
            assert a.float().norm() < 0.01 * do.float().norm(), name  # zero analytically (a single key)
            continue
        assert _rel(a, b) < 0.012, name  # BF16 flash attention level (exact inputs: ~0.003)


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backward", ["bf16", "8bit"])
def test_gradients(B, H, Hk, N, D, causal, dtype, backward):
    q, k, v = _qkv(B, H, Hk, N, D, dtype=dtype)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    torch.manual_seed(1)
    do = torch.randn_like(q)
    attention(q, k, v, causal=causal, backward=backward).backward(do)
    got = [t.grad for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    _exact(q, k, v, causal, D ** -0.5).backward(do.float())
    for name, a, b in zip("qkv", got, (t.grad for t in (q, k, v))):
        assert a.dtype == dtype and a.shape == b.shape and torch.isfinite(a).all()
        if N == 1 and name != "v":
            # dq and dk vanish analytically for a single key (dS = P (dP - Delta) = 0); with 8-bit dO / V in dP
            # and Delta = rowsum(dO * O) in high precision only the quantization residual remains
            assert a.float().norm() < 0.05 * do.float().norm(), name
            continue
        assert _rel(a, b) < 0.07, name


def test_backward_is_deterministic():
    # the 8-bit backward has no atomics; backward="bf16" is PyTorch's flash backward (deterministic only under
    # torch.use_deterministic_algorithms, as for SDPA)
    q, k, v = (t.requires_grad_() for t in _qkv(2, 4, 2, 1000, 128))
    do = torch.randn_like(q)
    runs = []
    for _ in range(2):
        attention(q, k, v, causal=True, backward="8bit").backward(do)
        runs.append([t.grad.clone() for t in (q, k, v)])
        for t in (q, k, v):
            t.grad = None
    assert all(torch.equal(a, b) for a, b in zip(*runs))


@pytest.mark.parametrize("backward", ["bf16", "8bit"])
def test_gradient_layouts_follow_inputs(backward):
    # q / k / v as views of a fused projection output [B, N, 3, H, D] (the usual case): no copies needed
    B, N, H, D = 2, 256, 4, 64
    x = torch.randn(B, N, 3, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    q, k, v = x.unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = attention(q, k, v, causal=True, backward=backward)
    assert o.transpose(1, 2).is_contiguous()  # [B, N, H, D] memory: the head merge is free
    # (q is not dense - it skips k and v - so its gradient comes as [B, N, H, D] memory, which the view
    # backward scatters into x's gradient)
    gq = torch.autograd.grad(o, q, torch.ones_like(o), retain_graph=True)[0]
    if backward == "8bit":  # (flash attention's backward picks its own layout)
        assert gq.transpose(1, 2).is_contiguous()
    o.sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    q2, k2, v2 = (t.detach().contiguous().requires_grad_() for t in (q, k, v))
    attention(q2, k2, v2, backward=backward).sum().backward()
    assert q2.grad.shape == q2.shape and k2.grad.shape == k2.shape and v2.grad.shape == v2.shape
    if backward == "8bit":  # contiguous [B, H, N, D] inputs get contiguous gradients
        assert q2.grad.is_contiguous() and k2.grad.is_contiguous() and v2.grad.is_contiguous()


def test_smooth_k_changes_nothing_exactly_in_exact_arithmetic():
    # the mean subtraction is softmax-invariant: with and without it both stay at the 8-bit error level, and
    # centring helps when keys carry a large channel offset
    q, k, v = _qkv(1, 4, 4, 512, 128, k_offset=4.0)
    exact = _exact(q, k, v, True, None)
    e_smooth = _rel(attention(q, k, v, causal=True, smooth_k=True), exact)
    e_raw = _rel(attention(q, k, v, causal=True, smooth_k=False), exact)
    assert e_smooth < 0.05 and e_smooth < e_raw


def test_fallback_to_sdpa():
    q = torch.randn(1, 2, 64, 96, device=DEV, dtype=torch.bfloat16)  # head dim 96: not covered
    assert not supported(q, q, q)
    torch.testing.assert_close(attention(q, q, q, causal=True),
                               F.scaled_dot_product_attention(q, q, q, is_causal=True), rtol=0, atol=0)
    q32 = q[..., :64].float()  # fp32: not covered
    assert not supported(q32, q32, q32)


@pytest.mark.parametrize("backward", ["bf16", "8bit"])
def test_torch_compile_fullgraph(backward):
    q, k, v = (t.requires_grad_() for t in _qkv(1, 4, 2, 256, 64))

    def f(q, k, v):
        return attention(q, k, v, causal=True, backward=backward).float().square().mean()

    eager = f(q, k, v)
    eager.backward()
    ge = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    torch._dynamo.reset()
    cf = torch.compile(f, fullgraph=True, backend=COMPILE_BACKEND)
    out = cf(q, k, v)
    out.backward()
    # the kernels are the same; the surrounding reductions (the loss here, K's mean) may be reordered by inductor
    torch.testing.assert_close(out, eager, rtol=1e-4, atol=0)
    assert all(_rel(t.grad, g) < 1e-2 for t, g in zip((q, k, v), ge))


def test_backward_argument():
    q, k, v = _qkv(1, 2, 2, 64, 64)
    with pytest.raises(ValueError):
        attention(q, k, v, backward="fp8")


def test_scale_and_no_grad():
    q, k, v = _qkv(1, 2, 2, 300, 128)
    with torch.no_grad():
        o = attention(q, k, v, scale=0.5 / math.sqrt(128))
    assert _rel(o, _exact(q, k, v, False, 0.5 / math.sqrt(128))) < 0.05


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
def test_math_backward_matches_flash_backward(B, H, Hk, N, D, causal, monkeypatch):
    """PyTorch builds without flash attention (its Windows wheels) get the backward's math in fp32, block by
    block of query rows: the same gradients as the flash backward, to its bf16 level."""
    import sys
    attn_mod = sys.modules["nvfp4.attention"]  # (nvfp4.attention, the attribute, is the function)
    q, k, v = _qkv(B, H, Hk, N, D)
    torch.manual_seed(1)
    do = torch.randn_like(q)
    grads = []
    for flash in (None, False):
        monkeypatch.setattr(attn_mod, "_FLASH_BWD", flash)
        if flash is False:  # smaller row blocks, so that several blocks and the causal edges are exercised
            monkeypatch.setattr(attn_mod, "_math_backward",
                                lambda *a, _f=attn_mod._math_backward: _f(*a, rows=64))
        ts = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        attention(*ts, causal=causal, backward="bf16").backward(do)
        grads.append([t.grad for t in ts])
    if torch.backends.cuda.is_flash_attention_available():
        for name, a, b in zip("qkv", *grads):
            assert a.dtype == b.dtype and a.shape == b.shape and b.isfinite().all()
            if a.float().norm() < 0.01 * do.float().norm():  # N == 1: zero analytically for q and k
                assert b.float().norm() < 0.01 * do.float().norm(), name
            else:
                assert _rel(b, a) < 0.01, name
