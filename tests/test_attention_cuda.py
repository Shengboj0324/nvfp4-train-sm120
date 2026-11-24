"""nvfp4.attention: 8-bit flash attention (INT8 scores, MXFP8 probabilities / gradients) for sm_120.

The operand quantizer is checked bit-exactly against nvfp4.reference (both layouts, mean subtraction, NaN);
the forward against the reference emulation of the same quantization (which differs only in where the
probabilities are rounded) and against exact attention; the gradients against exact attention, at the error
levels 8-bit operands imply; plus determinism, gradient layouts, GQA, ragged lengths, fallback and compile."""
import math

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
    row, rs, col, cs, _, raw = lib.attn_quant(x, mean, None, True, True, True)
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


def test_quantizer_delta_and_strides():
    # dO as a transposed view (the layout autograd hands over), delta = rowsum(dO * O)
    g = torch.randn(2, 300, 4, 64, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    o = torch.randn(2, 300, 4, 64, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    row, rs, col, cs, delta, _ = lib.attn_quant(g, None, o, True, True, False)
    row2, rs2, col2, cs2, _, _ = lib.attn_quant(g.contiguous(), None, None, True, True, False)
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


@pytest.mark.parametrize("B,H,Hk,N,D,causal", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gradients(B, H, Hk, N, D, causal, dtype):
    q, k, v = _qkv(B, H, Hk, N, D, dtype=dtype)
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    torch.manual_seed(1)
    do = torch.randn_like(q)
    attention(q, k, v, causal=causal).backward(do)
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
    q, k, v = (t.requires_grad_() for t in _qkv(2, 4, 2, 1000, 128))
    do = torch.randn_like(q)
    runs = []
    for _ in range(2):
        attention(q, k, v, causal=True).backward(do)
        runs.append([t.grad.clone() for t in (q, k, v)])
        for t in (q, k, v):
            t.grad = None
    assert all(torch.equal(a, b) for a, b in zip(*runs))


def test_gradient_layouts_follow_inputs():
    # q / k / v as views of a fused projection output [B, N, 3, H, D] (the usual case): no copies needed
    B, N, H, D = 2, 256, 4, 64
    x = torch.randn(B, N, 3, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    q, k, v = x.unbind(2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = attention(q, k, v, causal=True)
    assert o.transpose(1, 2).is_contiguous()  # [B, N, H, D] memory: the head merge is free
    # (q is not dense - it skips k and v - so its gradient comes as [B, N, H, D] memory, which the view
    # backward scatters into x's gradient)
    gq = torch.autograd.grad(o, q, torch.ones_like(o), retain_graph=True)[0]
    assert gq.transpose(1, 2).is_contiguous()
    o.sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    # contiguous [B, H, N, D] inputs get contiguous gradients
    q2, k2, v2 = (t.detach().contiguous().requires_grad_() for t in (q, k, v))
    attention(q2, k2, v2).sum().backward()
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


def test_torch_compile_fullgraph():
    q, k, v = (t.requires_grad_() for t in _qkv(1, 4, 2, 256, 64))

    def f(q, k, v):
        return attention(q, k, v, causal=True).float().square().mean()

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


def test_scale_and_no_grad():
    q, k, v = _qkv(1, 2, 2, 300, 128)
    with torch.no_grad():
        o = attention(q, k, v, scale=0.5 / math.sqrt(128))
    assert _rel(o, _exact(q, k, v, False, 0.5 / math.sqrt(128))) < 0.05
