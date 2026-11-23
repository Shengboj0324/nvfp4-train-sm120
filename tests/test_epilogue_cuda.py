"""v0.8 fusions at the op level: the GEMM's bias / residual epilogue, multi-part (in-place concatenated)
quantization with column sums, SwiGLU biases, and the LayerNorm / RMSNorm backward kernel. Each against
its nvfp4.reference spec (bit-exact where the spec is exact)."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops, reference as ref

pytestmark = requires_sm120
DEV = "cuda"


def _q(shape, seed, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    return ops.quantize_dual((torch.randn(*shape, device=DEV) * 3).to(dtype), col=False)[0]


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("res_dtype", [None, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("impl,splits", [("tma", 1), ("cp.async", 1), ("tma", 3)])
def test_gemm_epilogue_is_exact(out_dtype, res_dtype, with_bias, impl, splits):
    if res_dtype is None and not with_bias:
        pytest.skip("no epilogue")
    M, N, K = 300, 384, 1024  # ragged M
    a, b = _q((M, K), 0), _q((N, K), 1)
    torch.manual_seed(2)
    bias = torch.randn(N, device=DEV) if with_bias else None
    res = torch.randn(M, N, device=DEV).to(res_dtype) if res_dtype else None
    y = ops.gemm(a, b, out_dtype, bias=bias, residual=res, impl=impl, splits=splits)
    plain = ops.gemm(a, b, torch.bfloat16, impl=impl, splits=splits)
    want = ref.linear_epilogue(plain, out_dtype, bias, res)
    assert y.dtype == want.dtype and torch.equal(y, want)


def test_gemm_epilogue_rejects_accumulate():
    a, b = _q((128, 256), 0), _q((128, 256), 1)
    out = torch.zeros(128, 128, device=DEV)
    with pytest.raises(ValueError):
        ops.gemm(a, b, torch.float32, out=out, accumulate=True, bias=torch.zeros(128, device=DEV))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nparts", [1, 2, 3])
@pytest.mark.parametrize("R", [100, 512])
def test_quantize_parts_equals_concatenation(dtype, nparts, R):
    torch.manual_seed(nparts)
    parts = [(torch.randn(R, 256, device=DEV) * (i + 1)).to(dtype) for i in range(nparts)]
    cat = torch.cat(parts, 1)
    kw = dict(row_sr_seed=11, col_sr_seed=12, rht_seed=13, col_pad=128)
    r1, c1, db = ops.quantize_dual(parts, bias_grad=True, **kw)
    r0, c0 = ops.quantize_dual(cat, **kw)
    for p, q in ((r0, r1), (c0, c1)):
        assert all(torch.equal(u, v) for u, v in zip(p.as_tuple(), q.as_tuple()))
    rr, rc = ref.quantize_dual(cat, row_sr_seed=11, col_sr_seed=12, rht_seed=13, col_pad=128)
    assert torch.equal(r1.data, rr[0]) and torch.equal(c1.data, rc[0])
    assert torch.allclose(db, cat.float().sum(0), rtol=1e-5, atol=1e-4)
    # device seeds and a single tensor with bias_grad
    s = torch.tensor([11, 12, 13], device=DEV)
    r2, c2, db2 = ops.quantize_dual(parts[0], bias_grad=True, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2],
                                    col_pad=128)
    r3, c3 = ops.quantize_dual(parts[0], **kw)
    assert torch.equal(r2.data, r3.data) and torch.equal(c2.scales, c3.scales)
    assert torch.allclose(db2, parts[0].float().sum(0), rtol=1e-5, atol=1e-4)


def test_quantize_parts_rejects_mismatch():
    a = torch.randn(64, 128, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        ops.quantize_dual([a, a[:, :64].contiguous()])
    with pytest.raises(RuntimeError):
        ops.quantize_dual([a, a.float()])


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.parametrize("biases", ["none", "gate", "both"])
def test_swiglu_biases_are_exact(gate, biases):
    torch.manual_seed(4)
    R, C = 192, 256
    g, u, dy = (torch.randn(R, C, device=DEV).bfloat16() * 2 for _ in range(3))
    bg = torch.randn(C, device=DEV) if biases != "none" else None
    bu = torch.randn(C, device=DEV) if biases == "both" else None
    kw = dict(row_sr_seed=3, col_sr_seed=4, rht_seed=5, col_pad=128)
    q = ops.swiglu_quantize_dual(g, u, bias_g=bg, bias_u=bu, **kw)
    rq = ref.swiglu_quantize_dual(g, u, bias_g=bg, bias_u=bu, **kw)
    for a, b in zip(q, rq):
        assert torch.equal(a.data, b[0]) and torch.equal(a.scales, b[1])
    row, col, db = ops.swiglu_backward_quantize_dual(dy, g, u, gate=gate, bias_g=bg, bias_u=bu, bias_grad=True, **kw)
    rr, rc = ref.swiglu_backward_quantize_dual(dy, g, u, gate=gate, bias_g=bg, bias_u=bu, **kw)
    assert torch.equal(row.data, rr[0]) and torch.equal(col.data, rc[0]) and torch.equal(col.scales, rc[1])
    want = ref.swiglu_backward(dy, g, u, gate=gate, bias_g=bg, bias_u=bu).sum(0)
    assert torch.allclose(db, want, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("kind", ["ln", "rms"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_res", [False, True])
@pytest.mark.parametrize("R,C", [(333, 512), (64, 4096), (64, 6144), (40, 16384)])  # 1, 2, 3 (4) chunks; warp path
def test_norm_backward_matches_reference(kind, dtype, with_res, R, C):
    torch.manual_seed(6)
    x = (torch.randn(R, C, device=DEV) * 2 + 0.5).to(dtype)
    w = torch.rand(C, device=DEV) + 0.5
    dy = torch.randn(R, C, device=DEV)
    xf = x.float()
    mean = xf.mean(-1) if kind == "ln" else None
    rstd = ((xf - (mean[:, None] if mean is not None else 0)).square().mean(-1) + 1e-5).rsqrt()
    dres = torch.randn(R, C, device=DEV).to(dtype) if with_res else None
    dx, dw, db = ops.norm_backward(dy, x, mean, rstd, w, dres=dres, need_db=kind == "ln")
    rdx, rdw, rdb = ref.norm_backward(dy, x, mean, rstd, w, dres)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    assert dx.dtype == dtype
    assert (dx.float() - rdx.float()).abs().max() <= tol * rdx.float().abs().max()
    assert torch.allclose(dw, rdw, rtol=1e-4, atol=1e-3)
    if kind == "ln":
        assert torch.allclose(db, rdb, rtol=1e-4, atol=1e-3)
    # deterministic
    dx2, dw2, _ = ops.norm_backward(dy, x, mean, rstd, w, dres=dres, need_db=kind == "ln")
    assert torch.equal(dx, dx2) and torch.equal(dw, dw2)
    # and agrees with autograd through the PyTorch norm
    xr = xf.clone().requires_grad_()
    if kind == "ln":
        out = torch.nn.functional.layer_norm(xr, (C,), w, torch.zeros_like(w), 1e-5)
    else:
        out = torch.nn.functional.rms_norm(xr, (C,), w, 1e-5)
    out.backward(dy)
    want = xr.grad + (dres.float() if with_res else 0)
    assert (dx.float() - want).abs().max() <= 3 * tol * want.abs().max()


def test_norm_backward_partial_outputs():
    x = torch.randn(64, 128, device=DEV)
    rstd = torch.ones(64, device=DEV)
    dx, dw, db = ops.norm_backward(torch.randn_like(x), x, None, rstd, torch.ones(128, device=DEV), need_dx=False)
    assert dx is None and dw.shape == (128,) and db is None
    with pytest.raises(RuntimeError):
        ops.norm_backward(torch.randn_like(x), x, None, rstd, torch.ones(128, device=DEV), need_db=True)
