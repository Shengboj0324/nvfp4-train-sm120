"""Two-term forward activations (ops.set_fprop_terms(2), quantize_dual(row_terms=2)): the row output is
[Q(x) | Q(x - deq Q(x))] with one tensor scale, bit-exact against reference.quantize_dual on every producer path,
and the forward GEMM against [W | W] equals x W^T with the activations' rounding error about 8x smaller."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref
from nvfp4.fused import layernorm_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120

SHAPES = [(64, 64), (100, 128), (1000, 48), (130, 256), (512, 768)]
CONFIGS = {
    "row_only": dict(col=False),
    "with_col": dict(rht_seed=7, col_pad=128),
}


@pytest.fixture(autouse=True)
def _restore():
    yield
    ops.set_fprop_terms(1)
    ops.set_four_six(False)


def _check(got, want):
    for name, g, w in zip(("row", "col"), got, want):
        if w is None:
            assert g is None
            continue
        d, s, gd = w
        assert torch.equal(g.gdec.reshape(()), gd.reshape(())), f"{name}: global scale differs"
        assert torch.equal(g.scales, s), f"{name}: scales differ"
        assert torch.equal(g.data, d), f"{name}: data differs"


def _kw(cfg):
    return dict(CONFIGS[cfg], row_terms=2)


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize_dual_two_terms(cfg, shape, dtype):
    torch.manual_seed(shape[0] + 3 * shape[1])
    x = (torch.randn(shape, device="cuda") * 2 + 0.3).to(dtype)
    got = ops.quantize_dual(x, **_kw(cfg))
    assert got[0].data.shape == (shape[0], shape[1]) and got[0].scales.shape == (shape[0], shape[1] // 8)
    _check(got, ref.quantize_dual(x, **_kw(cfg)))
    # the first term and the col output are the one-term quantizer's
    one = ops.quantize_dual(x, **CONFIGS[cfg])
    assert torch.equal(got[0].data[:, :shape[1] // 2], one[0].data)
    assert torch.equal(got[0].scales[:, :shape[1] // 16], one[0].scales)
    if one[1] is not None:
        assert torch.equal(got[1].data, one[1].data)


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_producers_two_terms(cfg, shape):
    torch.manual_seed(shape[0] * 5 + shape[1])
    C = shape[1]
    x = (torch.randn(shape, device="cuda") * 2 + 0.5).bfloat16()
    w, b = torch.randn(C, device="cuda") * 0.5 + 1, torch.randn(C, device="cuda") * 0.1
    mean, rstd, *got = ops.layernorm_quantize_dual(x, w, b, 1e-5, **_kw(cfg))
    _check(got, ref.layernorm_quantize_dual(x, w, b, mean, rstd, **_kw(cfg)))
    rstd, *got = ops.rmsnorm_quantize_dual(x, w, 1e-6, **_kw(cfg))
    _check(got, ref.rmsnorm_quantize_dual(x, w, rstd, **_kw(cfg)))
    h = (torch.randn(shape, device="cuda") * 3).bfloat16()
    _check(ops.gelu_quantize_dual(h, bias=b, **_kw(cfg)), ref.gelu_quantize_dual(h, bias=b, **_kw(cfg)))
    _check(ops.gelu_quantize_dual(h, approximate="none", **_kw(cfg)),
           ref.gelu_quantize_dual(h, approximate="none", **_kw(cfg)))
    u = torch.randn(shape, device="cuda").bfloat16()
    _check(ops.swiglu_quantize_dual(h, u, **_kw(cfg)), ref.swiglu_quantize_dual(h, u, **_kw(cfg)))


def test_four_six_two_terms():
    torch.manual_seed(4)
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    ops.set_four_six(True)
    _check(ops.quantize_dual(x, col=False, row_terms=2), ref.quantize_dual(x, col=False, row_terms=2, four_six=True))


def test_error_and_gemm():
    torch.manual_seed(5)
    x = torch.randn(1024, 768, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16) * 0.05
    x1, _ = ops.quantize_dual(x, col=False)
    x2, _ = ops.quantize_dual(x, col=False, row_terms=2)
    d2 = ops.dequantize(x2.data, x2.scales, x2.gdec, torch.float32)
    e1 = (x1.dequantize() - x.float()).norm() / x.float().norm()
    e2 = (d2[:, :768] + d2[:, 768:] - x.float()).norm() / x.float().norm()
    assert e2 < e1 / 6, (e1.item(), e2.item())  # one term ~9.5%, two ~1%
    wq, _ = ops.quantize_dual(W, col=False, block_2d=True)
    w2 = ops.two_term_weight(wq)
    y = ops.gemm(x2, w2, torch.float32)
    want = ref.gemm(x2.as_tuple(), w2.as_tuple())
    assert torch.allclose(y, want, rtol=1e-5, atol=1e-4)
    exact = x.float() @ wq.dequantize().t()
    r1 = (ops.gemm(x1, wq, torch.float32) - exact).norm() / exact.norm()
    r2 = (y - exact).norm() / exact.norm()
    assert r2 < r1 / 5, (r1.item(), r2.item())


def test_rejects_unsupported():
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="row_terms"):
        ops.quantize_dual(x, row_sr_seed=3, row_terms=2)
    with pytest.raises(RuntimeError, match="row_terms"):
        ops.quantize_dual(x, block_2d=True, row_terms=2)
    with pytest.raises(ValueError):
        ops.quantize_dual([x, x], row_terms=2)
    with pytest.raises(ValueError):
        ops.set_fprop_terms(3)


@pytest.mark.parametrize("backend", ["cuda", "emulate"])
def test_linear_two_terms(backend):
    torch.manual_seed(6)
    lin = NVFP4Linear(256, 384, backend=backend, cache_weight_quantization=True).cuda()
    x = torch.randn(4, 64, 256, device="cuda", requires_grad=True)
    # against the quantized weight: what the activations' terms change
    wd = ops.quantize_dual(lin.weight.detach(), col=False, block_2d=True)[0].dequantize()
    exact = x.detach().reshape(-1, 256) @ wd.t() + lin.bias.detach()
    y1 = lin(x)
    ops.set_fprop_terms(2)
    torch.manual_seed(7)
    y2 = lin(x)
    y2.float().square().sum().backward()
    assert x.grad is not None and lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
    e1 = (y1.reshape(-1, 384).float() - exact).norm() / exact.norm()
    e2 = (y2.reshape(-1, 384).float() - exact).norm() / exact.norm()
    assert e2 < e1 / 4, (e1.item(), e2.item())
    if backend == "cuda":
        lin(x)
        assert lin.weight_quant_cache.wq2 is not None and lin.weight_quant_cache.hits >= 1
        assert lin.weight_quant_cache.wq2.data.shape == (384, 256)


def test_linear_cuda_matches_emulation():
    torch.manual_seed(8)
    lin = NVFP4Linear(256, 256, bias=False).cuda()
    emu = NVFP4Linear(256, 256, bias=False, backend="emulate").cuda()
    emu.weight.data.copy_(lin.weight.data)
    x = torch.randn(128, 256, device="cuda")
    ops.set_fprop_terms(2)
    with torch.no_grad():
        a, b = lin(x).float(), emu(x).float()
    assert (a - b).abs().max().item() <= 1e-2 * a.abs().max().item()


def test_fused_mlp_two_terms_is_exact_composition():
    # the fused LayerNorm -> fc1 -> GELU -> fc2 sub-block quantizes both producers in two terms and runs both GEMMs
    # against [W | W]
    torch.manual_seed(9)
    D, H = 256, 1024
    ln = torch.nn.LayerNorm(D).cuda()
    fc1, fc2 = NVFP4Linear(D, H, bias=False).cuda(), NVFP4Linear(H, D, bias=False).cuda()
    x = torch.randn(384, D, device="cuda") * 2 + 0.3
    ops.set_fprop_terms(2)
    with torch.no_grad():
        out = layernorm_mlp(x, ln, fc1, fc2)
    _, _, xq, _ = ops.layernorm_quantize_dual(x, ln.weight, ln.bias, ln.eps, col=False, row_terms=2)
    w1q, _ = ops.quantize_dual(fc1.weight.detach(), col=False, block_2d=True)
    h = ops.gemm(xq, ops.two_term_weight(w1q), torch.bfloat16)
    gq, _ = ops.gelu_quantize_dual(h, col=False, row_terms=2)
    w2q, _ = ops.quantize_dual(fc2.weight.detach(), col=False, block_2d=True)
    assert torch.equal(out, ops.gemm(gq, ops.two_term_weight(w2q), torch.bfloat16).float())
