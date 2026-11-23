"""LayerNorm / GELU / GELU-backward fused into the dual quantizer: bit-exact against the reference op
followed by reference.quantize_dual (computed on the GPU, so tanh is the same libdevice tanhf)."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120

CONFIGS = {
    "activation": dict(rht_seed=7, col_pad=128),
    "gradient": dict(row_sr_seed=11, col_sr_seed=2**40 + 3, rht_seed=5, col_pad=128),
    "row_only": dict(col=False),
    "col_only": dict(row=False, rht_seed=9, col_sr_seed=4, col_pad=64),
}
SHAPES = [(64, 64), (100, 128), (1000, 48), (130, 256), (512, 768)]


def _check(got, want):
    for name, g, w in zip(("row", "col"), got, want):
        if w is None:
            assert g is None
            continue
        d, s, gd = w
        assert torch.equal(g.gdec.reshape(()), gd), f"{name}: global scale differs"
        assert torch.equal(g.scales, s), f"{name}: scales differ"
        assert (g.data != d).sum().item() == 0, f"{name}: data differs"


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_layernorm_quantize_dual(cfg, shape, dtype):
    torch.manual_seed(shape[0] + shape[1])
    x = (torch.randn(shape, device="cuda") * 2 + 0.5).to(dtype)
    w = torch.randn(shape[1], device="cuda") * 0.5 + 1
    b = torch.randn(shape[1], device="cuda") * 0.1
    mean, rstd, *got = ops.layernorm_quantize_dual(x, w, b, 1e-5, **CONFIGS[cfg])
    m_ref, r_ref = ref.layernorm_stats(x, 1e-5)
    assert torch.allclose(mean, m_ref, rtol=1e-5, atol=1e-6)
    assert torch.allclose(rstd, r_ref, rtol=1e-5)
    _check(got, ref.layernorm_quantize_dual(x, w, b, mean, rstd, **CONFIGS[cfg]))


def test_layernorm_stats_match_torch():
    x = torch.randn(300, 1024, device="cuda") * 3 + 2
    mean, rstd, _, _ = ops.layernorm_quantize_dual(x, torch.ones(1024, device="cuda"), torch.zeros(1024, device="cuda"),
                                                   1e-5, col=False)
    y = torch.nn.functional.layer_norm(x, (1024,), eps=1e-5)
    mine = ref.layernorm_apply(x, mean, rstd, torch.ones(1024, device="cuda"), torch.zeros(1024, device="cuda"))
    assert (mine - y).abs().max().item() < 1e-5


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_quantize_dual(cfg, shape):
    torch.manual_seed(shape[0] * 7 + shape[1])
    h = (torch.randn(shape, device="cuda") * 3).bfloat16()
    _check(ops.gelu_quantize_dual(h, **CONFIGS[cfg]), ref.gelu_quantize_dual(h, **CONFIGS[cfg]))


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_backward_quantize_dual(cfg, shape):
    torch.manual_seed(shape[0] * 13 + shape[1])
    h = (torch.randn(shape, device="cuda") * 3).bfloat16()
    dy = (torch.randn(shape, device="cuda") * 1e-3).bfloat16()
    _check(ops.gelu_backward_quantize_dual(dy, h, **CONFIGS[cfg]),
           ref.gelu_backward_quantize_dual(dy, h, **CONFIGS[cfg]))


def test_gelu_spec_matches_torch():
    h = torch.randn(4096, device="cuda") * 4
    assert (ref.gelu_tanh(h) - torch.nn.functional.gelu(h, approximate="tanh")).abs().max().item() < 1e-6
    hh = h.clone().requires_grad_()
    torch.nn.functional.gelu(hh, approximate="tanh").backward(torch.ones_like(hh))
    assert (ref.gelu_tanh_backward(torch.ones_like(h), h) - hh.grad).abs().max().item() < 1e-6


def test_padding_rows_stay_zero():
    # LayerNorm of a zero padding row would be `bias`; the col output must still see zeros there.
    x = torch.randn(100, 64, device="cuda")
    w, b = torch.ones(64, device="cuda"), torch.full((64,), 3.0, device="cuda")
    mean, rstd, _, col = ops.layernorm_quantize_dual(x, w, b, 1e-5, row=False, col_pad=128)
    assert col.dequantize()[:, 100:].abs().sum().item() == 0.0


# --- Llama-style producers: RMSNorm, SwiGLU and its backward --------------------------------------------

@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_quantize_dual(cfg, shape, dtype):
    x = (torch.randn(shape, device="cuda") * 3 + 0.5).to(dtype)
    w = torch.rand(shape[1], device="cuda") + 0.5
    rstd, *got = ops.rmsnorm_quantize_dual(x, w, 1e-6, **CONFIGS[cfg])
    assert torch.allclose(rstd, ref.rmsnorm_stats(x, 1e-6), rtol=1e-5)
    _check(got, ref.rmsnorm_quantize_dual(x, w, rstd, **CONFIGS[cfg]))


def test_rmsnorm_stats_match_torch():
    x = torch.randn(300, 1024, device="cuda") * 2
    rstd, _, _ = ops.rmsnorm_quantize_dual(x, torch.ones(1024, device="cuda"), 1e-6, col=False)
    want = torch.nn.functional.rms_norm(x, (1024,), eps=1e-6)
    assert (ref.rmsnorm_apply(x, rstd, torch.ones(1024, device="cuda")) - want).abs().max().item() < 1e-5


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_swiglu_quantize_dual(cfg, shape):
    g = (torch.randn(shape, device="cuda") * 4).bfloat16()
    u = torch.randn(shape, device="cuda").bfloat16()
    _check(ops.swiglu_quantize_dual(g, u, **CONFIGS[cfg]), ref.swiglu_quantize_dual(g, u, **CONFIGS[cfg]))


@pytest.mark.parametrize("gate", [True, False])
@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_swiglu_backward_quantize_dual(gate, cfg, shape):
    dy = torch.randn(shape, device="cuda").bfloat16()
    g = (torch.randn(shape, device="cuda") * 4).bfloat16()
    u = torch.randn(shape, device="cuda").bfloat16()
    _check(ops.swiglu_backward_quantize_dual(dy, g, u, gate=gate, **CONFIGS[cfg]),
           ref.swiglu_backward_quantize_dual(dy, g, u, gate=gate, **CONFIGS[cfg]))


def test_swiglu_spec_matches_torch():
    g = torch.linspace(-30, 30, 4001, device="cuda")
    u = torch.randn(4001, device="cuda")
    assert (ref.swiglu(g, u) - torch.nn.functional.silu(g) * u).abs().max().item() < 1e-5
    gg, uu = g.clone().requires_grad_(), u.clone().requires_grad_()
    dy = torch.randn(4001, device="cuda")
    (torch.nn.functional.silu(gg) * uu).backward(dy)
    assert (ref.swiglu_backward_gate(dy, g, u) - gg.grad).abs().max().item() < 1e-5
    assert (ref.swiglu_backward_up(dy, g) - uu.grad).abs().max().item() < 1e-5


def test_swiglu_inputs_validated():
    g = torch.randn(64, 64, device="cuda").bfloat16()
    with pytest.raises(RuntimeError, match="shaped like"):
        ops.swiglu_quantize_dual(g, torch.randn(64, 32, device="cuda").bfloat16())


# --- GELU with a bias in front, exact (erf) GELU, bias gradient from the same read -----------------------

@pytest.mark.parametrize("approximate", ["tanh", "none"])
@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_bias_and_exact(approximate, cfg, shape):
    h = (torch.randn(shape, device="cuda") * 3).bfloat16()
    for bias in (None, torch.randn(shape[1], device="cuda")):
        _check(ops.gelu_quantize_dual(h, bias=bias, approximate=approximate, **CONFIGS[cfg]),
               ref.gelu_quantize_dual(h, bias=bias, approximate=approximate, **CONFIGS[cfg]))
        dy = torch.randn(shape, device="cuda").bfloat16()
        _check(ops.gelu_backward_quantize_dual(dy, h, bias=bias, approximate=approximate, **CONFIGS[cfg]),
               ref.gelu_backward_quantize_dual(dy, h, bias=bias, approximate=approximate, **CONFIGS[cfg]))


@pytest.mark.parametrize("cfg", ["activation", "row_only", "col_only"])
@pytest.mark.parametrize("shape", SHAPES)
def test_gelu_backward_bias_grad(cfg, shape):
    h = (torch.randn(shape, device="cuda") * 3).bfloat16()
    dy = torch.randn(shape, device="cuda").bfloat16()
    bias = torch.randn(shape[1], device="cuda")
    *_, db = ops.gelu_backward_quantize_dual(dy, h, bias=bias, bias_grad=True, **CONFIGS[cfg])
    want = ref.gelu_backward(dy, h, bias=bias).double().sum(0)
    assert db.shape == (shape[1],) and db.dtype == torch.float32
    assert torch.allclose(db.double(), want, rtol=1e-5, atol=1e-4 * want.abs().max().item())
    # deterministic: a fixed summation order, no atomics
    *_, db2 = ops.gelu_backward_quantize_dual(dy, h, bias=bias, bias_grad=True, **CONFIGS[cfg])
    assert torch.equal(db, db2)


def test_gelu_exact_spec_matches_torch():
    h = torch.linspace(-12, 12, 4001, device="cuda")
    assert (ref.gelu_erf(h) - torch.nn.functional.gelu(h)).abs().max().item() < 1e-6
    hh = h.clone().requires_grad_()
    torch.nn.functional.gelu(hh).backward(torch.ones_like(hh))
    assert (ref.gelu_erf_backward(torch.ones_like(h), h) - hh.grad).abs().max().item() < 1e-6
    with pytest.raises(ValueError, match="approximate"):
        ops.gelu_quantize_dual(h.view(1, -1)[:, :4000].reshape(40, 100).bfloat16(), approximate="sigmoid")
