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
