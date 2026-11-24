"""Four Over Six block scales (ops.set_four_six): every round-to-nearest NVFP4 output picks, per 16-element block,
the smaller-error of max -> 6 and max -> 4 with the tensor scale 6 * 256 / amax. The kernels must match
reference.quantize(four_six=True) bit for bit, leave stochastically rounded outputs unchanged, and lower the
quantization error of round-to-nearest outputs."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120


@pytest.fixture(autouse=True)
def _restore():
    prev = ops.four_six()
    yield
    ops.set_four_six(prev)


def _same(g, w, name=""):
    d, s, gd = w
    assert torch.equal(g.gdec.reshape(()), gd), f"{name}: global scale differs"
    assert torch.equal(g.scales, s), f"{name}: scales differ"
    assert (g.data != d).sum().item() == 0, f"{name}: data differs"


CONFIGS = {
    "activation": dict(rht_seed=7, col_pad=128),
    "gradient": dict(row_sr_seed=11, col_sr_seed=2**40 + 3, rht_seed=5, col_pad=128),
    "weight_2d": dict(block_2d=True),
    "mixed": dict(row_sr_seed=3, col_pad=64),
}


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", [(64, 64), (100, 128), (512, 768), (1000, 48)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dual_matches_reference(cfg, shape, dtype):
    kw = CONFIGS[cfg]
    if kw.get("block_2d") and shape[0] % 16:
        pytest.skip("2D blocks need rows % 16 == 0")
    torch.manual_seed(shape[0] * 7 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=dtype) * 3
    ops.set_four_six(True)
    got = ops.quantize_dual(x, **kw)
    want = ref.quantize_dual(x, four_six=True, **kw)
    for name, g, w in zip(("row", "col"), got, want):
        _same(g, w, name)
    # stochastically rounded outputs are the plain ones (4/6 applies to round-to-nearest only)
    ops.set_four_six(False)
    plain = ops.quantize_dual(x, **kw)
    for name, g, p, sr in zip(("row", "col"), got, plain, ("row_sr_seed" in kw, "col_sr_seed" in kw)):
        if sr:
            assert torch.equal(g.data, p.data) and torch.equal(g.scales, p.scales), name


@pytest.mark.parametrize("kw", [{}, {"transpose": True}, {"block_2d": True}, {"transpose": True, "rht_seed": 3}])
def test_single_output_matches_reference(kw):
    torch.manual_seed(1)
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) * 2
    ops.set_four_six(True)
    got = ops.quantize(x, **kw)
    y = x.float().T.contiguous() if kw.get("transpose") else x.float()
    if "rht_seed" in kw:
        y = ref.apply_rht(y, kw["rht_seed"])
    _same(got, ref.quantize(y, block_2d=kw.get("block_2d", False), four_six=True))


def test_producer_paths_use_it():
    # the fused LayerNorm quantizer goes through the same kernel: its output equals 4/6 quantization of the
    # reference LayerNorm
    torch.manual_seed(2)
    x = torch.randn(300, 768, device="cuda", dtype=torch.bfloat16)
    w, b = torch.randn(768, device="cuda") * 0.5 + 1, torch.randn(768, device="cuda") * 0.1
    ops.set_four_six(True)
    mean, rstd, xq, _ = ops.layernorm_quantize_dual(x, w, b, 1e-5, col=False)
    want = ref.layernorm_quantize_dual(x, w, b, mean, rstd, col=False, four_six=True)[0]
    _same(xq, want)


def test_lowers_error_and_switch_restores():
    torch.manual_seed(3)
    x = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    base = ops.quantize(x)
    with ops.four_six_scales(True):
        assert ops.four_six()
        q46 = ops.quantize(x)
    assert not ops.four_six()
    e0 = (base.dequantize() - x.float()).pow(2).mean().item()
    e1 = (q46.dequantize() - x.float()).pow(2).mean().item()
    assert e1 < 0.92 * e0, (e0, e1)  # Quartet II: 9.0e-3 -> 7.6e-3 relative MSE on N(0, 1)
    # the tensor scale uses 256 instead of 448
    amax = x.float().abs().max()
    assert torch.allclose(q46.gdec.reshape(()), amax / (6 * 256), rtol=1e-6)


def test_reference_picks_the_smaller_error_per_block():
    # per block, the 4/6 result is never worse than either candidate (its max -> 6 candidate is the plain
    # quantization with the tensor scale 6 * 256 / amax, i.e. with amax replaced by amax * 448 / 256)
    torch.manual_seed(4)
    x = torch.randn(64, 256) * torch.rand(64, 1) * 4
    d, s, g = ref.quantize(x, four_six=True)
    err46 = (ref.dequantize(d, s, g) - x).pow(2).reshape(64, 16, 16).sum(-1)
    d6, s6, g6 = ref.quantize(x, amax=x.abs().max() * (448.0 / 256.0))
    err6 = (ref.dequantize(d6, s6, g6) - x).pow(2).reshape(64, 16, 16).sum(-1)
    assert (err46 <= err6 * (1 + 1e-5) + 1e-12).all()
    assert err46.sum() < 0.95 * err6.sum()
