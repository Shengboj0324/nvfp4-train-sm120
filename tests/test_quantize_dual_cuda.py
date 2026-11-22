"""The fused dual quantizer must be bit-exact against reference.quantize_dual (and therefore against two
separate single-output quantizations)."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120

# Configurations used by NVFP4Linear, plus single-output variants.
CONFIGS = {
    "activation": dict(rht_seed=7, col_pad=128),                                # fprop X (+ saved X^T)
    "gradient": dict(row_sr_seed=11, col_sr_seed=2**40 + 3, rht_seed=5, col_pad=128),  # dY
    "weight_2d": dict(block_2d=True),                                           # W and W^T
    "row_only": dict(col=False, row_sr_seed=3),
    "col_only": dict(row=False, rht_seed=9, col_sr_seed=4, col_pad=64),
    "col_plain": dict(col_pad=32),
}


def _check(x, **kw):
    got = ops.quantize_dual(x, **kw)
    want = ref.quantize_dual(x, **kw)
    for name, g, w in zip(("row", "col"), got, want):
        if w is None:
            assert g is None
            continue
        d, s, gd = w
        assert g.data.shape == d.shape and g.scales.shape == s.shape, name
        assert torch.equal(g.gdec.reshape(()), gd), f"{name}: global scale differs"
        assert torch.equal(g.scales, s), f"{name}: scales differ"
        mism = (g.data != d).sum().item()
        assert mism == 0, f"{name}: {mism} data bytes differ"
    return got


SHAPES = [(16, 16), (64, 64), (100, 128), (128, 48), (1000, 48), (130, 256), (512, 768)]


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dual_matches_reference(cfg, shape, dtype):
    kw = CONFIGS[cfg]
    if kw.get("block_2d") and shape[0] % 16:
        pytest.skip("2D blocks need rows % 16 == 0")
    torch.manual_seed(list(CONFIGS).index(cfg) * 1000 + shape[0] + shape[1])
    _check(torch.randn(shape, device="cuda", dtype=dtype) * 3, **kw)


def test_dual_equals_separate_calls():
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    r, c = ops.quantize_dual(x, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)
    r1 = ops.quantize(x, stochastic=True, sr_seed=1)
    xp = torch.nn.functional.pad(x, (0, 0, 0, (-300) % 128))
    c1 = ops.quantize(xp, transpose=True, stochastic=True, sr_seed=2, rht_seed=3)
    for a, b in ((r, r1), (c, c1)):
        assert torch.equal(a.data, b.data) and torch.equal(a.scales, b.scales) and torch.equal(a.gdec, b.gdec)


def test_2d_col_is_exact_transpose():
    w = torch.randn(384, 256, device="cuda")
    r, c = ops.quantize_dual(w, block_2d=True)
    assert torch.equal(r.dequantize().T, c.dequantize())


def test_edge_values():
    x = torch.randn(128, 64, device="cuda")
    x[0] = 0.0
    x[1, :16] = 1e-30
    x[2, :16] = torch.tensor([6.0, -6.0] * 8)
    x[3, 0] = 1e4
    for kw in CONFIGS.values():
        _check(x, **kw)
    _check(torch.zeros(48, 32, device="cuda"), **CONFIGS["gradient"])


def test_unaligned_input_is_copied():
    base = torch.randn(64 * 64 + 4, device="cuda", dtype=torch.bfloat16)
    x = base[4:].view(64, 64)  # contiguous but only 8-byte aligned
    _check(x, **CONFIGS["activation"])


def test_rejects_bad_shapes():
    with pytest.raises(RuntimeError):
        ops.quantize_dual(torch.randn(32, 40, device="cuda"))
    with pytest.raises(RuntimeError):
        ops.quantize_dual(torch.randn(40, 32, device="cuda"), block_2d=True)
