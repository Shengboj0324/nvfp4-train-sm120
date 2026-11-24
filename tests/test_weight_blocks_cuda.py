"""ops.set_weight_blocks("1d"): NVFP4 layers quantize weights in 1x16 blocks for fprop and re-quantize the
dequantized fprop weight along N for dgrad (Quartet II / Mosaic), instead of 16x16 2D blocks."""
import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4.linear import NVFP4Linear, quantized_weight

pytestmark = requires_sm120


@pytest.fixture(autouse=True)
def _restore():
    prev = ops.weight_blocks(), ops.four_six()
    yield
    ops.set_weight_blocks(prev[0])
    ops.set_four_six(prev[1])


@pytest.mark.parametrize("four_six", [False, True])
def test_one_d_weights_and_requantized_transpose(four_six):
    torch.manual_seed(0)
    w = torch.randn(384, 256, device="cuda")
    ops.set_weight_blocks("1d")
    ops.set_four_six(four_six)
    wq, wtq = quantized_weight(w, True, "cuda", None)
    want, _ = ops.quantize_dual(w, col=False)
    assert torch.equal(wq.data, want.data) and torch.equal(wq.scales, want.scales)
    _, want_t = ops.quantize_dual(wq.dequantize(), row=False, col=True)
    assert torch.equal(wtq.data, want_t.data) and torch.equal(wtq.scales, want_t.scales)
    # both represent the fprop weight up to the second rounding
    e = (wtq.dequantize().T - wq.dequantize()).norm() / wq.dequantize().norm()
    assert e < 0.12


def test_layer_uses_it_and_cache_key_includes_it():
    torch.manual_seed(1)
    lin = NVFP4Linear(256, 384, bias=False, cache_weight_quantization=True).cuda()
    x = torch.randn(64, 256, device="cuda", requires_grad=True)
    y2 = lin(x)
    ops.set_weight_blocks("1d")
    y1 = lin(x)
    wq, _ = ops.quantize_dual(lin.weight.detach(), col=False)
    xq, _ = ops.quantize_dual(x.detach().bfloat16(), col=False)
    assert torch.equal(y1.float(), ops.gemm(xq, wq, torch.bfloat16).float())
    assert not torch.equal(y1, y2)  # the 2D-quantized weight was not served from the cache
    y1.float().sum().backward()
    ref = F.linear(x.detach().float(), lin.weight.detach().float())
    assert F.cosine_similarity(y1.float().flatten(), ref.flatten(), dim=0) > 0.98
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_invalid_mode():
    with pytest.raises(ValueError):
        ops.set_weight_blocks("3d")
