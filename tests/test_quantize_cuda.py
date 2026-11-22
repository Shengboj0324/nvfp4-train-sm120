"""CUDA quantizer must be bit-exact against nvfp4.reference."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120


def _ref(x, *, transpose=False, block_2d=False, stochastic=False, sr_seed=0, rht_seed=None):
    y = x.T.contiguous() if transpose else x
    y = y.float()
    if rht_seed is not None:
        y = ref.apply_rht(y, rht_seed)
    return ref.quantize(y, block_2d=block_2d, stochastic=stochastic, seed=sr_seed)


def _check(x, **kw):
    q = ops.quantize(x, **kw)
    d, s, g = _ref(x, **kw)
    assert q.data.shape == d.shape and q.scales.shape == s.shape
    assert torch.equal(q.scales, s), "scales differ"
    assert torch.equal(q.gdec.reshape(()), g), "global scale differs"
    mism = (q.data != d).sum().item()
    assert mism == 0, f"{mism} data bytes differ"


SHAPES = [(16, 16), (32, 64), (100, 128), (256, 512), (1000, 48), (64, 4096)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rowwise(shape, dtype):
    _check(torch.randn(shape, device="cuda", dtype=dtype) * 3)


@pytest.mark.parametrize("shape", [(16, 16), (64, 32), (128, 100), (512, 256), (48, 1000)])
def test_transpose(shape):
    _check(torch.randn(shape, device="cuda", dtype=torch.bfloat16), transpose=True)


@pytest.mark.parametrize("transpose", [False, True])
def test_block_2d(transpose):
    _check(torch.randn(256, 384, device="cuda", dtype=torch.bfloat16), block_2d=True, transpose=transpose)


@pytest.mark.parametrize("seed", [0, 1, 2**40 + 17])
def test_stochastic(seed):
    _check(torch.randn(128, 256, device="cuda"), stochastic=True, sr_seed=seed)


@pytest.mark.parametrize("transpose", [False, True])
def test_rht(transpose):
    _check(torch.randn(192, 256, device="cuda", dtype=torch.bfloat16), rht_seed=99, transpose=transpose)


def test_rht_plus_stochastic_transpose():  # the wgrad path for the gradient operand
    _check(torch.randn(256, 128, device="cuda", dtype=torch.bfloat16), rht_seed=5, stochastic=True, sr_seed=11,
           transpose=True)


def test_edge_values():
    x = torch.randn(64, 64, device="cuda")
    x[0] = 0.0                        # all-zero block -> scale 0
    x[1, :16] = 1e-30                 # tiny block -> scale underflows to 0 in e4m3
    x[2, :16] = torch.tensor([6.0, -6.0] * 8)
    x[3, 0] = 1e4                     # dominates amax
    _check(x)
    _check(torch.zeros(32, 32, device="cuda"))  # amax == 0 -> genc = 1
