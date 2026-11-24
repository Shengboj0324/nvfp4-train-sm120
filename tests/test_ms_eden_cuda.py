"""MS-EDEN for the wgrad operands (ops.set_ms_eden): the dual quantizer's RHT outputs use a 64-point RHT and the
stochastically rounded one an EDEN-corrected, stochastically rounded block scale with round-to-nearest elements.
The kernels must match reference.quantize_dual(ms_eden=True) bit for bit (host and GPU seeds, fused producers),
the 64-point rotation must cancel in dY^T X, and the gradient operand must carry about half the error of element
stochastic rounding."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120


@pytest.fixture(autouse=True)
def _restore():
    prev = ops.ms_eden(), ops.four_six()
    yield
    ops.set_ms_eden(prev[0])
    ops.set_four_six(prev[1])


def _same(g, w, name=""):
    d, s, gd = w
    assert torch.equal(g.gdec.reshape(()), gd), f"{name}: global scale differs"
    assert torch.equal(g.scales, s), f"{name}: {(g.scales != s).sum().item()} scales differ"
    assert (g.data != d).sum().item() == 0, f"{name}: data differs"


CONFIGS = {
    "gradient": dict(row_sr_seed=11, col_sr_seed=2**40 + 3, rht_seed=5, col_pad=128),
    "activation": dict(rht_seed=7, col_pad=128),
    "col_only": dict(row=False, rht_seed=9, col_sr_seed=4, col_pad=64),
}


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", [(64, 64), (100, 128), (512, 768), (1000, 48)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("four_six", [False, True])
def test_dual_matches_reference(cfg, shape, dtype, four_six):
    kw = CONFIGS[cfg]
    torch.manual_seed(shape[0] * 3 + shape[1])
    x = torch.randn(shape, device="cuda", dtype=dtype) * 3
    ops.set_ms_eden(True)
    ops.set_four_six(four_six)
    got = ops.quantize_dual(x, **kw)
    want = ref.quantize_dual(x, ms_eden=True, four_six=four_six, **kw)
    for name, g, w in zip(("row", "col"), got, want):
        if w is None:
            assert g is None
            continue
        _same(g, w, name)


def test_device_seeds_match_host_seeds():
    torch.manual_seed(1)
    x = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
    ops.set_ms_eden(True)
    seeds = [torch.tensor([v], device="cuda", dtype=torch.int64) for v in (21, 2**50 + 7, 13)]
    a = ops.quantize_dual(x, row_sr_seed=seeds[0], col_sr_seed=seeds[1], rht_seed=seeds[2], col_pad=128)
    b = ops.quantize_dual(x, row_sr_seed=21, col_sr_seed=2**50 + 7, rht_seed=13, col_pad=128)
    for g, h in zip(a, b):
        assert torch.equal(g.data, h.data) and torch.equal(g.scales, h.scales) and torch.equal(g.gdec, h.gdec)


def test_producer_path_uses_it():
    torch.manual_seed(2)
    x = torch.randn(300, 768, device="cuda", dtype=torch.bfloat16)
    w, b = torch.randn(768, device="cuda") * 0.5 + 1, torch.randn(768, device="cuda") * 0.1
    ops.set_ms_eden(True)
    mean, rstd, xq, xtq = ops.layernorm_quantize_dual(x, w, b, 1e-5, rht_seed=3, col_pad=128)
    want = ref.layernorm_quantize_dual(x, w, b, mean, rstd, rht_seed=3, col_pad=128, ms_eden=True)
    _same(xq, want[0], "row")
    _same(xtq, want[1], "col")


def test_rotation_cancels_in_wgrad_and_error_halves():
    # dW = dY^T X with both operands rotated along the tokens by the same 64-point RHT
    torch.manual_seed(3)
    T, N, K = 1024, 256, 384
    dy = torch.randn(T, N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    exact = dy.float().T @ x.float()
    errs = {}
    for eden in (False, True):
        ops.set_ms_eden(eden)
        _, dyt = ops.quantize_dual(dy, row=False, col_sr_seed=5, rht_seed=9, col_pad=128)
        _, xt = ops.quantize_dual(x, row=False, rht_seed=9, col_pad=128)
        dw = ops.gemm(dyt, xt, torch.float32)
        errs[eden] = ((dw - exact).norm() / exact.norm()).item()
        # the gradient operand alone, in the rotated domain
        y = ref.apply_rht(torch.nn.functional.pad(dy.float(), (0, 0, 0, 0)).T.contiguous(), 9, 64 if eden else 16)
        errs[(eden, "op")] = ((dyt.dequantize() - y).norm() / y.norm()).item()
    assert errs[True] < 0.2 and errs[False] < 0.2  # the rotation cancels (no rotation error left)
    assert errs[(True, "op")] < 0.8 * errs[(False, "op")], errs  # MS-EDEN ~ RTN error vs SR's ~1.4x RTN


def _rht64_matrix(seed):
    h = torch.ones(1, 1)
    while h.shape[0] < 64:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (ref.rht_signs(seed, 64).unsqueeze(1) * h / 8.0).cuda()  # y_blk = x_blk @ M, M orthogonal


def test_unbiased_over_rotation_and_scale_rounding():
    # mean over (RHT seed, SR seed) pairs of the un-rotated dequantized gradient operand: its deviation from the
    # input is no larger than element-wise SR's (unbiased) at the same number of samples
    torch.manual_seed(4)
    dy = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    dy[:, 3] *= 6  # an outlier channel
    want = dy.T
    stats = {}
    n = 256
    for eden in (False, True):
        ops.set_ms_eden(eden)
        acc = torch.zeros_like(want)
        for s in range(n):
            _, q = ops.quantize_dual(dy, row=False, col_sr_seed=1000 + s, rht_seed=s, col_pad=64)
            yq = q.dequantize()
            if eden:
                m = _rht64_matrix(s)
                acc += (yq.reshape(-1, 64) @ m.T).reshape(want.shape)
            else:
                acc += _unrht16(yq, s)
        stats[eden] = ((acc / n - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt()).item()
    assert stats[True] < stats[False] * 1.1 + 1e-3, stats


def _unrht16(y, seed):
    m = ref.rht_matrix(seed).cuda()  # y_blk = x_blk @ m, m orthogonal
    return (y.reshape(-1, 16) @ m.T).reshape(y.shape)


def test_col_pad_must_cover_the_64_point_rht():
    ops.set_ms_eden(True)
    x = torch.randn(80, 64, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        ops.quantize_dual(x, rht_seed=1, col_pad=16)  # 80 rows: not a multiple of 64
    ops.quantize_dual(x, col_pad=16)  # without the RHT nothing changes
