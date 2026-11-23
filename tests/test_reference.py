"""CPU-only tests of the NVFP4 reference (the spec). Run in CI without a GPU."""
import torch
import pytest

from nvfp4 import reference as ref


def test_e2m1_grid_roundtrip():
    grid = ref.E2M1_GRID
    vals = torch.cat([grid, -grid[1:]])
    assert torch.equal(ref.e2m1_decode(ref.e2m1_round_rne(vals)), vals)


@pytest.mark.parametrize("x,expected", [
    (0.25, 0.0), (0.2500001, 0.5), (0.75, 1.0), (1.25, 1.0), (1.2500001, 1.5), (1.75, 2.0),
    (2.5, 2.0), (2.5000002, 3.0), (3.5, 4.0), (5.0, 4.0), (5.0000005, 6.0), (100.0, 6.0), (-0.75, -1.0),
])
def test_e2m1_rne_ties_and_saturation(x, expected):
    got = ref.e2m1_decode(ref.e2m1_round_rne(torch.tensor([x]))).item()
    assert got == expected


def test_negative_zero_encodes_as_positive_zero():
    assert ref.e2m1_round_rne(torch.tensor([-0.1])).item() == 0


def test_quantize_error_is_bounded():
    torch.manual_seed(0)
    x = torch.randn(64, 256)
    y = ref.fake_quant(x)
    # 4-bit with fp8 block scales: ~ 9-10 dB-per-bit regime, expect roughly 19-22 dB SNR on Gaussians
    assert 17.0 < ref.snr_db(x, y) < 25.0
    # per-block max is representable to within the e4m3 scale rounding (<= 6.25% relative)
    xb = x.reshape(64, 16, 16).abs().amax(-1)
    yb = y.reshape(64, 16, 16).abs().amax(-1)
    assert torch.all((yb - xb).abs() <= 0.0625 * xb + 1e-6)


def test_block_2d_is_transpose_invariant():
    torch.manual_seed(1)
    w = torch.randn(128, 64)
    a = ref.fake_quant(w, block_2d=True)
    b = ref.fake_quant(w.T.contiguous(), block_2d=True).T
    assert torch.equal(a, b)


def test_rht_matches_dense_matrix_and_is_orthogonal():
    torch.manual_seed(2)
    m = ref.rht_matrix(1234)
    assert torch.allclose(m @ m.T, torch.eye(16), atol=1e-6)
    x = torch.randn(8, 64)
    dense = (x.reshape(8, 4, 16) @ m).reshape(8, 64)
    assert torch.allclose(ref.apply_rht(x, 1234), dense, atol=1e-5)
    # inner products along the transformed dim are preserved -> GEMM result unchanged
    a, b = torch.randn(32, 64), torch.randn(16, 64)
    assert torch.allclose(ref.apply_rht(a, 7) @ ref.apply_rht(b, 7).T, a @ b.T, atol=1e-4)


def test_stochastic_rounding_is_unbiased():
    torch.manual_seed(3)
    x = torch.rand(1, 16) * 5.5 + 0.2
    x[0, 0] = x.abs().max() * 1.0001  # keep the block max (exactly representable) away from the probes
    n = 4000
    acc = torch.zeros_like(x)
    for s in range(n):
        acc += ref.fake_quant(x, stochastic=True, seed=s)
    mean = acc / n
    det = ref.fake_quant(x)
    # SR mean converges to x (bias << RNE error); RNE has systematic error
    assert (mean - x).abs()[0, 1:].max() < 0.06
    assert (mean - x).abs()[0, 1:].mean() < 0.5 * (det - x).abs()[0, 1:].mean()


def test_gemm_reference_matches_dequantized_matmul():
    torch.manual_seed(4)
    a, b = torch.randn(32, 128), torch.randn(48, 128)
    qa, qb = ref.quantize(a), ref.quantize(b)
    d = ref.gemm(qa, qb)
    assert torch.allclose(d, ref.dequantize(*qa) @ ref.dequantize(*qb).T, rtol=1e-4, atol=1e-4)
    assert ref.snr_db(a @ b.T, d) > 10.0


def test_block_scale_uses_ieee_division():
    # amax 14 -> genc 192; block max 6.25: IEEE 6.25 / 6 * 192 is exactly 200, the E4M3 midpoint of 192
    # and 208, which rounds to even (192, code 0x74). Multiplying by a rounded 1/6 instead gives 208.
    x = torch.zeros(2, 32)
    x[0, 0] = 14.0
    x[1, 16] = 6.25
    _, s, _ = ref.quantize(x)
    assert s[1, 1].item() == 0x74


def test_quantize_dual_spec():
    torch.manual_seed(0)
    x = torch.randn(100, 64)
    (rd, rs, rg), (cd, cs, cg) = ref.quantize_dual(x, rht_seed=3, col_sr_seed=4, col_pad=128)
    assert rd.shape == (100, 32) and cd.shape == (64, 64)
    want = ref.quantize(ref.apply_rht(torch.nn.functional.pad(x, (0, 0, 0, 28)).T.contiguous(), 3),
                        stochastic=True, seed=4)
    assert all(torch.equal(a, b) for a, b in zip((cd, cs, cg), want))
    # The RHT mixes the block holding tokens 96..111 (4 real, 12 padding), so only fully padded blocks are zero.
    assert torch.equal(ref.dequantize(cd, cs, cg)[:, 112:].abs().sum(), torch.tensor(0.0))
    # 2D: the col output is the exact transpose of the row output
    (rd, rs, rg), (cd, cs, cg) = ref.quantize_dual(torch.randn(64, 48), block_2d=True)
    assert torch.equal(ref.dequantize(rd, rs, rg).T, ref.dequantize(cd, cs, cg))
    assert ref.quantize_dual(x, col=False)[1] is None and ref.quantize_dual(x, row=False)[0] is None


def test_producer_specs_match_torch():
    # The fused-producer specs (op-by-op fp32) agree with torch's own ops to fp32 rounding.
    torch.manual_seed(0)
    h = torch.randn(2048) * 4
    assert (ref.gelu_tanh(h) - torch.nn.functional.gelu(h, approximate="tanh")).abs().max().item() < 2e-6
    hh = h.clone().requires_grad_()
    torch.nn.functional.gelu(hh, approximate="tanh").backward(torch.ones_like(hh))
    assert (ref.gelu_tanh_backward(torch.ones_like(h), h) - hh.grad).abs().max().item() < 2e-6
    x = torch.randn(64, 256) * 3 + 1
    w, b = torch.rand(256) + 0.5, torch.randn(256) * 0.1
    mean, rstd = ref.layernorm_stats(x, 1e-5)
    y = torch.nn.functional.layer_norm(x, (256,), w, b, 1e-5)
    assert (ref.layernorm_apply(x, mean, rstd, w, b) - y).abs().max().item() < 1e-5
    (rd, _, _), (cd, _, _) = ref.layernorm_quantize_dual(x, w, b, mean, rstd, rht_seed=1, col_pad=128)
    assert rd.shape == (64, 128) and cd.shape == (256, 64)


def test_non_finite_amax_gives_non_finite_scale():
    # NaN must not quantize to finite values (fp32 max semantics would drop it).
    genc, gdec = ref.global_scales(torch.tensor(float("nan")))
    assert torch.isnan(genc) and torch.isnan(gdec)
    genc, gdec = ref.global_scales(torch.tensor(float("inf")))
    assert genc == 0 and torch.isinf(gdec)
    genc, gdec = ref.global_scales(torch.tensor(0.0))
    assert genc == 1 and gdec == 1
    x = torch.randn(32, 64)
    x[5, 5] = float("nan")
    d, s, g = ref.quantize(x)
    assert torch.isnan(g)
