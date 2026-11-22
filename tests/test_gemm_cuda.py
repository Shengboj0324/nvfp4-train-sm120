"""Block-scaled MMA GEMM vs the reference (dequantize -> fp64 matmul)."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4 import reference as ref

pytestmark = requires_sm120


def _run(m, n, k, out_dtype=torch.float32, scale_a=1.0, scale_b=1.0):
    a = torch.randn(m, k, device="cuda") * scale_a
    b = torch.randn(n, k, device="cuda") * scale_b
    qa, qb = ops.quantize(a), ops.quantize(b)
    d = ops.gemm(qa, qb, out_dtype=out_dtype).float()
    r = ref.gemm(qa.as_tuple(), qb.as_tuple())
    return d, r, a @ b.T


@pytest.mark.parametrize("mnk", [(128, 128, 128), (256, 384, 512), (16, 8, 128), (200, 72, 256),
                                 (1, 1, 128), (517, 1031, 640), (128, 128, 4096)])
def test_matches_reference_fp32_out(mnk):
    d, r, _ = _run(*mnk)
    # products of e2m1*e4m3 values are exact; only fp32 accumulation order differs
    err = (d - r).abs().max().item()
    assert err <= 1e-5 * r.abs().max().item() + 1e-6, err


def test_bf16_output():
    d, r, _ = _run(256, 256, 1024, out_dtype=torch.bfloat16)
    assert torch.allclose(d, r, rtol=1e-2, atol=1e-2 * r.abs().max().item())


def test_scale_extremes():
    # very different magnitudes exercise the per-tensor and per-block scale paths
    d, r, _ = _run(128, 256, 512, scale_a=1e-3, scale_b=50.0)
    assert (d - r).abs().max().item() <= 1e-5 * r.abs().max().item()


def test_varied_block_scales():
    # rows/blocks with wildly different magnitudes: every scale byte must be routed to the right block
    m, n, k = 128, 128, 512
    a = torch.randn(m, k, device="cuda") * torch.logspace(-3, 3, k // 16, device="cuda").repeat_interleave(16)
    b = torch.randn(n, k, device="cuda") * torch.logspace(2, -2, n, device="cuda")[:, None]
    qa, qb = ops.quantize(a), ops.quantize(b)
    d = ops.gemm(qa, qb, out_dtype=torch.float32)
    r = ref.gemm(qa.as_tuple(), qb.as_tuple())
    assert (d - r).abs().max().item() <= 1e-5 * r.abs().max().item()


def test_accuracy_vs_full_precision():
    d, _, full = _run(512, 512, 2048)
    assert ref.snr_db(full, d) > 10.0


def test_out_and_accumulate():
    a = torch.randn(200, 256, device="cuda")
    b = torch.randn(136, 256, device="cuda")
    qa, qb = ops.quantize(a), ops.quantize(b)
    d = ops.gemm(qa, qb, torch.float32)
    out = torch.full((200, 136), 0.5, device="cuda")
    ops.gemm(qa, qb, out=out, accumulate=True)
    assert torch.equal(out, 0.5 + d)
    out2 = torch.empty(200, 136, device="cuda", dtype=torch.bfloat16)
    ops.gemm(qa, qb, out=out2)
    assert torch.equal(out2, ops.gemm(qa, qb, torch.bfloat16))
    with pytest.raises(RuntimeError):
        ops.gemm(qa, qb, out=out2, accumulate=True)  # bf16 accumulation is not supported


# --- main loops and split-K -------------------------------------------------------------------------
# K values cover: one k-tile, partial scale loads (K/256 not a multiple of 4), long K; M/N cover
# partial CTA tiles.
IMPL_SHAPES = [(128, 128, 256), (200, 72, 768), (517, 1031, 1280), (384, 256, 4096), (130, 260, 2304)]


def _ops(m, n, k, seed=0):
    torch.manual_seed(seed)
    qa = ops.quantize(torch.randn(m, k, device="cuda"))
    qb = ops.quantize(torch.randn(n, k, device="cuda"))
    return qa, qb


@pytest.mark.parametrize("mnk", IMPL_SHAPES)
def test_tma_and_cp_async_loops_are_bit_identical(mnk):
    # Both loops add the same 64-deep MMA products in the same order, so unsplit results are identical.
    qa, qb = _ops(*mnk)
    d_tma = ops.gemm(qa, qb, torch.float32, impl="tma", splits=1)
    d_cp = ops.gemm(qa, qb, torch.float32, impl="cp.async", splits=1)
    assert torch.equal(d_tma, d_cp)
    r = ref.gemm(qa.as_tuple(), qb.as_tuple())
    assert (d_tma - r).abs().max().item() <= 1e-5 * r.abs().max().item() + 1e-6


@pytest.mark.parametrize("impl", ["tma", "cp.async"])
@pytest.mark.parametrize("splits", [2, 3, 5])
def test_split_k_matches_reference_and_is_deterministic(impl, splits):
    qa, qb = _ops(300, 200, 2560)
    d = ops.gemm(qa, qb, torch.float32, impl=impl, splits=splits)
    r = ref.gemm(qa.as_tuple(), qb.as_tuple())
    assert (d - r).abs().max().item() <= 1e-5 * r.abs().max().item()
    for _ in range(3):  # partials are summed in split order, whichever CTA arrives last
        assert torch.equal(ops.gemm(qa, qb, torch.float32, impl=impl, splits=splits), d)
    # bf16 output goes through the same reduction
    assert torch.equal(ops.gemm(qa, qb, torch.bfloat16, impl=impl, splits=splits), d.bfloat16())


def test_split_k_accumulate_equals_add():
    qa, qb = _ops(256, 192, 4096)
    d = ops.gemm(qa, qb, torch.float32, splits=4)
    out = torch.randn(256, 192, device="cuda")
    want = out + d
    ops.gemm(qa, qb, out=out, accumulate=True, splits=4)
    assert torch.equal(out, want)


def test_split_k_more_splits_than_k_tiles():
    qa, qb = _ops(64, 64, 256)  # a single k-tile: the split count is clamped to 1
    assert torch.equal(ops.gemm(qa, qb, torch.float32, splits=8), ops.gemm(qa, qb, torch.float32, splits=1))


def test_auto_split_for_small_grids():
    # 768 x 768 output = 36 CTA tiles on an 82-SM GPU: auto mode splits K; result stays within fp32
    # reassociation of the unsplit GEMM.
    qa, qb = _ops(768, 768, 8192)
    auto = ops.gemm(qa, qb, torch.float32)
    one = ops.gemm(qa, qb, torch.float32, splits=1)
    assert (auto - one).abs().max().item() <= 1e-5 * one.abs().max().item()


def test_tma_rejects_unsupported_k_and_falls_back():
    qa, qb = _ops(128, 128, 384)  # K % 256 != 0: no TMA descriptor for the scale rows
    with pytest.raises(RuntimeError):
        ops.gemm(qa, qb, impl="tma")
    r = ref.gemm(qa.as_tuple(), qb.as_tuple())
    d = ops.gemm(qa, qb, torch.float32)  # auto falls back to the cp.async loop
    assert (d - r).abs().max().item() <= 1e-5 * r.abs().max().item()
