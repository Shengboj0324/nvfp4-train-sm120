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
