"""ops.gemm_grouped: ragged-M (expert forward / input gradient) and ragged-K (expert weight gradient) grouped
NVFP4 GEMMs on the TMA main loop, against per-group dense GEMMs (bit-identical where the dense GEMM runs the
same loop: TMA, one split, K % 256 == 0) and an fp64 reference of the dequantized operands."""
import pytest
import torch

from conftest import requires_sm120
from nvfp4 import ops

pytestmark = requires_sm120


def _ref(a, b):
    return a.dequantize().double() @ b.dequantize().double().T


def _close(x, ref):
    err = (x.double() - ref).norm() / ref.norm().clamp_min(1e-30)
    assert err < (4e-3 if x.dtype == torch.bfloat16 else 1e-5), float(err)  # bf16: output rounding


@pytest.mark.parametrize("K", [256, 384, 1408])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_ragged_m(K, out_dtype):
    torch.manual_seed(K)
    N, sizes = 256, [128, 0, 384, 256, 128]          # rows per group (multiples of 128; group 1 empty)
    G = len(sizes)
    a_src = torch.randn(sum(sizes) + 128, K, device="cuda")  # one trailing tile of no group (zeros)
    a_src[-128:] = 0
    b_src = torch.randn(G * N, K, device="cuda")
    aq, _ = ops.quantize_dual(a_src, col=False)
    bq, _ = ops.quantize_dual(b_src, col=False)
    tg = torch.tensor([g for g, n in enumerate(sizes) for _ in range(n // 128)] + [0], device="cuda",
                      dtype=torch.int32)
    d = ops.gemm_grouped(aq, bq, "m", tg, group_rows=N, out_dtype=out_dtype)
    assert d.shape == (a_src.shape[0], N) and d.dtype == out_dtype
    full = _ref(aq, bq)
    off = 0
    for g, n in enumerate(sizes):
        if n:
            ref = full[off:off + n, g * N:(g + 1) * N]
            _close(d[off:off + n], ref)
            if K % 256 == 0:  # the dense TMA GEMM of this group: same loop, same summation order
                ag = ops.NVFP4Tensor(aq.data[off:off + n].contiguous(), aq.scales[off:off + n].contiguous(), aq.gdec)
                bg = ops.NVFP4Tensor(bq.data[g * N:(g + 1) * N].contiguous(), bq.scales[g * N:(g + 1) * N].contiguous(),
                                     bq.gdec)
                assert torch.equal(d[off:off + n], ops.gemm(ag, bg, out_dtype, impl="tma", splits=1))
        off += n
    assert torch.count_nonzero(d[-128:]) == 0


@pytest.mark.parametrize("accumulate", [False, True])
def test_ragged_k(accumulate):
    torch.manual_seed(1)
    R, N = 256, 384                                   # rows of A (per-group output rows), rows of B
    lens = [256, 128, 640, 384]                       # odd multiples of 128 use the half k-tile path
    starts = [0, 256, 512, 1152]                      # back to back, plus gaps that must be ignored
    Ktot = 1664
    a_src, b_src = torch.randn(R, Ktot, device="cuda"), torch.randn(N, Ktot, device="cuda")
    aq, _ = ops.quantize_dual(a_src, col=False)
    bq, _ = ops.quantize_dual(b_src, col=False)
    groups = [0, 2, 3, 1]                             # output blocks in any order
    meta = torch.tensor([[starts[i], lens[i], groups[i] * R] for i in range(4)], device="cuda", dtype=torch.int32)
    base = torch.randn(4 * R, N, device="cuda")
    out = base.clone() if accumulate else torch.zeros(4 * R, N, device="cuda")
    ops.gemm_grouped(aq, bq, "k", meta, out_dtype=torch.float32, out=out, accumulate=accumulate)
    for i in range(4):
        k0, n, r0 = starts[i], lens[i], groups[i] * R
        sl = lambda t: ops.NVFP4Tensor(t.data[:, k0 // 2:(k0 + n) // 2].contiguous(),  # noqa: E731
                                       t.scales[:, k0 // 16:(k0 + n) // 16].contiguous(), t.gdec)
        ref = _ref(sl(aq), sl(bq))
        got = out[r0:r0 + R] - (base[r0:r0 + R] if accumulate else 0)
        _close(got, ref)
        if n % 256 == 0 and not accumulate:
            assert torch.equal(out[r0:r0 + R], ops.gemm(sl(aq), sl(bq), torch.float32, impl="tma", splits=1))


@pytest.mark.parametrize("accumulate", [False, True])
def test_unused_tiles_and_empty_segments(accumulate):
    """Work units without k-tiles (tile group -1; ragged-K segments of length 0, at k0 % 256 == 0 and == 128)
    load nothing and store a zero tile (or leave an accumulated output unchanged), interleaved with real units
    so that the load cursors must skip them to stay in step with the consumers; more units than SMs."""
    torch.manual_seed(5)
    K, N = 512, 256
    a_src = torch.randn(128 * 300, K, device="cuda")
    b_src = torch.randn(3 * N, K, device="cuda")
    aq, _ = ops.quantize_dual(a_src, col=False)
    bq, _ = ops.quantize_dual(b_src, col=False)
    tg = torch.tensor([(t % 3) if t % 4 else -1 for t in range(300)], device="cuda", dtype=torch.int32)
    base = torch.randn(128 * 300, N, device="cuda")
    out = base.clone() if accumulate else torch.full_like(base, float("nan"))
    ops.gemm_grouped(aq, bq, "m", tg, group_rows=N, out_dtype=torch.float32, out=out, accumulate=accumulate)
    full = _ref(aq, bq)
    for t in range(300):
        rows, g = slice(128 * t, 128 * (t + 1)), int(tg[t])
        got = out[rows] - (base[rows] if accumulate else 0)
        if g < 0:
            assert torch.count_nonzero(got) == 0
        else:
            _close(got, full[rows, g * N:(g + 1) * N])
    # ragged K: R = 128 output rows per group, segments of length 0 between real ones
    R, Ktot = 128, 1024
    a2, _ = ops.quantize_dual(torch.randn(R, Ktot, device="cuda"), col=False)
    b2, _ = ops.quantize_dual(torch.randn(N, Ktot, device="cuda"), col=False)
    segs = [(0, 0), (0, 384), (384, 0), (384, 128), (512, 0), (512, 512), (1024, 0)] * 6
    meta = torch.tensor([[k0, n, i * R] for i, (k0, n) in enumerate(segs)], device="cuda", dtype=torch.int32)
    base2 = torch.randn(len(segs) * R, N, device="cuda")
    out2 = base2.clone() if accumulate else torch.full_like(base2, float("nan"))
    ops.gemm_grouped(a2, b2, "k", meta, out_dtype=torch.float32, out=out2, accumulate=accumulate)
    for i, (k0, n) in enumerate(segs):
        got = out2[i * R:(i + 1) * R] - (base2[i * R:(i + 1) * R] if accumulate else 0)
        if n == 0:
            assert torch.count_nonzero(got) == 0
        else:
            sl = lambda t: ops.NVFP4Tensor(t.data[:, k0 // 2:(k0 + n) // 2].contiguous(),  # noqa: E731
                                           t.scales[:, k0 // 16:(k0 + n) // 16].contiguous(), t.gdec)
            _close(got, _ref(sl(a2), sl(b2)))


def test_stack_transposed_matches_per_group_transpose():
    torch.manual_seed(2)
    G, N, K = 3, 256, 384
    w = torch.randn(G * N, K, device="cuda")
    _, wt = ops.quantize_dual(w, block_2d=True)
    st = ops.stack_transposed(wt, G)
    assert st.shape == (G * K, N)
    for g in range(G):
        want = w[g * N:(g + 1) * N].T
        got = st.dequantize()[g * K:(g + 1) * K]
        # same 16x16 blocks as the row quantization of the stacked weight
        assert torch.equal(got, ops.quantize_dual(w, block_2d=True, col=False)[0].dequantize()[g * N:(g + 1) * N].T)
        assert (got - want).norm() / want.norm() < 0.15


def test_rejects_bad_inputs():
    a, _ = ops.quantize_dual(torch.randn(128, 256, device="cuda"), col=False)
    b, _ = ops.quantize_dual(torch.randn(256, 256, device="cuda"), col=False)
    with pytest.raises(RuntimeError):
        ops.gemm_grouped(a, b, "m", torch.zeros(1, device="cuda", dtype=torch.int32), group_rows=100)
    with pytest.raises(RuntimeError):
        ops.gemm_grouped(a, b, "k", torch.zeros(1, 3, device="cuda", dtype=torch.int32))  # no out
