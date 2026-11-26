"""Dynamic tile claiming in the persistent TMA GEMMs (ops.set_dynamic_tiles): the same bits as the fixed
assignment, for dense GEMMs (bf16 / fp32 out, split-K, accumulate, bias + residual), grouped GEMMs with unused
tiles (runs longer than the claim ring) and ragged-K groups, under SM margins, with another kernel holding SMs
on a second stream, back to back on several streams, and under CUDA graph capture (fixed assignment)."""
import pytest
import torch

from nvfp4 import ops

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and ops.available()), reason="needs the CUDA extension")
DEV = "cuda"


def q(*shape):
    return ops.quantize(torch.randn(*shape, device=DEV))


@pytest.fixture
def restore():
    yield
    ops.set_dynamic_tiles(False)
    ops.set_sm_margin(0)


def cases():
    torch.manual_seed(0)
    a, b = q(2048, 1024), q(1536, 1024)
    a2, b2 = q(640, 2048), q(384, 2048)  # few tiles: split-K
    bias = torch.randn(1536, device=DEV)
    res = torch.randn(2048, 1536, device=DEV, dtype=torch.bfloat16)
    acc0 = torch.randn(2048, 1536, device=DEV)
    # 70 tiles: unused runs of 40 (longer than the claim ring) and at the start / end
    tiles = torch.tensor([-1] * 3 + [0] * 9 + [-1] * 40 + [1] * 7 + [2] * 6 + [-1] * 5, device=DEV, dtype=torch.int32)
    am = q(70 * 128, 1024)
    bg = q(3 * 768, 1024)
    ak, bk = q(256, 4096), q(384, 4096)
    segs = torch.tensor([[0, 1024, 0], [1024, 0, 256], [1024, 1536, 512], [2560, 1536, 768]], device=DEV,
                        dtype=torch.int32)

    def run():
        out = {"bf16": ops.gemm(a, b, impl="tma"), "fp32": ops.gemm(a, b, torch.float32, impl="tma"),
               "epi": ops.gemm(a, b, impl="tma", bias=bias, residual=res),
               "split": ops.gemm(a2, b2, torch.float32, impl="tma", splits=4)}
        acc = acc0.clone()
        ops.gemm(a, b, torch.float32, out=acc, accumulate=True, impl="tma")
        out["accum"] = acc
        out["grouped_m"] = ops.gemm_grouped(am, bg, "m", tiles, group_rows=768)
        ok = torch.zeros(1024, 384, device=DEV)
        out["grouped_k"] = ops.gemm_grouped(ak, bk, "k", segs, out=ok, out_dtype=torch.float32)
        acck = torch.randn(1024, 384, device=DEV)
        ops.gemm_grouped(ak, bk, "k", segs, out=acck, out_dtype=torch.float32, accumulate=True)
        out["grouped_k_acc"] = acck
        return out
    return run


def same(x, y):
    assert x.keys() == y.keys()
    for k in x:
        assert torch.equal(x[k], y[k]), k


def test_dynamic_matches_static(restore):
    run = cases()
    ops.set_dynamic_tiles(False)
    assert not ops.dynamic_tiles()
    torch.manual_seed(1)
    want = run()
    ops.set_dynamic_tiles(True)
    assert ops.dynamic_tiles()
    for margin in (0, 7, 70, 150):  # 150: one CTA; 70: 12 CTAs (the small GEMMs claim too)
        ops.set_sm_margin(margin)
        torch.manual_seed(1)
        same(run(), want)


def test_dynamic_with_sms_held_and_streams(restore):
    run = cases()
    ops.set_dynamic_tiles(False)
    torch.manual_seed(1)
    want = run()
    ops.set_dynamic_tiles(True)
    side = torch.cuda.Stream()
    big = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(3):
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # another kernel on the SMs while the GEMMs launch
            for _ in range(4):
                big @ big
        torch.manual_seed(1)
        same(run(), want)
    torch.cuda.synchronize()
    # the same GEMMs on two streams at once: each stream has its own claim counters
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    for s in (s1, s2):
        s.wait_stream(torch.cuda.current_stream())
    outs = []
    for s in (s1, s2):
        with torch.cuda.stream(s):
            torch.manual_seed(1)
            outs.append(run())
    torch.cuda.synchronize()
    for o in outs:
        same(o, want)


def test_graph_capture_uses_fixed_assignment(restore):
    a, b = q(2048, 1024), q(1536, 1024)
    want = ops.gemm(a, b, impl="tma")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ops.gemm(a, b, impl="tma")  # warm-up outside the capture
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        y = ops.gemm(a, b, impl="tma")
    for _ in range(3):
        g.replay()
        ops.gemm(a, b, impl="tma")  # eager launches between replays
    torch.cuda.synchronize()
    assert torch.equal(y, want)
