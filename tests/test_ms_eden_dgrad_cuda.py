"""MS-EDEN on the dgrad operands (ops.set_ms_eden(True, dgrad=True)): stochastically rounded row outputs of the
dual quantizer get a 64-point RHT along the row (signs from the row seed ^ ROW_RHT_SALT) and EDEN scales,
bit-exact against reference.quantize_dual(ms_eden_dgrad=True) (host / GPU seeds, both outputs at once, multi-part
inputs, the backward producers); the layers rotate the dgrad weight operand identically, so dX = dY W up to
quantization error, in NVFP4Linear (CUDA = emulated backend, bit for bit), the fused sub-blocks and the MoE."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import convert_linears, ops
from nvfp4 import reference as ref
from nvfp4.fused import layernorm_mlp, swiglu_mlp
from nvfp4.linear import NVFP4Linear
from nvfp4.moe import MoE

pytestmark = requires_sm120


@pytest.fixture(autouse=True)
def _restore():
    prev = ops.ms_eden(), ops.ms_eden_dgrad(), ops.four_six()
    yield
    ops.set_ms_eden(prev[0], dgrad=prev[1])
    ops.set_four_six(prev[2])


def _same(g, w, name=""):
    d, s, gd = w
    assert torch.equal(g.gdec.reshape(()), gd), f"{name}: global scale differs"
    assert torch.equal(g.scales, s), f"{name}: {(g.scales != s).sum().item()} scales differ"
    assert (g.data != d).sum().item() == 0, f"{name}: data differs"


CONFIGS = {
    "gradient": dict(row_sr_seed=11, col_sr_seed=2**40 + 3, rht_seed=5, col_pad=128),
    "row_only": dict(col=False, row_sr_seed=2**45 + 9),
    "row_rtn": dict(col=False),  # no row SR: untouched
}


@pytest.mark.parametrize("cfg", list(CONFIGS))
@pytest.mark.parametrize("shape", [(64, 64), (100, 128), (512, 768), (1000, 192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dual_matches_reference(cfg, shape, dtype):
    kw = CONFIGS[cfg]
    torch.manual_seed(shape[0] + shape[1])
    x = torch.randn(shape, device="cuda", dtype=dtype) * 3
    x[:, 5] *= 8  # an outlier channel
    ops.set_ms_eden(True, dgrad=True)
    got = ops.quantize_dual(x, **kw)
    want = ref.quantize_dual(x, ms_eden=True, ms_eden_dgrad=True, **kw)
    for name, g, w in zip(("row", "col"), got, want):
        if w is None:
            assert g is None
            continue
        _same(g, w, name)


def test_device_seeds_and_parts():
    torch.manual_seed(1)
    ops.set_ms_eden(True, dgrad=True)
    parts = [torch.randn(200, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    seeds = [torch.tensor([v], device="cuda", dtype=torch.int64) for v in (21, 2**50 + 7, 13)]
    a = ops.quantize_dual(parts, row_sr_seed=seeds[0], col_sr_seed=seeds[1], rht_seed=seeds[2], col_pad=128)
    b = ops.quantize_dual(parts, row_sr_seed=21, col_sr_seed=2**50 + 7, rht_seed=13, col_pad=128)
    want = ref.quantize_dual(torch.cat(parts, 1), row_sr_seed=21, col_sr_seed=2**50 + 7, rht_seed=13, col_pad=128,
                             ms_eden=True, ms_eden_dgrad=True)
    for g, h, w, name in zip(a, b, want, ("row", "col")):
        assert torch.equal(g.data, h.data) and torch.equal(g.scales, h.scales) and torch.equal(g.gdec, h.gdec)
        _same(g, w, name)


def test_backward_producers():
    torch.manual_seed(2)
    ops.set_ms_eden(True, dgrad=True)
    dy, h = (torch.randn(300, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    kw = dict(row_sr_seed=3, col_sr_seed=4, rht_seed=5, col_pad=128)
    got = ops.gelu_backward_quantize_dual(dy, h, **kw)
    want = ref.gelu_backward_quantize_dual(dy, h, ms_eden=True, ms_eden_dgrad=True, **kw)
    for g, w, name in zip(got, want, ("row", "col")):
        _same(g, w, "gelu " + name)
    g_, u_ = h, torch.randn_like(h)
    for gate in (True, False):
        got = ops.swiglu_backward_quantize_dual(dy, g_, u_, gate=gate, **kw)
        want = ref.swiglu_backward_quantize_dual(dy, g_, u_, gate=gate, ms_eden=True, ms_eden_dgrad=True, **kw)
        for g, w, name in zip(got, want, ("row", "col")):
            _same(g, w, f"swiglu gate={gate} {name}")


def test_rotation_cancels_in_dgrad_and_estimator_is_unbiased():
    # dX = dY W from Q(R dY) and the rotated re-quantized weight; averaged over row seeds the error of the
    # gradient operand alone falls like an unbiased estimator's
    torch.manual_seed(3)
    T, N, K = 256, 512, 384
    dy = torch.randn(T, N, device="cuda")
    w = torch.randn(N, K, device="cuda") * 0.05
    wq, _ = ops.quantize_dual(w, block_2d=True, col=False)
    exact = dy @ wq.dequantize()
    ops.set_ms_eden(True, dgrad=True)
    errs, acc = [], torch.zeros_like(exact)
    for s in range(64):
        dyq, _ = ops.quantize_dual(dy, col=False, row_sr_seed=1000 + s)
        wt = ops.rotated_weight_t(wq, 1000 + s)
        dx = ops.gemm(dyq, wt, torch.float32)
        errs.append(((dx - exact).norm() / exact.norm()).item())
        acc += dx
    assert max(errs) < 0.2  # the rotation cancels: only quantization error left
    mean_err = ((acc / 64 - exact).norm() / exact.norm()).item()
    assert mean_err < 0.5 * sum(errs) / len(errs), (mean_err, errs[:4])  # averaging over seeds reduces it


def test_linear_cuda_matches_emulated_backend():
    torch.manual_seed(4)
    ops.set_ms_eden(True, dgrad=True)
    lin = NVFP4Linear(256, 384, bias=False).cuda()
    emu = copy.deepcopy(lin)
    emu.backend = "emulate"
    x = torch.randn(96, 256, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    g = torch.randn(96, 384, device="cuda")
    torch.manual_seed(5)
    lin(x).backward(g)
    torch.manual_seed(5)
    emu(x2).backward(g)
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-3, atol=1e-4)  # GEMM summation order only
    ref_dx = g @ lin.weight.detach()
    assert F.cosine_similarity(x.grad.flatten(), ref_dx.flatten(), dim=0) > 0.97


@pytest.mark.parametrize("kind", ["linear", "gelu_mlp", "swiglu_mlp", "moe"])
def test_input_gradients_track_bf16(kind):
    torch.manual_seed(6)
    d, f = 256, 512
    x = torch.randn(512, d, device="cuda")
    if kind == "linear":
        ref_m = torch.nn.Linear(d, f).cuda()
        m = NVFP4Linear.from_linear(ref_m)
        fwd, fwd_ref = m, ref_m
    elif kind == "gelu_mlp":
        ln = torch.nn.LayerNorm(d).cuda()
        r1, r2 = torch.nn.Linear(d, f).cuda(), torch.nn.Linear(f, d).cuda()
        l1, l2 = NVFP4Linear.from_linear(r1), NVFP4Linear.from_linear(r2)
        fwd = lambda t: layernorm_mlp(t, ln, l1, l2)  # noqa: E731
        fwd_ref = lambda t: r2(F.gelu(r1(ln(t)), approximate="tanh"))  # noqa: E731
    elif kind == "swiglu_mlp":
        rs = [torch.nn.Linear(d, f, bias=False).cuda(), torch.nn.Linear(d, f, bias=False).cuda(),
              torch.nn.Linear(f, d, bias=False).cuda()]
        ls = [NVFP4Linear.from_linear(r) for r in rs]
        fwd = lambda t: swiglu_mlp(t, *ls)  # noqa: E731
        fwd_ref = lambda t: rs[2](F.silu(rs[0](t)) * rs[1](t))  # noqa: E731
    else:
        fwd_ref = MoE(d, f, 4, top_k=2).cuda()
        fwd = copy.deepcopy(fwd_ref)
        convert_linears(fwd)
    cos = {}
    for dgrad in (False, True):
        ops.set_ms_eden(True, dgrad=dgrad)
        xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ya, yb = fwd(xa), fwd_ref(xb)
        torch.manual_seed(7)
        go = torch.randn_like(yb.float())
        ya.float().backward(go)
        yb.float().backward(go)
        cos[dgrad] = F.cosine_similarity(xa.grad.flatten(), xb.grad.flatten(), dim=0).item()
    assert cos[True] > 0.95 and cos[True] > cos[False] - 0.02, cos


def test_requires_ms_eden_and_c_multiple_of_64():
    with pytest.raises(ValueError):
        ops.set_ms_eden(False, dgrad=True)
    ops.set_ms_eden(True, dgrad=True)
    with pytest.raises(RuntimeError):
        ops.quantize_dual(torch.randn(64, 80, device="cuda"), col=False, row_sr_seed=1)
    ops.quantize_dual(torch.randn(64, 80, device="cuda"), col=False)  # RTN rows: unaffected
