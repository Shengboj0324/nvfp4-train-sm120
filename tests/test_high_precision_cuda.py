"""ops.set_high_precision: while on, NVFP4Linear and the nvfp4.fused sub-blocks compute exactly what the
modules they replace would (the precision switch for the end of training), and switching back restores the
NVFP4 path."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4.fused import layernorm_linear, layernorm_mlp, swiglu_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120


@pytest.fixture(autouse=True)
def _restore():
    yield
    ops.set_high_precision(False)


def _pair(i, o, bias=True):
    ref = torch.nn.Linear(i, o, bias=bias).cuda()
    return ref, NVFP4Linear.from_linear(ref, fuse_wgrad_accumulation=True)


@pytest.mark.parametrize("autocast", [False, True])
def test_linear_matches_nn_linear(autocast):
    torch.manual_seed(0)
    ref, lin = _pair(256, 384)
    x = torch.randn(4, 32, 256, device="cuda")
    r = torch.randn(4, 32, 384, device="cuda")
    xr, xl = x.clone().requires_grad_(), x.clone().requires_grad_()
    ops.set_high_precision(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        yr = r + ref(xr)
        yl = lin(xl, residual=r)
    assert yl.dtype == yr.dtype and torch.equal(yl, yr)
    g = torch.randn_like(yr)
    yr.backward(g)
    yl.backward(g)
    assert torch.equal(xl.grad, xr.grad)
    assert torch.equal(lin.weight.grad, ref.weight.grad) and torch.equal(lin.bias.grad, ref.bias.grad)


def test_switch_back_restores_nvfp4():
    torch.manual_seed(1)
    ref, lin = _pair(256, 256, bias=False)
    x = torch.randn(64, 256, device="cuda")
    y4 = lin(x)
    ops.set_high_precision(True)
    yh = lin(x)
    ops.set_high_precision(False)
    assert torch.equal(lin(x), y4) and not torch.equal(yh, y4)
    assert torch.equal(yh, ref(x))


def test_fused_norm_linear_chunks_and_residual_handle():
    torch.manual_seed(2)
    ln = torch.nn.LayerNorm(256).cuda()
    ref, lin = _pair(256, 3 * 256, bias=False)
    proj_ref, proj = _pair(256, 256, bias=False)
    x = torch.randn(2, 64, 256, device="cuda")
    xr, xl = x.clone().requires_grad_(), x.clone().requires_grad_()
    ops.set_high_precision(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q, k, v, h = layernorm_linear(xl, ln, lin, chunks=3, return_residual=True)
        yl = proj(q * k + v, residual=h)
        qr, kr, vr = ref(ln(xr)).split(256, dim=-1)
        yr = xr + proj_ref(qr * kr + vr)
    assert torch.equal(yl, yr)
    yl.float().sum().backward()
    yr.float().sum().backward()
    assert torch.equal(xl.grad, xr.grad) and torch.equal(lin.weight.grad, ref.weight.grad)


@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_fused_mlps(kind):
    torch.manual_seed(3)
    d, f = 256, 512
    x = torch.randn(128, d, device="cuda")
    xr, xl = x.clone().requires_grad_(), x.clone().requires_grad_()
    ops.set_high_precision(True)
    if kind == "gelu":
        ln = torch.nn.LayerNorm(d).cuda()
        (r1, l1), (r2, l2) = _pair(d, f), _pair(f, d)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yl = layernorm_mlp(xl, ln, l1, l2, residual=True)
            yr = xr + r2(F.gelu(r1(ln(xr)), approximate="tanh"))
        mods = [(l1, r1), (l2, r2)]
    else:
        (rg, lg), (ru, lu), (rd, ld) = _pair(d, f, False), _pair(d, f, False), _pair(f, d, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yl = swiglu_mlp(xl, lg, lu, ld)
            yr = rd(F.silu(rg(xr)) * ru(xr))
        mods = [(lg, rg), (lu, ru), (ld, rd)]
    assert torch.equal(yl, yr)
    yl.float().sum().backward()
    yr.float().sum().backward()
    assert torch.equal(xl.grad, xr.grad)
    for m, r in mods:
        assert torch.equal(m.weight.grad, r.weight.grad)


def test_training_step_after_switch_keeps_working():
    torch.manual_seed(4)
    _, lin = _pair(256, 256)
    lin2 = copy.deepcopy(lin)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    x = torch.randn(64, 256, device="cuda")
    for hp in (False, True, False):
        ops.set_high_precision(hp)
        lin(x).float().pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    assert not torch.equal(lin.weight, lin2.weight) and torch.isfinite(lin.weight).all()
