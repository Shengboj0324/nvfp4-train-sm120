"""NVFP4Linear: CUDA path == emulated recipe, and gradients track full precision."""
import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4.linear import NVFP4Linear, convert_linears


def _cos(a, b):
    return F.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def _run(layer, x, dy, seed):
    torch.manual_seed(seed)  # fixes the SR / RHT seeds drawn in backward
    x = x.clone().requires_grad_(True)
    layer.zero_grad(set_to_none=True)
    y = layer(x)
    y.backward(dy)
    return y.detach().float(), x.grad.float(), layer.weight.grad.float()


@requires_sm120
@pytest.mark.parametrize("m", [256, 200])  # 200: exercises the wgrad padding path
def test_cuda_matches_emulation(m):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 384, bias=True, device="cuda")
    cu = NVFP4Linear.from_linear(lin, backend="cuda")
    em = NVFP4Linear.from_linear(lin, backend="emulate")
    x = torch.randn(m, 256, device="cuda")
    dy = torch.randn(m, 384, device="cuda")
    yc, dxc, dwc = _run(cu, x, dy, 123)
    ye, dxe, dwe = _run(em, x, dy, 123)
    for c, e in ((yc, ye), (dxc, dxe), (dwc, dwe)):
        # identical quantized operands; differences are fp32-accumulation order + bf16 output rounding
        assert (c - e).abs().max().item() <= 1e-2 * e.abs().max().item()


@requires_sm120
def test_tracks_full_precision():
    torch.manual_seed(1)
    lin = torch.nn.Linear(1024, 1024, bias=False, device="cuda")
    q = NVFP4Linear.from_linear(lin)
    x = torch.randn(4096, 1024, device="cuda")
    dy = torch.randn(4096, 1024, device="cuda") * 1e-3
    xr = x.clone().requires_grad_(True)
    yr = lin(xr)
    yr.backward(dy)
    y, dx, dw = _run(q, x, dy, 7)
    assert _cos(y, yr) > 0.98
    assert _cos(dx, xr.grad) > 0.98
    assert _cos(dw, lin.weight.grad) > 0.97


@requires_sm120
def test_stochastic_rounding_averages_out():
    torch.manual_seed(2)
    lin = torch.nn.Linear(512, 512, bias=False, device="cuda")
    q = NVFP4Linear.from_linear(lin)
    x = torch.randn(1024, 512, device="cuda")
    dy = torch.randn(1024, 512, device="cuda")
    xr = x.clone().requires_grad_(True)
    lin(xr).backward(dy)
    single = _run(q, x, dy, 0)[2]
    mean = sum(_run(q, x, dy, s)[2] for s in range(32)) / 32
    e1 = (single - lin.weight.grad).norm() / lin.weight.grad.norm()
    e32 = (mean - lin.weight.grad).norm() / lin.weight.grad.norm()
    assert e32 < 0.6 * e1, (e1.item(), e32.item())  # SR + RHT noise is zero-mean


@requires_sm120
def test_3d_input_bias_and_convert():
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256),
                                torch.nn.Linear(256, 10)).cuda()
    ref_out = model(torch.zeros(1, 256, device="cuda"))  # noqa: F841
    names = convert_linears(model)
    assert names == ["0", "2"]  # the 256->10 head is not 128-aligned and stays nn.Linear
    x = torch.randn(4, 64, 256, device="cuda")
    out = model(x)
    assert out.shape == (4, 64, 10)
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rejects_unaligned_dims():
    with pytest.raises(ValueError):
        NVFP4Linear(100, 128)


def test_emulate_backend_runs_on_cpu():
    torch.manual_seed(4)
    q = NVFP4Linear(128, 128, backend="emulate")
    x = torch.randn(130, 128, requires_grad=True)
    q(x).float().sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(q.weight.grad).all()
