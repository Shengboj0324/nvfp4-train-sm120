"""nvfp4.fused sub-blocks: forward is exactly the documented op composition; gradients are as close
to fp32 as the unfused NVFP4 path; seeds, partial gradients and fused wgrad accumulation behave."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import ops
from nvfp4.fused import layernorm_linear, layernorm_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
D, H, T = 256, 1024, 384


def _modules(seed=0, **kw):
    torch.manual_seed(seed)
    ln = torch.nn.LayerNorm(D).cuda()
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.normal_(0, 0.1)
    lin = NVFP4Linear(D, 3 * D, bias=False, **kw).cuda()
    fc1 = NVFP4Linear(D, H, bias=False, **kw).cuda()
    fc2 = NVFP4Linear(H, D, bias=False, **kw).cuda()
    return ln, lin, fc1, fc2


def _x(seed=1):
    torch.manual_seed(seed)
    return (torch.randn(T, D, device="cuda") * 2 + 0.3).requires_grad_()


def test_layernorm_linear_forward_is_exact_composition():
    ln, lin, _, _ = _modules()
    x = _x().detach()
    y = layernorm_linear(x, ln, lin)
    _, _, xq, _ = ops.layernorm_quantize_dual(x, ln.weight, ln.bias, ln.eps, col=False)
    wq, _ = ops.quantize_dual(lin.weight.detach(), col=False, block_2d=True)
    assert torch.equal(y, ops.gemm(xq, wq, torch.bfloat16).float())


def test_layernorm_mlp_forward_is_exact_composition():
    ln, _, fc1, fc2 = _modules()
    x = _x().detach()
    out = layernorm_mlp(x, ln, fc1, fc2)
    _, _, xq, _ = ops.layernorm_quantize_dual(x, ln.weight, ln.bias, ln.eps, col=False)
    w1q, _ = ops.quantize_dual(fc1.weight.detach(), col=False, block_2d=True)
    h = ops.gemm(xq, w1q, torch.bfloat16)
    gq, _ = ops.gelu_quantize_dual(h, col=False)
    w2q, _ = ops.quantize_dual(fc2.weight.detach(), col=False, block_2d=True)
    assert torch.equal(out, ops.gemm(gq, w2q, torch.bfloat16).float())


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _grads(fn, mods, x):
    for m in mods:
        m.zero_grad(set_to_none=True)
    x.grad = None
    out = fn(x)
    torch.manual_seed(5)
    out.backward(torch.randn_like(out) * 1e-2)
    return [x.grad] + [p.grad for m in mods for p in m.parameters()]


@pytest.mark.parametrize("which", ["linear", "mlp"])
def test_gradients_as_good_as_unfused(which):
    ln, lin, fc1, fc2 = _modules()
    mods = [ln, lin] if which == "linear" else [ln, fc1, fc2]
    fused = (lambda x: layernorm_linear(x, ln, lin)) if which == "linear" else (lambda x: layernorm_mlp(x, ln, fc1, fc2))
    unfused = (lambda x: lin(ln(x))) if which == "linear" else (
        lambda x: fc2(F.gelu(fc1(ln(x)), approximate="tanh")))

    def exact(x):  # fp32 everything
        y = F.layer_norm(x, (D,), ln.weight, ln.bias, ln.eps)
        if which == "linear":
            return y @ lin.weight.T
        return F.gelu(y @ fc1.weight.T, approximate="tanh") @ fc2.weight.T

    x = _x()
    g_exact = _grads(exact, mods, x)
    torch.manual_seed(9)
    g_fused = _grads(fused, mods, x)
    torch.manual_seed(9)
    g_unfused = _grads(unfused, mods, x)
    for ge, gf, gu in zip(g_exact, g_fused, g_unfused):
        cf, cu = _cos(gf, ge), _cos(gu, ge)
        assert cf > 0.95 and cf >= cu - 0.01, (cf, cu)


def test_seeded_determinism():
    ln, _, fc1, fc2 = _modules()
    x = _x()
    torch.manual_seed(3)
    a = _grads(lambda x: layernorm_mlp(x, ln, fc1, fc2), [ln, fc1, fc2], x)
    torch.manual_seed(3)
    b = _grads(lambda x: layernorm_mlp(x, ln, fc1, fc2), [ln, fc1, fc2], x)
    assert all(torch.equal(u, v) for u, v in zip(a, b))


def test_partial_gradients():
    ln, lin, fc1, fc2 = _modules()
    x = _x()
    with torch.no_grad():
        y = layernorm_mlp(x, ln, fc1, fc2)
    assert not y.requires_grad
    fc1.weight.requires_grad_(False)
    ln.weight.requires_grad_(False)
    layernorm_mlp(x, ln, fc1, fc2).sum().backward()
    assert fc1.weight.grad is None and ln.weight.grad is None
    assert x.grad is not None and fc2.weight.grad is not None and ln.bias.grad is not None
    xd = x.detach()  # no input gradient: only the weight gradient is computed
    layernorm_linear(xd, ln, lin).sum().backward()
    assert lin.weight.grad is not None


def test_fused_wgrad_accumulation_matches():
    mods_a = _modules(fuse_wgrad_accumulation=False)
    mods_b = copy.deepcopy(mods_a)
    for m in mods_b[1:]:
        m.fuse_wgrad_accumulation = True
    x = _x()
    for mods in (mods_a, mods_b):
        ln, lin, fc1, fc2 = mods
        torch.manual_seed(11)
        for _ in range(2):  # second micro-batch accumulates into existing fp32 .grad
            (layernorm_linear(x, ln, lin).float().sum() + layernorm_mlp(x, ln, fc1, fc2).float().sum()).backward()
    for ma, mb in zip(mods_a, mods_b):
        for pa, pb in zip(ma.parameters(), mb.parameters()):
            assert torch.equal(pa.grad, pb.grad)


def test_rejects_unsupported_modules():
    ln, lin, fc1, fc2 = _modules()
    with pytest.raises(TypeError):
        layernorm_linear(_x(), ln, torch.nn.Linear(D, D, bias=False).cuda())
    with pytest.raises(ValueError):
        layernorm_linear(_x(), ln, NVFP4Linear(D, D, bias=True).cuda())
