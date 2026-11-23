"""Llama-style fused sub-blocks (rmsnorm_linear, rmsnorm_swiglu_mlp): forward is exactly the documented
op composition; gradients are as close to fp32 as the unfused NVFP4 path; partial gradients, fused
wgrad accumulation, the weight cache, torch.compile and the usual RMSNorm module variants work."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4 import ops
from nvfp4.fused import rmsnorm_linear, rmsnorm_swiglu_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
D, H, T = 256, 768, 384


def _modules(seed=0, **kw):
    torch.manual_seed(seed)
    norm = torch.nn.RMSNorm(D, eps=1e-6).cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    lin = NVFP4Linear(D, 3 * D, bias=False, **kw).cuda()
    gate = NVFP4Linear(D, H, bias=False, **kw).cuda()
    up = NVFP4Linear(D, H, bias=False, **kw).cuda()
    down = NVFP4Linear(H, D, bias=False, **kw).cuda()
    return norm, lin, gate, up, down


def _x(seed=1, dtype=torch.float32):
    torch.manual_seed(seed)
    return (torch.randn(T, D, device="cuda") * 2 + 0.3).to(dtype).requires_grad_()


def test_rmsnorm_linear_forward_is_exact_composition():
    norm, lin, *_ = _modules()
    x = _x().detach()
    y = rmsnorm_linear(x, norm, lin)
    _, xq, _ = ops.rmsnorm_quantize_dual(x, norm.weight, norm.eps, col=False)
    wq, _ = ops.quantize_dual(lin.weight.detach(), col=False, block_2d=True)
    assert torch.equal(y, ops.gemm(xq, wq, torch.bfloat16).float())


def test_rmsnorm_swiglu_mlp_forward_is_exact_composition():
    norm, _, gate, up, down = _modules()
    x = _x().detach()
    out = rmsnorm_swiglu_mlp(x, norm, gate, up, down)
    _, xq, _ = ops.rmsnorm_quantize_dual(x, norm.weight, norm.eps, col=False)
    q = lambda lin: ops.quantize_dual(lin.weight.detach(), col=False, block_2d=True)[0]  # noqa: E731
    g, u = ops.gemm(xq, q(gate), torch.bfloat16), ops.gemm(xq, q(up), torch.bfloat16)
    aq, _ = ops.swiglu_quantize_dual(g, u, col=False)
    assert torch.equal(out, ops.gemm(aq, q(down), torch.bfloat16).float())


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
    norm, lin, gate, up, down = _modules()
    if which == "linear":
        mods = [norm, lin]
        fused = lambda x: rmsnorm_linear(x, norm, lin)  # noqa: E731
        unfused = lambda x: lin(norm(x))  # noqa: E731
    else:
        mods = [norm, gate, up, down]
        fused = lambda x: rmsnorm_swiglu_mlp(x, norm, gate, up, down)  # noqa: E731

        def unfused(x):
            n = norm(x)
            return down(F.silu(gate(n)) * up(n))

    def exact(x):  # fp32 everything
        n = F.rms_norm(x, (D,), norm.weight, norm.eps)
        if which == "linear":
            return n @ lin.weight.T
        return (F.silu(n @ gate.weight.T) * (n @ up.weight.T)) @ down.weight.T

    x = _x()
    g_exact = _grads(exact, mods, x)
    torch.manual_seed(9)
    g_fused = _grads(fused, mods, x)
    torch.manual_seed(9)
    g_unfused = _grads(unfused, mods, x)
    for ge, gf, gu in zip(g_exact, g_fused, g_unfused):
        cf, cu = _cos(gf, ge), _cos(gu, ge)
        assert cf > 0.95 and cf >= cu - 0.01, (cf, cu)


def test_seeded_determinism_and_bf16_input():
    norm, _, gate, up, down = _modules()
    for dtype in (torch.float32, torch.bfloat16):
        x = _x(dtype=dtype)
        torch.manual_seed(3)
        a = _grads(lambda x: rmsnorm_swiglu_mlp(x, norm, gate, up, down), [norm, gate, up, down], x)
        torch.manual_seed(3)
        b = _grads(lambda x: rmsnorm_swiglu_mlp(x, norm, gate, up, down), [norm, gate, up, down], x)
        assert all(torch.equal(u, v) for u, v in zip(a, b))
        assert a[0].dtype == dtype


def test_partial_gradients():
    norm, lin, gate, up, down = _modules()
    x = _x()
    with torch.no_grad():
        assert not rmsnorm_swiglu_mlp(x, norm, gate, up, down).requires_grad
    gate.weight.requires_grad_(False)
    norm.weight.requires_grad_(False)
    rmsnorm_swiglu_mlp(x, norm, gate, up, down).sum().backward()
    assert gate.weight.grad is None and norm.weight.grad is None
    assert x.grad is not None and up.weight.grad is not None and down.weight.grad is not None
    up.weight.grad = down.weight.grad = None
    up.weight.requires_grad_(False)  # only the down projection trains: no gradient through SwiGLU
    rmsnorm_swiglu_mlp(x.detach(), norm, gate, up, down).sum().backward()
    assert down.weight.grad is not None and up.weight.grad is None
    rmsnorm_linear(x.detach(), norm, lin).sum().backward()
    assert lin.weight.grad is not None


def test_fused_wgrad_accumulation_and_weight_cache_match():
    base = _modules()
    variants = [copy.deepcopy(base) for _ in range(3)]
    for m in variants[1][1:]:
        m.fuse_wgrad_accumulation = True
    for m in variants[2][1:]:
        m.fuse_wgrad_accumulation = True
        m.cache_weight_quantization = True
    x = _x()
    for norm, lin, gate, up, down in variants:
        torch.manual_seed(11)
        for _ in range(3):  # micro-batches accumulate into existing fp32 .grad
            (rmsnorm_linear(x, norm, lin).float().sum() + rmsnorm_swiglu_mlp(x, norm, gate, up, down).float().sum()
             ).backward()
    for other in variants[1:]:
        for ma, mb in zip(variants[0], other):
            for pa, pb in zip(ma.parameters(), mb.parameters()):
                assert torch.equal(pa.grad, pb.grad)
    assert sum(m.weight_quant_cache.hits for m in variants[2][1:]) == 4 * 2  # 4 linears x 2 later micro-batches


def test_compile_fullgraph():
    norm, lin, gate, up, down = _modules()

    def f(t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (rmsnorm_swiglu_mlp(t, norm, gate, up, down) + rmsnorm_linear(t, norm, lin)[:, :D]).float()

    torch._dynamo.reset()
    cf = torch.compile(f, fullgraph=True, backend=COMPILE_BACKEND)
    x = _x()
    y = cf(x)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(gate.weight.grad).all() and torch.isfinite(x.grad).all()
    ye = f(x.detach())
    assert ((y - ye).norm() / ye.norm()).item() < 1e-2


class _HFRMSNorm(torch.nn.Module):  # Hugging Face LlamaRMSNorm's attributes
    def __init__(self, d, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps


@pytest.mark.parametrize("variant", ["no_affine", "default_eps", "hf_style"])
def test_rmsnorm_module_variants(variant):
    _, lin, *_ = _modules()
    norm = {"no_affine": lambda: torch.nn.RMSNorm(D, eps=1e-5, elementwise_affine=False),
            "default_eps": lambda: torch.nn.RMSNorm(D),
            "hf_style": lambda: _HFRMSNorm(D, 1e-5)}[variant]().cuda()
    x = _x().detach()
    y = rmsnorm_linear(x, norm, lin)
    eps = 1e-5 if variant != "default_eps" else torch.finfo(torch.float32).eps
    _, xq, _ = ops.rmsnorm_quantize_dual(x, torch.ones(D, device="cuda"), eps, col=False)
    wq, _ = ops.quantize_dual(lin.weight.detach(), col=False, block_2d=True)
    assert torch.equal(y, ops.gemm(xq, wq, torch.bfloat16).float())


def test_rejects_unsupported_modules():
    norm, lin, gate, up, down = _modules()
    with pytest.raises(TypeError):
        rmsnorm_linear(_x(), norm, torch.nn.Linear(D, D, bias=False).cuda())
    with pytest.raises(ValueError, match="same out_features"):
        rmsnorm_swiglu_mlp(_x(), norm, gate, NVFP4Linear(D, 2 * H, bias=False).cuda(), down)
    with pytest.raises(ValueError, match="last dim"):
        rmsnorm_linear(_x(), torch.nn.RMSNorm(2 * D).cuda(), lin)
