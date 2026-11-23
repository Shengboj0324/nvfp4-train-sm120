"""v0.8 sub-block fusions: q/k/v chunks of the qkv projection, residual adds fused into the epilogue (and
their gradients into the norm backward), norm-free MLPs (post-norm blocks), SwiGLU biases, and the bias /
residual epilogue of NVFP4Linear. Forward results equal the unfused composition bit for bit; gradients
equal it bit for bit where the same values are quantized, and are as close to fp32 as the unfused path
otherwise (an fp32 residual-stream gradient is quantized without the bf16 rounding)."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4.fused import (gelu_mlp, layernorm_linear, layernorm_mlp, rmsnorm_linear, rmsnorm_swiglu_mlp,
                         swiglu_mlp)
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
D, H, T, HEADS = 256, 768, 256, 2


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _x(seed=1, dtype=torch.float32, b=2):
    torch.manual_seed(seed)
    return (torch.randn(b, T, D, device="cuda") * 2 + 0.3).to(dtype).requires_grad_()


def _grads(fn, mods, x, seed=5):
    for m in mods:
        m.zero_grad(set_to_none=True)
    x.grad = None
    out = fn(x)
    torch.manual_seed(seed)
    out.float().backward(torch.randn_like(out.float()) * 1e-2)
    return [x.grad] + [p.grad for m in mods for p in m.parameters()], out


# --- NVFP4Linear: bias and residual in the epilogue ---------------------------------------------------

@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("res_dtype", [torch.float32, torch.bfloat16])
def test_linear_residual_and_bias_epilogue(autocast, res_dtype):
    torch.manual_seed(0)
    lin = NVFP4Linear(D, D).cuda()
    with torch.no_grad():
        lin.bias.normal_()
    x = _x(dtype=res_dtype)
    r = _x(seed=2, dtype=res_dtype)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        torch.manual_seed(3)
        fused = lin(x, residual=r)
        torch.manual_seed(3)
        plain = lin.forward(x)
    assert torch.equal(fused, r + plain)
    g = torch.randn_like(fused)
    gx, gw, gb, gr = torch.autograd.grad(fused, (x, lin.weight, lin.bias, r), g)
    assert torch.equal(gr, g)
    assert torch.allclose(gb, g.float().reshape(-1, D).sum(0), rtol=1e-4, atol=1e-3)
    assert gx is not None and gw is not None


def test_linear_bias_gradient_matches_autograd():
    torch.manual_seed(0)
    lin = NVFP4Linear(D, 2 * D).cuda()
    ref = torch.nn.Linear(D, 2 * D).cuda()
    ref.load_state_dict(lin.state_dict())
    x = _x()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        g = torch.randn(2, T, 2 * D, device="cuda")
        (lin(x).float() * g).sum().backward()
        (ref(x).float() * g).sum().backward()
    assert torch.allclose(lin.bias.grad, ref.bias.grad, rtol=1e-2, atol=1e-2)


# --- Llama block: chunks + residual handles ------------------------------------------------------------

def _llama(seed=0, bias=False, **kw):
    torch.manual_seed(seed)
    m = torch.nn.ModuleDict(dict(
        n1=torch.nn.RMSNorm(D, eps=1e-6), n2=torch.nn.RMSNorm(D, eps=1e-6),
        qkv=NVFP4Linear(D, 3 * D, bias=bias, **kw), o=NVFP4Linear(D, D, bias=bias, **kw),
        gate=NVFP4Linear(D, H, bias=bias, **kw), up=NVFP4Linear(D, H, bias=bias, **kw),
        down=NVFP4Linear(H, D, bias=bias, **kw))).cuda()
    with torch.no_grad():
        m.n1.weight.uniform_(0.5, 1.5)
        m.n2.weight.uniform_(0.5, 1.5)
        for p in m.parameters():
            if p.dim() == 1 and p is not m.n1.weight and p is not m.n2.weight:
                p.normal_(0, 0.1)
    return m


def _attn(q, k, v):
    b, t, _ = q.shape
    q, k, v = (z.view(b, t, HEADS, -1).transpose(1, 2) for z in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, -1)


def _block_v07(m, x):  # the 0.7 fused path: concatenated qkv, separate residual adds
    qkv = rmsnorm_linear(x, m.n1, m.qkv)
    x = x + m.o(_attn(*qkv.split(D, -1)))
    return x + rmsnorm_swiglu_mlp(x, m.n2, m.gate, m.up, m.down)


def _block_v08(m, x, attn=_attn):  # chunks, residual handle, fused residual adds
    q, k, v, res = rmsnorm_linear(x, m.n1, m.qkv, chunks=3, return_residual=True)
    x = m.o(attn(q, k, v), residual=res)
    return rmsnorm_swiglu_mlp(x, m.n2, m.gate, m.up, m.down, residual=True)


def _block_unfused(m, x):
    q, k, v = m.qkv(m.n1(x)).split(D, -1)
    x = x + m.o(_attn(q, k, v))
    n = m.n2(x)
    return x + m.down(F.silu(m.gate(n)) * m.up(n))


def _block_exact(m, x):
    q, k, v = (F.rms_norm(x, (D,), m.n1.weight, 1e-6) @ m.qkv.weight.T + _b(m.qkv)).split(D, -1)
    x = x + _attn(q, k, v) @ m.o.weight.T + _b(m.o)
    n = F.rms_norm(x, (D,), m.n2.weight, 1e-6)
    a = F.silu(n @ m.gate.weight.T + _b(m.gate)) * (n @ m.up.weight.T + _b(m.up))
    return x + a @ m.down.weight.T + _b(m.down)


def _b(lin):
    return 0 if lin.bias is None else lin.bias


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_llama_block_v08_forward_exact_and_gradients(bias, dtype):
    m = _llama(bias=bias)
    mods = list(m.values())
    x = _x(dtype=dtype)

    def run(f):
        def g(t):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return f(m, t)
        return g

    torch.manual_seed(9)
    g8, y8 = _grads(run(_block_v08), mods, x)
    torch.manual_seed(9)
    g7, y7 = _grads(run(_block_v07), mods, x)
    if not bias:  # with biases, 0.7 added the SwiGLU biases unfused (not supported) - compare to exact only
        assert torch.equal(y8, y7)
    torch.manual_seed(9)
    gu, _ = _grads(run(_block_unfused), mods, x)
    ge, _ = _grads(lambda t: _block_exact(m, t.float()), mods, x)
    for a, u, e in zip(g8, gu, ge):
        c8, cu = _cos(a, e), _cos(u, e)
        assert c8 > 0.9 and c8 >= cu - 0.02, (c8, cu)
    assert g8[0].dtype == dtype


def test_chunks_are_views_with_identical_gradients():
    """chunks=3 only changes how the output is handed out: same values, and the q/k/v gradients quantized
    in place give bit-identical gradients to quantizing their concatenation."""
    m = _llama()
    x = _x()
    torch.manual_seed(4)
    q, k, v = rmsnorm_linear(x, m.n1, m.qkv, chunks=3)
    torch.manual_seed(4)
    y = rmsnorm_linear(x, m.n1, m.qkv)
    assert torch.equal(torch.cat([q, k, v], -1), y) and q._base is not None
    gq, gk, gv = (torch.randn_like(q) for _ in range(3))
    a = torch.autograd.grad((q * gq).sum() + (k * gk).sum() + (v * gv).sum(), (x, m.qkv.weight, m.n1.weight))
    b = torch.autograd.grad((y * torch.cat([gq, gk, gv], -1)).sum(), (x, m.qkv.weight, m.n1.weight))
    assert all(torch.equal(u, w) for u, w in zip(a, b))
    # an unused chunk gets a zero gradient
    torch.autograd.grad(rmsnorm_linear(x, m.n1, m.qkv, chunks=3)[0].sum(), x)


def test_residual_handle_gradient_is_summed_in_the_norm_backward():
    m = _llama()
    x = _x()
    torch.manual_seed(4)
    y, res = rmsnorm_linear(x, m.n1, m.qkv, return_residual=True)
    assert torch.equal(res, x) and res.shape == x.shape
    gy, gr = torch.randn_like(y), torch.randn_like(x)
    a = torch.autograd.grad((y * gy).sum() + (res * gr).sum(), x)[0]
    torch.manual_seed(4)
    y2 = rmsnorm_linear(x, m.n1, m.qkv)
    b = torch.autograd.grad((y2 * gy).sum(), x)[0] + gr
    assert (a - b).abs().max() <= 1e-5 * b.abs().max()


@pytest.mark.parametrize("which", ["gelu", "swiglu"])
@pytest.mark.parametrize("norm", ["pre", "none"])
def test_mlp_residual_forward_exact(which, norm):
    torch.manual_seed(0)
    ln = torch.nn.LayerNorm(D).cuda() if which == "gelu" else torch.nn.RMSNorm(D).cuda()
    ln = ln if norm == "pre" else None
    fc1, fc2 = NVFP4Linear(D, H).cuda(), NVFP4Linear(H, D).cuda()
    gate, up, down = NVFP4Linear(D, H).cuda(), NVFP4Linear(D, H).cuda(), NVFP4Linear(H, D).cuda()
    x = _x()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(2)
        if which == "gelu":
            a = layernorm_mlp(x, ln, fc1, fc2, residual=True)
            torch.manual_seed(2)
            b = x + layernorm_mlp(x, ln, fc1, fc2)
        else:
            a = rmsnorm_swiglu_mlp(x, ln, gate, up, down, residual=True)
            torch.manual_seed(2)
            b = x + rmsnorm_swiglu_mlp(x, ln, gate, up, down)
    assert a.dtype == torch.float32 and torch.equal(a, b)


@pytest.mark.parametrize("which", ["gelu", "swiglu"])
def test_norm_free_mlps_match_unfused(which):
    """gelu_mlp / swiglu_mlp (post-norm blocks): forward equals NVFP4Linear + the activation computed in fp32
    inside the quantizer; gradients as good as the unfused path."""
    torch.manual_seed(0)
    if which == "gelu":
        mods = [NVFP4Linear(D, H).cuda(), NVFP4Linear(H, D).cuda()]
        fused = lambda x: gelu_mlp(x, *mods)  # noqa: E731
        unfused = lambda x: mods[1](F.gelu(mods[0](x), approximate="tanh"))  # noqa: E731
        exact = lambda x: F.gelu(x @ mods[0].weight.T + mods[0].bias, approximate="tanh") @ mods[1].weight.T + mods[1].bias  # noqa: E731,E501
    else:
        mods = [NVFP4Linear(D, H).cuda(), NVFP4Linear(D, H).cuda(), NVFP4Linear(H, D).cuda()]
        fused = lambda x: swiglu_mlp(x, *mods)  # noqa: E731
        unfused = lambda x: mods[2](F.silu(mods[0](x)) * mods[1](x))  # noqa: E731

        def exact(x):
            a = F.silu(x @ mods[0].weight.T + mods[0].bias) * (x @ mods[1].weight.T + mods[1].bias)
            return a @ mods[2].weight.T + mods[2].bias
    with torch.no_grad():
        for m in mods:
            m.bias.normal_(0, 0.5)
    x = _x()
    ge, _ = _grads(exact, mods, x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(9)
        gf, yf = _grads(fused, mods, x)
        torch.manual_seed(9)
        gu, yu = _grads(unfused, mods, x)
    assert ((yf.float() - yu.float()).norm() / yu.float().norm()).item() < 0.05
    for a, u, e in zip(gf, gu, ge):
        cf, cu = _cos(a, e), _cos(u, e)
        assert cf > 0.9 and cf >= cu - 0.02, (cf, cu)


def test_swiglu_biases_partial_gradients():
    m = _llama(bias=True)
    x = _x()
    m.gate.weight.requires_grad_(False)
    m.up.bias.requires_grad_(False)
    rmsnorm_swiglu_mlp(x, m.n2, m.gate, m.up, m.down, residual=True).sum().backward()
    assert m.gate.weight.grad is None and m.up.bias.grad is None
    assert all(t is not None for t in (x.grad, m.gate.bias.grad, m.up.weight.grad, m.down.bias.grad))
    # bias-only training of the gate projection still gets its gradient
    for p in m.parameters():
        p.requires_grad_(False)
        p.grad = None
    m.gate.bias.requires_grad_(True)
    rmsnorm_swiglu_mlp(x.detach(), m.n2, m.gate, m.up, m.down).sum().backward()
    assert m.gate.bias.grad is not None and torch.isfinite(m.gate.bias.grad).all()


def _mix(q, k, v):  # deterministic stand-in for attention (flash attention's backward is not deterministic)
    return q * torch.sigmoid(k) + v


def test_fused_accumulation_cache_and_determinism():
    base = _llama(bias=True)
    variants = [copy.deepcopy(base) for _ in range(2)]
    for lin in ("qkv", "o", "gate", "up", "down"):
        variants[1][lin].fuse_wgrad_accumulation = True
        variants[1][lin].cache_weight_quantization = True
    x = _x()
    grads = []
    for m in variants:
        torch.manual_seed(11)
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _block_v08(m, x, _mix).float().square().mean().backward()
        grads.append([p.grad.clone() for p in m.parameters()])
    assert all(torch.equal(a, b) for a, b in zip(*grads))


def test_compile_fullgraph_v08_block():
    m = _llama(bias=True)

    def f(t):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, k, v, res = rmsnorm_linear(t, m.n1, m.qkv, chunks=3, return_residual=True)
            h = m.o(q, residual=res)  # no attention (not what is tested); k, v enter below
            out = rmsnorm_swiglu_mlp(h, m.n2, m.gate, m.up, m.down, residual=True).float()
            return out + (k.float() * v.float()).mean()

    torch._dynamo.reset()
    cf = torch.compile(f, fullgraph=True, backend=COMPILE_BACKEND)
    x = _x()
    y = cf(x)
    y.square().mean().backward()
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all() and torch.isfinite(m.gate.bias.grad).all()
    ye = f(x.detach())
    assert ((y - ye).norm() / ye.norm()).item() < 1e-2


def test_gpt_block_v08():
    """GPT-2 style: LayerNorm qkv with biases in 3 chunks, residual handles, MLP with fused residual."""
    torch.manual_seed(0)
    ln1, ln2 = torch.nn.LayerNorm(D).cuda(), torch.nn.LayerNorm(D).cuda()
    qkv, o = NVFP4Linear(D, 3 * D).cuda(), NVFP4Linear(D, D).cuda()
    fc1, fc2 = NVFP4Linear(D, 4 * D).cuda(), NVFP4Linear(4 * D, D).cuda()
    mods = [ln1, ln2, qkv, o, fc1, fc2]

    def v08(x):
        q, k, v, res = layernorm_linear(x, ln1, qkv, chunks=3, return_residual=True)
        x = o(_attn(q, k, v), residual=res)
        return layernorm_mlp(x, ln2, fc1, fc2, residual=True)

    def v07(x):
        q, k, v = layernorm_linear(x, ln1, qkv).split(D, -1)
        x = x + o(_attn(q, k, v))
        return x + layernorm_mlp(x, ln2, fc1, fc2)

    def exact(x):
        q, k, v = F.linear(F.layer_norm(x, (D,), ln1.weight, ln1.bias), qkv.weight, qkv.bias).split(D, -1)
        x = x + F.linear(_attn(q, k, v), o.weight, o.bias)
        h = F.gelu(F.linear(F.layer_norm(x, (D,), ln2.weight, ln2.bias), fc1.weight, fc1.bias), approximate="tanh")
        return x + F.linear(h, fc2.weight, fc2.bias)

    x = _x()
    ge, _ = _grads(exact, mods, x)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch.manual_seed(9)
        g8, y8 = _grads(v08, mods, x)
        torch.manual_seed(9)
        g7, y7 = _grads(v07, mods, x)
    assert torch.equal(y8, y7)
    for a, b, e in zip(g8, g7, ge):
        c8, c7 = _cos(a, e), _cos(b, e)
        assert c8 > 0.9 and c8 >= c7 - 0.02, (c8, c7)
