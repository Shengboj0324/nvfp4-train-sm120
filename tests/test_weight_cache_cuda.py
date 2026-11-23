"""Weight-quantization cache (NVFP4Linear(cache_weight_quantization=True)): results bit-identical to
quantizing every call, reuse across the micro-batches of an optimizer step, and every way a weight can
change is noticed (or, for in-place writes through ``.data``, needs the documented invalidation)."""
import copy
import io

import pytest
import torch

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4 import invalidate_weight_caches
from nvfp4.fused import layernorm_linear, layernorm_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
DEV = "cuda"
D = 256


class Block(torch.nn.Module):
    def __init__(self, cache: bool, fused: bool):
        super().__init__()
        kw = dict(bias=False, cache_weight_quantization=cache, fuse_wgrad_accumulation=True)
        self.ln1, self.ln2 = torch.nn.LayerNorm(D), torch.nn.LayerNorm(D)
        self.qkv = NVFP4Linear(D, D, **kw)
        self.fc1, self.fc2 = NVFP4Linear(D, 4 * D, **kw), NVFP4Linear(4 * D, D, **kw)
        self.fused = fused

    def forward(self, x):
        if self.fused:
            x = x + layernorm_linear(x, self.ln1, self.qkv)
            return x + layernorm_mlp(x, self.ln2, self.fc1, self.fc2)
        x = x + self.qkv(self.ln1(x))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


def _make(cache, fused):
    torch.manual_seed(0)
    return Block(cache, fused).to(DEV)


def _train(m, steps=2, micro=3, opt_cls=torch.optim.AdamW, **opt_kw):
    opt = opt_cls(m.parameters(), lr=1e-2, **opt_kw)
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        for mb in range(micro):
            g = torch.Generator(device=DEV).manual_seed(10 * s + mb)
            x = torch.randn(2, 128, D, device=DEV, generator=g)
            torch.manual_seed(100 * s + mb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m(x).float().square().mean().backward()
        opt.step()
    return [p.detach().clone() for p in m.parameters()]


def _stats(m):
    caches = [lin.weight_quant_cache for lin in (m.qkv, m.fc1, m.fc2)]
    return sum(c.hits for c in caches), sum(c.misses for c in caches)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("opt", ["adamw_fused", "adamw_foreach", "sgd"])
def test_bit_identical_and_reused_within_a_step(fused, opt):
    opt_cls, kw = {"adamw_fused": (torch.optim.AdamW, {"fused": True}),
                   "adamw_foreach": (torch.optim.AdamW, {"foreach": True}),
                   "sgd": (torch.optim.SGD, {"momentum": 0.9})}[opt]
    plain = _train(_make(False, fused), opt_cls=opt_cls, **kw)
    m = _make(True, fused)
    cached = _train(m, opt_cls=opt_cls, **kw)
    assert all(torch.equal(a, b) for a, b in zip(plain, cached))
    # 3 layers x 2 steps: the first micro-batch of a step re-quantizes (the optimizer changed the weights),
    # the other two reuse
    assert _stats(m) == (3 * 2 * 2, 3 * 2), _stats(m)


def test_every_parameter_update_is_noticed():
    lin = NVFP4Linear(D, D, bias=False, cache_weight_quantization=True).to(DEV)
    c = lin.weight_quant_cache
    x = torch.randn(64, D, device=DEV)

    def run():
        return lin(x)

    ref = run()
    assert torch.equal(run(), ref) and (c.hits, c.misses) == (1, 1)
    with torch.no_grad():
        lin.weight.mul_(2)                         # in place through the parameter
    y = run()
    assert c.misses == 2 and not torch.equal(y, ref)
    lin.load_state_dict({"weight": torch.randn(D, D, device=DEV)})
    run()
    assert c.misses == 3
    lin.weight.data = torch.randn(D, D, device=DEV)  # new storage
    run()
    assert c.misses == 4
    lin.to(torch.bfloat16)
    lin(x.bfloat16())
    assert c.misses == 5


def test_in_place_data_writes_need_invalidation():
    """The documented limit: an in-place write through ``.data`` does not bump the version counter."""
    lin = NVFP4Linear(D, D, bias=False, cache_weight_quantization=True).to(DEV)
    x = torch.randn(64, D, device=DEV)
    lin(x)
    lin.weight.data.mul_(-1)
    stale = lin(x)
    assert lin.weight_quant_cache.hits == 1  # not noticed
    assert invalidate_weight_caches(torch.nn.Sequential(lin)) == 1
    fresh = lin(x)
    assert lin.weight_quant_cache.misses == 2
    assert torch.equal(fresh, -stale)  # exact: negation commutes with the symmetric quantization


def test_inference_then_training_adds_the_transpose():
    lin = NVFP4Linear(D, D, bias=False, cache_weight_quantization=True).to(DEV)
    x = torch.randn(64, D, device=DEV)
    with torch.no_grad():
        lin(x)                                     # W only
    assert lin.weight_quant_cache.wtq is None
    x.requires_grad_()
    lin(x).sum().backward()                        # needs W^T for dgrad: re-quantized
    assert lin.weight_quant_cache.misses == 2 and lin.weight_quant_cache.wtq is not None
    plain = NVFP4Linear(D, D, bias=False).to(DEV)
    plain.load_state_dict(lin.state_dict())
    x2 = x.detach().clone().requires_grad_()
    plain(x2).sum().backward()
    assert torch.equal(x.grad, x2.grad)


def test_cuda_graph_capture_bypasses_the_cache():
    """In a captured training step the weights change on every replay with no Python running, so the
    cache must not be used inside the capture; results equal the uncached graph's bit for bit."""
    def build(cache):
        torch.manual_seed(0)
        m = torch.nn.Sequential(NVFP4Linear(D, 2 * D, bias=False, cache_weight_quantization=cache),
                                NVFP4Linear(2 * D, D, bias=False, cache_weight_quantization=cache)).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, capturable=True)
        x = torch.randn(128, D, device=DEV)

        def step():
            opt.zero_grad(set_to_none=True)
            loss = m(x).float().square().mean()
            loss.backward()
            opt.step()
            return loss.detach()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(5)
        with torch.cuda.graph(g):
            loss = step()
        losses = []
        for _ in range(10):
            g.replay()
            losses.append(loss.clone())
        return torch.stack(losses), [p.detach().clone() for p in m.parameters()], m

    l0, p0, _ = build(False)
    l1, p1, m1 = build(True)
    assert torch.equal(l0, l1) and all(torch.equal(a, b) for a, b in zip(p0, p1))
    assert l1[-1] < l1[0]


def test_compile_and_checkpointing_with_cache_enabled():
    from torch.utils.checkpoint import checkpoint
    m = _make(True, True)
    x = torch.randn(2, 128, D, device=DEV, requires_grad=True)
    torch.manual_seed(1)
    ref = m(x)
    ref.float().sum().backward()
    g_ref = [p.grad.clone() for p in (m.qkv.weight, m.fc1.weight, m.fc2.weight)]
    for p in m.parameters():
        p.grad = None
    torch.manual_seed(1)
    y = checkpoint(m, x, use_reentrant=False)
    assert torch.equal(y, ref)
    y.float().sum().backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(g_ref, (m.qkv.weight, m.fc1.weight, m.fc2.weight)))
    torch._dynamo.reset()
    cm = torch.compile(_make(True, False), fullgraph=True, backend=COMPILE_BACKEND)
    cm(torch.randn(2, 128, D, device=DEV)).float().sum().backward()


def test_copies_and_pickles_start_empty():
    lin = NVFP4Linear(D, D, bias=False, cache_weight_quantization=True).to(DEV)
    lin(torch.randn(64, D, device=DEV))
    assert lin.weight_quant_cache.wq is not None
    assert copy.deepcopy(lin).weight_quant_cache.wq is None
    buf = io.BytesIO()
    torch.save(lin, buf)
    assert buf.tell() < 2 * D * D * 4  # the weight, not the cached copies
    buf.seek(0)
    again = torch.load(buf, weights_only=False)
    assert again.weight_quant_cache.wq is None and again.cache_weight_quantization
    assert "cache_weight_quantization=True" in repr(lin)
