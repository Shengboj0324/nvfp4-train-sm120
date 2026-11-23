"""Production-facing behaviour: non-finite inputs never turn into finite outputs, clear errors on
misuse, and the integrations training code relies on (checkpointing, state dicts, torch.compile, CUDA
graphs, side streams, DDP) work."""
import io
import os

import pytest
import torch
import torch.nn.functional as F

from conftest import COMPILE_BACKEND, requires_sm120
from nvfp4 import ops
from nvfp4.fused import layernorm_linear, layernorm_mlp
from nvfp4.linear import NVFP4Linear

pytestmark = requires_sm120
DEV = "cuda"


# --- non-finite inputs -----------------------------------------------------------------------------
# One bad element anywhere makes the per-tensor decode scale non-finite, so every output that tensor
# feeds is non-finite: a diverging run shows NaN/inf in its loss instead of training on garbage.

def _poison(x, bad, idx=(3, 7)):
    x = x.clone()
    x[idx] = float(bad)
    return x


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_quantize_and_gemm_propagate_non_finite(bad):
    a, b = torch.randn(256, 512, device=DEV), torch.randn(128, 512, device=DEV)
    qa = ops.quantize(_poison(a, bad))
    assert not torch.isfinite(qa.gdec).all()
    out = ops.gemm(qa, ops.quantize(b), torch.float32)
    assert not torch.isfinite(out).any()
    # the clean operand is unaffected
    assert torch.isfinite(ops.quantize(b).gdec).all()


@pytest.mark.parametrize("bad", ["nan", "inf"])
@pytest.mark.parametrize("kw", [dict(), dict(rht_seed=1, col_pad=128), dict(block_2d=True),
                                dict(row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)])
def test_quantize_dual_propagates_non_finite(bad, kw):
    x = _poison(torch.randn(256, 512, device=DEV), bad).bfloat16()
    r, c = ops.quantize_dual(x, **kw)
    assert not torch.isfinite(r.gdec).all() and not torch.isfinite(c.gdec).all()


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_fused_producers_propagate_non_finite(bad):
    x = _poison(torch.randn(256, 512, device=DEV), bad)
    w, b = torch.ones(512, device=DEV), torch.zeros(512, device=DEV)
    _, _, r, c = ops.layernorm_quantize_dual(x, w, b, 1e-5, rht_seed=1, col_pad=128)
    assert not torch.isfinite(r.gdec).all() and not torch.isfinite(c.gdec).all()
    h = x.bfloat16()
    r, c = ops.gelu_quantize_dual(h, rht_seed=1, col_pad=128)
    assert not torch.isfinite(r.gdec).all()
    r, c = ops.gelu_backward_quantize_dual(torch.randn_like(h), h, rht_seed=1, col_pad=128)
    assert not torch.isfinite(r.gdec).all()


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("where", ["g", "u", "dy"])
def test_llama_producers_propagate_non_finite(bad, where):
    t = {k: torch.randn(256, 512, device=DEV) for k in ("g", "u", "dy")}
    t[where] = _poison(t[where], bad)
    g, u, dy = (t[k].bfloat16() for k in ("g", "u", "dy"))
    if where != "dy":
        r, c = ops.swiglu_quantize_dual(g, u, rht_seed=1, col_pad=128)
        assert not torch.isfinite(r.gdec).all() and not torch.isfinite(c.gdec).all()
    r, _ = ops.swiglu_backward_quantize_dual(dy, g, u, gate=True, col=False)
    assert not torch.isfinite(r.gdec).all()
    if where != "u":  # the up-gradient dy * silu(g) does not read u
        r, _ = ops.swiglu_backward_quantize_dual(dy, g, u, gate=False, col=False)
        assert not torch.isfinite(r.gdec).all()
    rstd, r, c = ops.rmsnorm_quantize_dual(_poison(torch.randn(256, 512, device=DEV), bad),
                                           torch.ones(512, device=DEV), 1e-6, rht_seed=1, col_pad=128)
    assert not torch.isfinite(r.gdec).all() and not torch.isfinite(c.gdec).all()


@pytest.mark.parametrize("bad", ["nan", "inf"])
def test_layers_propagate_non_finite(bad):
    lin = NVFP4Linear(512, 256, bias=False).to(DEV)
    x = _poison(torch.randn(128, 512, device=DEV), bad).requires_grad_()
    y = lin(x)
    assert not torch.isfinite(y).all()
    y.float().sum().backward()
    assert not torch.isfinite(lin.weight.grad).all()

    ln = torch.nn.LayerNorm(512).to(DEV)
    fc1, fc2 = NVFP4Linear(512, 1024, bias=False).to(DEV), NVFP4Linear(1024, 512, bias=False).to(DEV)
    assert not torch.isfinite(layernorm_mlp(x.detach(), ln, fc1, fc2)).all()
    from nvfp4.fused import rmsnorm_swiglu_mlp
    norm = torch.nn.RMSNorm(512).to(DEV)
    up = NVFP4Linear(512, 1024, bias=False).to(DEV)
    assert not torch.isfinite(rmsnorm_swiglu_mlp(x.detach(), norm, fc1, up, fc2)).all()


def test_non_finite_gradient_propagates():
    lin = NVFP4Linear(512, 256, bias=False).to(DEV)
    x = torch.randn(128, 512, device=DEV, requires_grad=True)
    y = lin(x)
    g = torch.ones_like(y)
    g[0, 0] = float("nan")
    y.backward(g)
    assert not torch.isfinite(x.grad).all() and not torch.isfinite(lin.weight.grad).all()


# --- misuse gives clear errors -----------------------------------------------------------------------

def test_clear_errors():
    with pytest.raises(RuntimeError, match="CUDA tensors"):
        ops.quantize(torch.randn(16, 16))
    with pytest.raises(RuntimeError, match="bf16 or fp32"):
        ops.quantize(torch.randn(16, 16, device=DEV, dtype=torch.float16))
    with pytest.raises(RuntimeError, match="same K"):
        ops.gemm(ops.quantize(torch.randn(16, 512, device=DEV)), ops.quantize(torch.randn(16, 256, device=DEV)))
    with pytest.raises(ValueError, match="divisible by 128"):
        NVFP4Linear(100, 128)
    with pytest.raises(RuntimeError, match="multiple of 16"):
        ops.quantize_dual(torch.randn(32, 40, device=DEV))


def test_available_is_exact_arch():
    assert ops.available()
    assert torch.cuda.get_device_capability() == ops.SUPPORTED_CAPABILITY


# --- shapes and dtypes the layer accepts ---------------------------------------------------------------

@pytest.mark.parametrize("shape", [(0, 512), (1, 512), (3, 259, 512), (2, 5, 7, 512)])
def test_any_leading_shape(shape):
    lin = NVFP4Linear(512, 256).to(DEV)
    x = torch.randn(shape, device=DEV, requires_grad=True)
    y = lin(x)
    assert y.shape == (*shape[:-1], 256)
    y.float().sum().backward()
    assert x.grad.shape == x.shape and torch.isfinite(lin.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_input_dtypes_and_noncontiguous(dtype):
    lin = NVFP4Linear(512, 256).to(DEV)
    x = torch.randn(512, 64, device=DEV, dtype=dtype).T.requires_grad_()  # non-contiguous
    y = lin(x)
    assert y.dtype == dtype
    y.float().sum().backward()
    assert x.grad.dtype == dtype


def test_bf16_weights_and_bias():
    lin = NVFP4Linear(512, 256, bias=True).to(DEV, torch.bfloat16)
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    lin(x).float().sum().backward()
    assert lin.weight.grad.dtype == torch.bfloat16 and lin.bias.grad is not None


# --- integrations ----------------------------------------------------------------------------------------

def test_state_dict_interchangeable_with_nn_linear():
    lin = NVFP4Linear(512, 256).to(DEV)
    buf = io.BytesIO()
    torch.save(lin.state_dict(), buf)
    buf.seek(0)
    lin2 = NVFP4Linear(512, 256).to(DEV)
    lin2.load_state_dict(torch.load(buf))
    x = torch.randn(32, 512, device=DEV)
    assert torch.equal(lin(x), lin2(x))
    plain = torch.nn.Linear(512, 256).to(DEV)
    plain.load_state_dict(lin.state_dict())
    assert torch.equal(NVFP4Linear.from_linear(plain)(x), lin(x))


def test_activation_checkpointing_is_exact():
    from torch.utils.checkpoint import checkpoint
    ln = torch.nn.LayerNorm(512).to(DEV)
    fc1, fc2 = NVFP4Linear(512, 1024, bias=False).to(DEV), NVFP4Linear(1024, 512, bias=False).to(DEV)
    lin = NVFP4Linear(512, 512, bias=False).to(DEV)
    x = torch.randn(256, 512, device=DEV, requires_grad=True)

    def f(t):
        return layernorm_mlp(t, ln, fc1, fc2) + lin(t)

    torch.manual_seed(1)
    f(x).float().sum().backward()
    ref = [p.grad.clone() for p in (fc1.weight, fc2.weight, lin.weight)] + [x.grad.clone()]
    for p in (fc1.weight, fc2.weight, lin.weight, x):
        p.grad = None
    torch.manual_seed(1)
    checkpoint(f, x, use_reentrant=False).float().sum().backward()
    got = [p.grad for p in (fc1.weight, fc2.weight, lin.weight)] + [x.grad]
    assert all(torch.equal(a, b) for a, b in zip(ref, got))


def test_torch_compile_matches_eager():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.LayerNorm(512), NVFP4Linear(512, 1024, bias=False), torch.nn.GELU(),
                            NVFP4Linear(1024, 512, bias=False)).to(DEV)
    x = torch.randn(256, 512, device=DEV)
    with torch.no_grad():
        eager = m(x)
        compiled = torch.compile(m, backend=COMPILE_BACKEND)(x)
    assert torch.equal(eager, compiled)
    xg = x.clone().requires_grad_()
    torch.compile(m, backend=COMPILE_BACKEND)(xg).float().sum().backward()
    assert torch.isfinite(xg.grad).all()


def test_cuda_graph_inference_replay():
    lin = NVFP4Linear(512, 256, bias=False).to(DEV)
    x = torch.randn(64, 512, device=DEV)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(s):
        lin(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        y = lin(x)
    x.copy_(torch.randn_like(x))
    g.replay()
    with torch.no_grad():
        assert torch.equal(y, lin(x))


def test_side_stream_and_split_k_on_side_stream():
    qa = ops.quantize(torch.randn(768, 8192, device=DEV))
    qb = ops.quantize(torch.randn(768, 8192, device=DEV))
    ref = ops.gemm(qa, qb, torch.float32)  # auto split-K
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    s1.wait_stream(torch.cuda.current_stream())  # the operands were produced on the default stream
    s2.wait_stream(torch.cuda.current_stream())
    outs = []
    for _ in range(4):  # split-K workspaces and counters are per call, so concurrent streams don't collide
        with torch.cuda.stream(s1):
            outs.append(ops.gemm(qa, qb, torch.float32))
        with torch.cuda.stream(s2):
            outs.append(ops.gemm(qa, qb, torch.float32))
    torch.cuda.synchronize()
    assert all(torch.equal(o, ref) for o in outs)


def test_ddp_single_process():
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_nccl_available():
        pytest.skip("NCCL not available")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29541")
    dist.init_process_group("nccl", rank=0, world_size=1)
    try:
        lin = NVFP4Linear(512, 256, bias=False).to(DEV)
        ref = NVFP4Linear(512, 256, bias=False).to(DEV)
        ref.load_state_dict(lin.state_dict())
        m = torch.nn.parallel.DistributedDataParallel(lin)
        x = torch.randn(64, 512, device=DEV)
        torch.manual_seed(4)
        m(x).float().sum().backward()
        torch.manual_seed(4)
        ref(x).float().sum().backward()
        assert torch.equal(lin.weight.grad, ref.weight.grad)  # world size 1: allreduce is the identity
    finally:
        dist.destroy_process_group()


def test_fused_and_unfused_modules_share_state():
    # The fused path reads the same parameters as the modules, so checkpoints work either way.
    ln = torch.nn.LayerNorm(512).to(DEV)
    lin = NVFP4Linear(512, 512, bias=False).to(DEV)
    x = torch.randn(64, 512, device=DEV)
    sd = {"ln": ln.state_dict(), "lin": lin.state_dict()}
    ln2, lin2 = torch.nn.LayerNorm(512).to(DEV), NVFP4Linear(512, 512, bias=False).to(DEV)
    ln2.load_state_dict(sd["ln"])
    lin2.load_state_dict(sd["lin"])
    with torch.no_grad():
        assert torch.equal(layernorm_linear(x, ln, lin), layernorm_linear(x, ln2, lin2))
        y_unfused = lin(F.layer_norm(x, (512,), ln.weight, ln.bias, ln.eps))
        y_fused = layernorm_linear(x, ln, lin)
    rel = ((y_fused - y_unfused).norm() / y_unfused.norm()).item()
    assert rel < 0.15  # both are 4-bit approximations of the same product


def test_forced_ptx_jit_gives_a_clear_error(monkeypatch):
    """CUDA_FORCE_PTX_JIT=1 makes every launch fail with 'no kernel image'; explain it instead."""
    x = torch.randn(64, 64, device=DEV)
    monkeypatch.setenv("CUDA_FORCE_PTX_JIT", "1")
    assert not ops.available()
    with pytest.raises(RuntimeError, match="CUDA_FORCE_PTX_JIT"):
        ops.quantize(x)
    monkeypatch.setenv("CUDA_FORCE_PTX_JIT", "0")
    assert ops.available()
    ops.quantize(x)
