"""nvfp4.moe.MoE: routing against a per-token reference loop (BF16 experts), NVFP4 experts against BF16 ones
(NVFP4 error level, forward and gradients), experts without tokens, the load-balancing loss, and that NVFP4
experts run the fused MLPs."""
import copy

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_sm120
from nvfp4 import convert_linears
from nvfp4.linear import NVFP4Linear
from nvfp4.moe import MoE

pytestmark = requires_sm120


def _cos(a, b):
    return F.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0).item()


def _moe(activation="swiglu", E=4, k=2, d=256, d_ff=512, seed=0):
    torch.manual_seed(seed)
    return MoE(d, d_ff, E, top_k=k, activation=activation).cuda()


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh", "gelu"])
@pytest.mark.parametrize("k", [1, 2])
def test_routing_matches_per_token_loop(activation, k):
    moe = _moe(activation, k=k)
    x = torch.randn(3, 50, 256, device="cuda")
    y = moe(x)
    x2 = x.reshape(-1, 256)
    w, idx, _ = moe.route(x2)
    want = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        for j in range(k):
            want[t] += w[t, j] * moe.experts[int(idx[t, j])](x2[t:t + 1])[0]
    torch.testing.assert_close(y.reshape(-1, 256), want, rtol=1e-4, atol=1e-5)
    assert sum(moe.last_counts) == x2.shape[0] * k


@pytest.mark.parametrize("activation", ["swiglu", "gelu_tanh"])
def test_nvfp4_experts_track_bf16(activation):
    ref = _moe(activation, E=4, k=2, d=512, d_ff=1024)
    moe = copy.deepcopy(ref)
    names = convert_linears(moe)
    assert len(names) == 4 * (3 if activation == "swiglu" else 2)  # every expert linear, not the router
    assert not isinstance(moe.router, NVFP4Linear)
    x = torch.randn(4, 256, 512, device="cuda", requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = moe(x)
        y_ref = ref(x2)
    assert _cos(y, y_ref) > 0.96  # two NVFP4 GEMMs in series (one NVFP4Linear: > 0.98, test_linear.py)
    torch.manual_seed(1)
    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert _cos(x.grad, x2.grad) > 0.95
    for (n, p), (_, q) in zip(moe.named_parameters(), ref.named_parameters()):
        if q.grad is not None and q.grad.abs().sum() > 0:
            assert p.grad is not None and _cos(p.grad, q.grad) > 0.9, n


def test_experts_without_tokens():
    moe = _moe(E=4, k=1)
    convert_linears(moe)
    with torch.no_grad():  # route everything to expert 2
        moe.router.weight.zero_()
        moe.router.weight[2] = 1.0
    x = torch.rand(64, 256, device="cuda") + 0.1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = moe(x)
    y.float().sum().backward()
    assert moe.last_counts == [0, 0, 64, 0]
    assert moe.experts[2].gate.weight.grad is not None
    assert moe.experts[0].gate.weight.grad is None
    assert moe.aux_loss.item() == pytest.approx(4 * 1.0 * moe.route(x)[2].mean(0)[2].item(), rel=1e-5)


def test_balanced_router_aux_loss_is_one():
    moe = _moe(E=4, k=2)
    with torch.no_grad():
        moe.router.weight.zero_()  # uniform probabilities; top-k ties broken by index -> experts 0, 1
    moe(torch.randn(32, 256, device="cuda"))
    # f = (0.5, 0.5, 0, 0), P = 1/4 each: E * sum f P = 4 * (0.5 * 0.25 * 2) = 1
    assert moe.aux_loss.item() == pytest.approx(1.0)


def test_validation():
    with pytest.raises(ValueError):
        MoE(256, 512, 4, top_k=5)
    with pytest.raises(ValueError):
        MoE(256, 512, 4, activation="relu")
