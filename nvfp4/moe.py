"""Top-k mixture-of-experts MLP whose experts train in NVFP4.

    from nvfp4.moe import MoE
    moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()
    nvfp4.convert_linears(moe)               # experts -> NVFP4Linear (the router stays BF16: too few outputs)
    y = moe(x)                               # x [..., d]; moe.aux_loss: Switch load-balancing loss

Routing: softmax over the router logits (fp32), top-k experts per token, weights renormalized over the k
choices. The (token, expert) pairs are sorted by expert (stable), the tokens gathered into one contiguous
buffer, and each expert runs on its slice: with NVFP4 experts that is the fused NVFP4 MLP (gelu_mlp /
swiglu_mlp: the activation inside the second GEMM's quantizer, NVFP4 fprop / dgrad / wgrad, fused wgrad
accumulation), so every expert's operands get their own tensor scales and the backward is the dense layers'
tested path. The outputs, scaled by the routing weights, are summed back per token (index_add).

The per-expert token counts are read on the host once per call (one synchronization, as in MegaBlocks'
dropless MoE); experts that receive no tokens are skipped. No tokens are dropped (no capacity factor).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4.linear import NVFP4Linear


class _Expert(nn.Module):
    def __init__(self, d: int, d_ff: int, activation: str, bias: bool, device=None, dtype=None):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.gate = nn.Linear(d, d_ff, bias=bias, device=device, dtype=dtype)
            self.up = nn.Linear(d, d_ff, bias=bias, device=device, dtype=dtype)
            self.down = nn.Linear(d_ff, d, bias=bias, device=device, dtype=dtype)
        else:
            self.fc1 = nn.Linear(d, d_ff, bias=bias, device=device, dtype=dtype)
            self.fc2 = nn.Linear(d_ff, d, bias=bias, device=device, dtype=dtype)

    def _nvfp4(self) -> bool:
        mods = (self.gate, self.up, self.down) if self.activation == "swiglu" else (self.fc1, self.fc2)
        return all(isinstance(m, NVFP4Linear) and m.backend == "cuda" for m in mods)

    def forward(self, x):
        from nvfp4.fused import gelu_mlp, swiglu_mlp
        if self.activation == "swiglu":
            if self._nvfp4():
                return swiglu_mlp(x, self.gate, self.up, self.down)
            return self.down(F.silu(self.gate(x)) * self.up(x))
        approx = "tanh" if self.activation == "gelu_tanh" else "none"
        if self._nvfp4():
            return gelu_mlp(x, self.fc1, self.fc2, approximate=approx)
        return self.fc2(F.gelu(self.fc1(x), approximate=approx))


class MoE(nn.Module):
    """Dropless top-k mixture-of-experts MLP (see the module docstring).

    d, d_ff: model and expert hidden sizes (multiples of 128 for NVFP4 experts); activation: "swiglu",
    "gelu_tanh" or "gelu". After forward, ``aux_loss`` holds the Switch Transformer load-balancing loss
    E * sum_e f_e P_e (f_e: fraction of routed slots, P_e: mean router probability of expert e), 1.0 when
    balanced; add ``coef * moe.aux_loss`` to the training loss."""

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int = 2, activation: str = "swiglu",
                 bias: bool = False, normalize_weights: bool = True, device=None, dtype=None):
        super().__init__()
        if activation not in ("swiglu", "gelu_tanh", "gelu"):
            raise ValueError(f"activation must be swiglu, gelu_tanh or gelu, got {activation!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")
        self.d, self.n_experts, self.top_k = d, n_experts, top_k
        self.normalize_weights = normalize_weights
        self.router = nn.Linear(d, n_experts, bias=False, device=device, dtype=dtype)
        self.experts = nn.ModuleList(_Expert(d, d_ff, activation, bias, device, dtype) for _ in range(n_experts))
        self.aux_loss: torch.Tensor | None = None
        self.last_counts: list[int] | None = None

    def route(self, x2d: torch.Tensor):
        """(weights [T, k] fp32, experts [T, k] int64, router probabilities [T, E] fp32)."""
        probs = torch.softmax(self.router(x2d).float(), dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.normalize_weights and self.top_k > 1:
            w = w / w.sum(-1, keepdim=True)
        return w, idx, probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2d = x.reshape(-1, self.d)
        T, k, E = x2d.shape[0], self.top_k, self.n_experts
        w, idx, probs = self.route(x2d)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)  # (token, choice) slots grouped by expert
        token = order // k
        counts = torch.bincount(flat, minlength=E)
        host = counts.tolist()  # one host sync: the slice bounds
        self.last_counts = host
        xs = x2d.index_select(0, token)
        outs, off = [], 0
        for e, n in enumerate(host):
            if n:
                outs.append(self.experts[e](xs[off:off + n]))
            off += n
        ys = torch.cat(outs, 0)
        ys = ys * w.reshape(-1).index_select(0, order).unsqueeze(-1).to(ys.dtype)
        y = torch.zeros(T, ys.shape[-1], device=x.device, dtype=ys.dtype).index_add(0, token, ys)
        frac = counts.float() / max(1, T * k)
        self.aux_loss = E * (frac * probs.mean(0)).sum()
        return y.reshape(*lead, ys.shape[-1])


__all__ = ["MoE"]
