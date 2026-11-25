"""Top-k mixture-of-experts MLP whose experts train in NVFP4 with grouped GEMMs.

    from nvfp4.moe import MoE
    moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()
    nvfp4.convert_linears(moe)               # experts -> NVFP4 (moe.to_nvfp4(...)); the router stays BF16
    y = moe(x)                               # x [..., d]; moe.aux_loss: Switch load-balancing loss

Routing: softmax over the router logits (fp32), top-k experts per token, weights renormalized over the k
choices. The (token, expert) pairs are sorted by expert (stable). No tokens are dropped (no capacity factor).
The NVFP4 path never reads the expert sizes on the host: its buffers have the worst-case size for the
number of tokens, and the layout is computed on the GPU, so a training step with NVFP4 MoE layers can be
captured in a CUDA graph and traced by ``torch.compile`` without graph breaks. (The BF16 loop reads the
sizes once per call, as MegaBlocks does.)

Expert weights are stacked parameters, ``w_gate`` / ``w_up`` [E, d_ff, d] and ``w_down`` [E, d, d_ff]
(SwiGLU) or ``w_fc1`` [E, d_ff, d] and ``w_fc2`` [E, d, d_ff] (GELU), so all experts run in one GEMM launch
per projection:

  * the routed tokens are gathered into one buffer, each expert's rows padded to a multiple of 128 (zero
    rows; T k + E 127 rows in all, the unused tail tiles skipped by the GEMMs), and quantized once (one
    tensor scale, like a dense layer's activation);
  * forward and input-gradient GEMMs are grouped "ragged M" GEMMs (``ops.gemm_grouped(mode="m")``): every
    128-row tile reads its expert's weight rows; the stacked weights are quantized as one [E N, K] tensor
    (16x16 blocks, so the transposed copy is the same quantization, rearranged per expert);
  * the weight gradients are one grouped "ragged K" GEMM (``mode="k"``): each expert's output tiles reduce
    over its own token range of the column-quantized operands (RHT along the tokens, within each expert's
    128-aligned range), accumulated into ``.grad`` in place with ``fuse_wgrad_accumulation``;
  * activation (SwiGLU / GELU) and its backward are fused into the quantizers, as in nvfp4.fused.

Per step that is a fixed number of kernels, independent of the number of experts. In BF16 (before
``to_nvfp4``, or under ``ops.set_high_precision``) the experts run one after another with torch matmuls.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4 import ops
from nvfp4.linear import ALIGN, WeightQuantCache, _accumulation_target, dgrad_weight_operand, draw_seeds

BF16 = torch.bfloat16
_ROWS = 128  # every expert's rows start on a GEMM tile


def _stack(w: torch.Tensor, need_t: bool, cache, row_seed):
    """(W [E N, K], the dgrad operand W^T rearranged to [E K, N] or None) of a stacked [E, N, K] weight
    (``row_seed``: the row SR seed of the gradient it multiplies, for MS-EDEN on dgrad)."""
    wq, wtq = dgrad_weight_operand("cuda", w, need_t, cache, row_seed if need_t else None)
    return wq, (ops.stack_transposed(wtq, w.shape[0]) if need_t else None)


def _wgrad(ctx, param, a: ops.NVFP4Tensor, b: ops.NVFP4Tensor, segs: torch.Tensor):
    """Weight gradients of all experts, [E, R, C] = per expert a^T b over its tokens (ragged-K grouped GEMM);
    added into an existing fp32 ``param.grad`` in place when fused accumulation applies (see linear._weight_grad;
    experts that got no tokens keep their gradient)."""
    E, R, C = param.shape
    g = _accumulation_target(ctx, param) if ctx.fuse else None
    if g is not None:
        ops.gemm_grouped(a, b, "k", segs, out_dtype=torch.float32, out=g.view(E * R, C), accumulate=True)
        param.grad = None
        return g
    out = torch.zeros(E * R, C, device=a.data.device, dtype=torch.float32)
    ops.gemm_grouped(a, b, "k", segs, out_dtype=torch.float32, out=out, accumulate=True)
    return out.view(E, R, C).to(param.dtype)


class _Dispatch(torch.autograd.Function):
    """rows[i] = x[src[i]] (src[i] == T: a zero row), with a deterministic backward: each token's gradient is
    the sum of its k slots' rows, gathered (``pos`` [T, k]: the rows of token t's slots), not scatter-added
    with atomics in a varying order."""

    @staticmethod
    def forward(ctx, x, src, pos):
        ctx.save_for_backward(pos)
        return torch.cat([x, x.new_zeros(1, x.shape[1])]).index_select(0, src)

    @staticmethod
    def backward(ctx, g):
        (pos,) = ctx.saved_tensors
        T, k = pos.shape
        dx = g.index_select(0, pos.reshape(-1)).view(T, k, g.shape[1])
        return (dx.float().sum(1).to(g.dtype) if k > 1 else dx[:, 0]), None, None


def _quantize_grad(dy, row: bool, col: bool, seeds, rht: bool):
    return ops.quantize_dual(dy, row=row, col=col, row_sr_seed=seeds[0] if row else None,
                             col_sr_seed=seeds[1] if col else None, rht_seed=seeds[2] if col and rht else None,
                             col_pad=ALIGN)


class _GroupedSwiGLU(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for every expert's rows of x. cfg: (wgrad_rht, need_dx, need_w, fuse,
    out_dtype)."""

    @staticmethod
    def forward(ctx, xs, wg, wu, wd, tiles, segs_d, segs_f, caches, cfg):
        rht, need_dx, need_w, fuse, dt = cfg
        E, f, d = wg.shape
        need_a = need_dx or need_w  # the gradient flows back through down and the SwiGLU
        # [dG row SR, dG^T SR, RHT of x^T, dOut row SR, dOut^T SR, RHT of a^T, dU row SR, dU^T SR]
        seeds = draw_seeds(8, xs.device) if need_a or need_w else None
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN)
        wgq, wgt = _stack(wg, need_dx, caches[0], seeds[0] if need_dx else None)
        wuq, wut = _stack(wu, need_dx, caches[1], seeds[6] if need_dx else None)
        wdq, wdt = _stack(wd, need_a, caches[2], seeds[3] if need_a else None)
        g = ops.gemm_grouped(xq, wgq, "m", tiles, group_rows=f)
        u = ops.gemm_grouped(xq, wuq, "m", tiles, group_rows=f)
        aq, atq = ops.swiglu_quantize_dual(g, u, col=need_w, rht_seed=seeds[5] if need_w and rht else None,
                                           col_pad=ALIGN)
        y = ops.gemm_grouped(aq, wdq, "m", tiles, group_rows=d)
        saved = [g, u, seeds, tiles, segs_d, segs_f]
        for flag, q in ((need_w, xtq), (need_dx, wgt), (need_dx, wut), (need_w, atq), (need_a, wdt)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.fuse, ctx.x_dtype = cfg, fuse, xs.dtype
        ctx.params = (wg, wu, wd)
        return y if dt == BF16 else y.to(dt)

    @staticmethod
    def backward(ctx, dy):
        rht, need_dx, need_w, fuse, dt = ctx.cfg
        g, u, seeds, tiles, segs_d, segs_f, *rest = ctx.saved_tensors
        qs = {}
        for name, flag in (("xtq", need_w), ("wgt", need_dx), ("wut", need_dx), ("atq", need_w),
                           ("wdt", need_dx or need_w)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_dx, do_wg, do_wu, do_wd = need_dx and nin[0], need_w and nin[1], need_w and nin[2], need_w and nin[3]
        grads = [None] * 9
        do_a = do_dx or do_wg or do_wu
        doq, dotq = _quantize_grad(dy.contiguous(), do_a, do_wd, seeds[3:6], rht)
        wg, wu, wd = ctx.params
        if do_wd:
            grads[3] = _wgrad(ctx, wd, dotq, qs.pop("atq"), segs_d)
        del dotq
        if not do_a:
            return tuple(grads)
        E, f, d = wg.shape
        da = ops.gemm_grouped(doq, qs.pop("wdt"), "m", tiles, group_rows=f)  # gradient of the SwiGLU output
        del doq
        dx = None
        for k, (do_w, param, row_s, col_s, wt) in enumerate(((do_wg, wg, 0, 1, "wgt"), (do_wu, wu, 6, 7, "wut"))):
            if not (do_dx or do_w):
                continue
            dq, dtq = ops.swiglu_backward_quantize_dual(
                da, g, u, gate=k == 0, row=do_dx, col=do_w, col_pad=ALIGN,
                row_sr_seed=seeds[row_s] if do_dx else None, col_sr_seed=seeds[col_s] if do_w else None,
                rht_seed=seeds[2] if do_w and rht else None)
            if do_w:  # dW = dG^T x per expert: both operands carry x^T's RHT
                grads[1 + k] = _wgrad(ctx, param, dtq, qs["xtq"], segs_f)
            del dtq
            if do_dx:  # dx = dG Wg + dU Wu, accumulated in fp32
                w = qs.pop(wt)
                if dx is None:
                    dx = ops.gemm_grouped(dq, w, "m", tiles, group_rows=d, out_dtype=torch.float32)
                else:
                    ops.gemm_grouped(dq, w, "m", tiles, group_rows=d, out_dtype=torch.float32, out=dx,
                                     accumulate=True)
            del dq
        if do_dx:
            grads[0] = dx.to(ctx.x_dtype)
        return tuple(grads)


class _GroupedGelu(torch.autograd.Function):
    """fc2(gelu(fc1(x))) for every expert's rows of x. cfg: (wgrad_rht, need_dx, need_w, fuse, out_dtype,
    approximate)."""

    @staticmethod
    def forward(ctx, xs, w1, w2, tiles, segs_d, segs_f, caches, cfg):
        rht, need_dx, need_w, fuse, dt, approximate = cfg
        E, f, d = w1.shape
        need_h = need_dx or need_w
        seeds = draw_seeds(6, xs.device) if need_h else None  # [dH row SR, dH^T SR, RHT x^T, dOut row, col, RHT]
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN)
        w1q, w1t = _stack(w1, need_dx, caches[0], seeds[0] if need_dx else None)
        w2q, w2t = _stack(w2, need_h, caches[1], seeds[3] if need_h else None)
        h = ops.gemm_grouped(xq, w1q, "m", tiles, group_rows=f)
        gq, gtq = ops.gelu_quantize_dual(h, approximate=approximate, col=need_w,
                                         rht_seed=seeds[5] if need_w and rht else None, col_pad=ALIGN)
        y = ops.gemm_grouped(gq, w2q, "m", tiles, group_rows=d)
        saved = [h, seeds, tiles, segs_d, segs_f]
        for flag, q in ((need_w, xtq), (need_dx, w1t), (need_w, gtq), (need_h, w2t)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.fuse, ctx.x_dtype = cfg, fuse, xs.dtype
        ctx.params = (w1, w2)
        return y if dt == BF16 else y.to(dt)

    @staticmethod
    def backward(ctx, dy):
        rht, need_dx, need_w, fuse, dt, approximate = ctx.cfg
        h, seeds, tiles, segs_d, segs_f, *rest = ctx.saved_tensors
        qs = {}
        for name, flag in (("xtq", need_w), ("w1t", need_dx), ("gtq", need_w), ("w2t", need_dx or need_w)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_dx, do_w1, do_w2 = need_dx and nin[0], need_w and nin[1], need_w and nin[2]
        grads = [None] * 8
        do_h = do_dx or do_w1
        doq, dotq = _quantize_grad(dy.contiguous(), do_h, do_w2, seeds[3:6], rht)
        w1, w2 = ctx.params
        if do_w2:
            grads[2] = _wgrad(ctx, w2, dotq, qs.pop("gtq"), segs_d)
        del dotq
        if not do_h:
            return tuple(grads)
        E, f, d = w1.shape
        dg = ops.gemm_grouped(doq, qs.pop("w2t"), "m", tiles, group_rows=f)  # gradient of gelu(h)
        del doq
        dhq, dhtq = ops.gelu_backward_quantize_dual(dg, h, approximate=approximate, row=do_dx, col=do_w1,
                                                    col_pad=ALIGN, row_sr_seed=seeds[0] if do_dx else None,
                                                    col_sr_seed=seeds[1] if do_w1 else None,
                                                    rht_seed=seeds[2] if do_w1 and rht else None)
        del dg
        if do_w1:
            grads[1] = _wgrad(ctx, w1, dhtq, qs.pop("xtq"), segs_f)
        del dhtq
        if do_dx:
            grads[0] = ops.gemm_grouped(dhq, qs.pop("w1t"), "m", tiles, group_rows=d,
                                        out_dtype=torch.float32).to(ctx.x_dtype)
        return tuple(grads)


class MoE(nn.Module):
    """Dropless top-k mixture-of-experts MLP (see the module docstring).

    d, d_ff: model and expert hidden sizes (multiples of 128 for NVFP4 experts); activation: "swiglu",
    "gelu_tanh" or "gelu". After forward, ``aux_loss`` holds the Switch Transformer load-balancing loss
    E * sum_e f_e P_e (f_e: fraction of routed slots, P_e: mean router probability of expert e), 1.0 when
    balanced; add ``coef * moe.aux_loss`` to the training loss. ``last_counts``: tokens per expert."""

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int = 2, activation: str = "swiglu",
                 normalize_weights: bool = True, device=None, dtype=None):
        super().__init__()
        if activation not in ("swiglu", "gelu_tanh", "gelu"):
            raise ValueError(f"activation must be swiglu, gelu_tanh or gelu, got {activation!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")
        self.d, self.d_ff, self.n_experts, self.top_k = d, d_ff, n_experts, top_k
        self.activation, self.normalize_weights = activation, normalize_weights
        self.router = nn.Linear(d, n_experts, bias=False, device=device, dtype=dtype)
        kw = dict(device=device, dtype=dtype)
        names = (("w_gate", d_ff, d), ("w_up", d_ff, d), ("w_down", d, d_ff)) if activation == "swiglu" else \
            (("w_fc1", d_ff, d), ("w_fc2", d, d_ff))
        for name, n, k in names:
            self.register_parameter(name, nn.Parameter(torch.empty(n_experts, n, k, **kw)))
        self._names = tuple(n for n, _, _ in names)
        self.reset_parameters()
        self.nvfp4 = False
        self.wgrad_rht = True
        self.fuse_wgrad_accumulation = False
        self.cache_weight_quantization = False
        self._caches = tuple(WeightQuantCache() for _ in self._names)
        self.aux_loss: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        """nn.Linear's default initialization for every expert matrix (uniform, bound 1/sqrt(fan_in))."""
        with torch.no_grad():
            for name in self._names:
                w = getattr(self, name)
                w.uniform_(-1 / math.sqrt(w.shape[-1]), 1 / math.sqrt(w.shape[-1]))

    def to_nvfp4(self, *, wgrad_rht: bool = True, fuse_wgrad_accumulation: bool = False,
                 cache_weight_quantization: bool = False, backend: str = "cuda") -> "MoE":
        """Run the experts in NVFP4 (``convert_linears`` calls this with its keyword arguments)."""
        if self.d % ALIGN or self.d_ff % ALIGN:
            raise ValueError(f"NVFP4 experts need d and d_ff divisible by {ALIGN}, got {self.d}, {self.d_ff}")
        if backend != "cuda":
            raise ValueError("NVFP4 experts need the CUDA backend")
        self.nvfp4 = True
        self.wgrad_rht, self.fuse_wgrad_accumulation = wgrad_rht, fuse_wgrad_accumulation
        self.cache_weight_quantization = cache_weight_quantization
        return self

    def experts_weights(self):
        return tuple(getattr(self, n) for n in self._names)

    def __getstate__(self):
        """Copies and pickles leave out the last forward's outputs (``aux_loss`` is a non-leaf tensor, which
        deepcopy refuses; neither belongs to the module's state)."""
        state = self.__dict__.copy()
        state["aux_loss"] = None
        state["_counts"] = None
        return state

    @property
    def last_counts(self) -> list[int] | None:
        """Tokens routed to each expert in the last forward (read from the GPU: synchronizes)."""
        return None if self._counts is None else self._counts.tolist()

    def route(self, x2d: torch.Tensor):
        """(weights [T, k] fp32, experts [T, k] int64, router probabilities [T, E] fp32)."""
        probs = torch.softmax(self.router(x2d).float(), dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.normalize_weights and self.top_k > 1:
            w = w / w.sum(-1, keepdim=True)
        return w, idx, probs

    def _experts_loop(self, xs: torch.Tensor, counts: list[int]) -> torch.Tensor:
        """BF16 / reference path: each expert's slice of the sorted tokens, one after another."""
        ws = [getattr(self, n).unbind(0) for n in self._names]  # unbind: one stacked gradient, not E zero-fills
        outs = []
        for e, part in enumerate(xs.split(counts)):
            if not counts[e]:
                continue
            if self.activation == "swiglu":
                outs.append(F.linear(F.silu(F.linear(part, ws[0][e])) * F.linear(part, ws[1][e]), ws[2][e]))
            else:
                approx = "tanh" if self.activation == "gelu_tanh" else "none"
                outs.append(F.linear(F.gelu(F.linear(part, ws[0][e]), approximate=approx), ws[1][e]))
        return torch.cat(outs, 0)

    def _experts_grouped(self, x2d, token, slot_expert, slot_of_sorted, counts: torch.Tensor):
        """(expert outputs of the padded buffer [M, d], the buffer row of every (token, choice) slot [T, k]).

        No host synchronization: the buffer has the worst-case size for T tokens, M = T k + E (128 - 1) rounded
        up to 256 (every expert's rows padded to 128), and the layout tables are computed on the GPU from the
        device-side counts. Tiles past the used rows are tagged -1 (the grouped GEMM stores zeros there without
        loading anything), and the ragged-K segments of experts without tokens are empty."""
        dev, E, d, f = x2d.device, self.n_experts, self.d, self.d_ff
        T, k = x2d.shape[0], self.top_k
        m_pad = (T * k + E * (_ROWS - 1) + 255) // 256 * 256  # also the column operands' K: a multiple of 256
        padded = (counts + _ROWS - 1) // _ROWS * _ROWS
        pend = torch.cumsum(padded, 0)
        pstart = pend - padded
        cstart = torch.cumsum(counts, 0) - counts
        tile_row = torch.arange(0, m_pad, _ROWS, device=dev)
        tiles_t = torch.searchsorted(pend, tile_row, right=True)
        tiles_t = torch.where(tiles_t < E, tiles_t, -1).to(torch.int32)  # rows past the last expert: unused
        experts = torch.arange(E, device=dev)
        segs_d = torch.stack([pstart, padded, experts * d], 1).to(torch.int32)
        segs_f = torch.stack([pstart, padded, experts * f], 1).to(torch.int32)
        dest = torch.arange(token.numel(), device=dev) - cstart[slot_expert] + pstart[slot_expert]
        src = torch.full((m_pad,), T, device=dev, dtype=torch.long).index_copy_(0, dest, token)
        pos = torch.empty_like(dest).index_copy_(0, slot_of_sorted, dest).view(T, k)
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x2d.dtype
        xs = _Dispatch.apply(x2d.to(BF16), src, pos)  # bf16 first: the quantizer reads bf16 (as NVFP4Linear)
        grad = torch.is_grad_enabled()
        ws = self.experts_weights()
        need_w = grad and any(w.requires_grad for w in ws)
        need_dx = grad and xs.requires_grad
        caches = self._caches if self.cache_weight_quantization else (None,) * len(ws)
        if self.activation == "swiglu":
            y = _GroupedSwiGLU.apply(xs, *ws, tiles_t, segs_d, segs_f, caches,
                                     (self.wgrad_rht, need_dx, need_w, self.fuse_wgrad_accumulation, dt))
        else:
            approx = "tanh" if self.activation == "gelu_tanh" else "none"
            y = _GroupedGelu.apply(xs, *ws, tiles_t, segs_d, segs_f, caches,
                                   (self.wgrad_rht, need_dx, need_w, self.fuse_wgrad_accumulation, dt, approx))
        return y, pos

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2d = x.reshape(-1, self.d)
        T, k, E = x2d.shape[0], self.top_k, self.n_experts
        w, idx, probs = self.route(x2d)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)  # (token, choice) slots grouped by expert
        token = order // k
        # integer adds commute, so the scatter is deterministic (bincount's output size is data-dependent)
        counts = torch.zeros(E, device=x.device, dtype=torch.long).scatter_add_(0, flat, torch.ones_like(flat))
        self._counts = counts
        frac = counts.float() / max(1, T * k)
        self.aux_loss = E * (frac * probs.mean(0)).sum()
        if self.nvfp4 and not ops.high_precision() and x2d.is_cuda:
            ys, pos = self._experts_grouped(x2d, token, flat.index_select(0, order), order, counts)
        else:  # the BF16 loop splits the sorted rows: one host synchronization for the expert sizes
            pos = torch.empty_like(order).index_copy_(0, order, torch.arange(T * k, device=x.device)).view(T, k)
            ys = self._experts_loop(_Dispatch.apply(x2d, token, pos), counts.tolist())
        # combine: y[t] = sum_j w[t, j] ys[pos[t, j]], a gather and a k-term weighted sum (deterministic; the
        # gather's backward writes every row once)
        g = ys.index_select(0, pos.reshape(-1)).view(T, k, self.d)
        y = torch.bmm(w.to(g.dtype).unsqueeze(1), g).squeeze(1)
        return y.reshape(*lead, self.d)

    def extra_repr(self) -> str:
        r = f"d={self.d}, d_ff={self.d_ff}, n_experts={self.n_experts}, top_k={self.top_k}, " \
            f"activation={self.activation}"
        return r + (", nvfp4=True" if self.nvfp4 else "")


__all__ = ["MoE"]
