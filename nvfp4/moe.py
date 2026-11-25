"""Top-k mixture-of-experts MLP whose experts train in NVFP4 with grouped GEMMs.

    from nvfp4.moe import MoE
    moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()
    nvfp4.convert_linears(moe)               # experts -> NVFP4 (moe.to_nvfp4(...)); the router stays BF16
    y = moe(x)                               # x [..., d]; moe.aux_loss: Switch load-balancing loss

Routing: softmax over the router logits (fp32), top-k experts per token, weights renormalized over the k
choices. The (token, expert) pairs are sorted by expert (stable). No tokens are dropped (no capacity factor).
The layer never reads the expert sizes on the host: its buffers have the worst-case size for the number of
tokens, and the layout is computed on the GPU, so a training step with MoE layers can be captured in a CUDA
graph and traced by ``torch.compile`` without graph breaks. That holds for the NVFP4 experts and for the
high-precision ones (Triton grouped GEMMs, nvfp4._bf16_grouped); only without Triton (native Windows) or
with ``bf16_impl="loop"`` do the high-precision experts run one after another, reading the sizes once per
call.

Expert weights are stacked parameters, ``w_gate`` / ``w_up`` [E, d_ff, d] and ``w_down`` [E, d, d_ff]
(SwiGLU) or ``w_fc1`` [E, d_ff, d] and ``w_fc2`` [E, d, d_ff] (GELU), and with ``bias=True`` the biases
``b_gate`` / ``b_up`` [E, d_ff], ``b_down`` [E, d] (``b_fc1``, ``b_fc2``). All experts run in one GEMM
launch per projection:

  * the routed tokens are gathered into one buffer, each expert's rows padded to a multiple of 128 (zero
    rows; T k + E 127 rows in all, the unused tail tiles skipped by the GEMMs), and quantized once (one
    tensor scale, like a dense layer's activation);
  * forward and input-gradient GEMMs are grouped "ragged M" GEMMs (``ops.gemm_grouped(mode="m")``): every
    128-row tile reads its expert's weight rows; the stacked weights are quantized as one [E N, K] tensor
    (16x16 blocks, so the transposed copy is the same quantization, rearranged per expert); the biases are
    added in the GEMM's store, to the real rows only (padding rows stay zero);
  * the weight gradients are one grouped "ragged K" GEMM (``mode="k"``): each expert's output tiles reduce
    over its own token range of the column-quantized operands (RHT along the tokens, within each expert's
    128-aligned range), accumulated into ``.grad`` in place with ``fuse_wgrad_accumulation``; the bias
    gradients come from the column sums the gradient quantizers compute anyway;
  * activation (SwiGLU / GELU) and its backward are fused into the quantizers, as in nvfp4.fused.

Per step that is a fixed number of kernels, independent of the number of experts.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvfp4 import _bf16_grouped, ops
from nvfp4._bf16_grouped import GroupedLinear, expert_sums
from nvfp4.linear import ALIGN, WeightQuantCache, _accumulation_target, dgrad_weight_operand, draw_seeds

BF16 = torch.bfloat16
_ROWS = _bf16_grouped.ROWS  # 128: every expert's rows start on a GEMM tile


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


def _bias_grad(partial: torch.Tensor, tiles: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Per-expert bias gradient from a quantizer's per-64-row column sums (two per 128-row tile; padding rows
    and unused tiles carry zero gradient: nothing downstream reads them)."""
    return expert_sums(partial, tiles.repeat_interleave(_ROWS // 64), bias.shape[0]).to(bias.dtype)


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


def _quantize_grad(dy, row: bool, col: bool, seeds, rht: bool, bias_grad: bool = False):
    """(row, col[, per-64-row column sums]) of an output gradient. A bias gradient alone still makes the row
    output (the sums come from the quantizer's read)."""
    row = row or (bias_grad and not col)
    return ops.quantize_dual(dy, row=row, col=col, row_sr_seed=seeds[0] if row else None,
                             col_sr_seed=seeds[1] if col else None, rht_seed=seeds[2] if col and rht else None,
                             col_pad=ALIGN, bias_grad="tiles" if bias_grad else False)


class _GroupedSwiGLU(torch.autograd.Function):
    """down(silu(gate(x)) * up(x)) for every expert's rows of x, biases optional. cfg: (wgrad_rht, need_dx,
    need_w, need_b, fuse, out_dtype)."""

    @staticmethod
    def forward(ctx, xs, wg, wu, wd, bg, bu, bd, tiles, trows, segs_d, segs_f, caches, cfg):
        rht, need_dx, need_w, need_b, fuse, dt = cfg
        E, f, d = wg.shape
        need_a = need_dx or need_w or need_b  # the gradient flows back through down and the SwiGLU
        # [dG row SR, dG^T SR, RHT of x^T, dOut row SR, dOut^T SR, RHT of a^T, dU row SR, dU^T SR]
        seeds = draw_seeds(8, xs.device) if need_a else None
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN)
        wgq, wgt = _stack(wg, need_dx, caches[0], seeds[0] if need_dx else None)
        wuq, wut = _stack(wu, need_dx, caches[1], seeds[6] if need_dx else None)
        wdq, wdt = _stack(wd, need_a, caches[2], seeds[3] if need_a else None)
        g = ops.gemm_grouped(xq, wgq, "m", tiles, group_rows=f, bias=bg, tile_rows=trows)
        u = ops.gemm_grouped(xq, wuq, "m", tiles, group_rows=f, bias=bu, tile_rows=trows)
        aq, atq = ops.swiglu_quantize_dual(g, u, col=need_w, rht_seed=seeds[5] if need_w and rht else None,
                                           col_pad=ALIGN)
        y = ops.gemm_grouped(aq, wdq, "m", tiles, group_rows=d, bias=bd, tile_rows=trows)
        saved = [g, u, seeds, tiles, segs_d, segs_f]
        for flag, q in ((need_w, xtq), (need_dx, wgt), (need_dx, wut), (need_w, atq), (need_a, wdt)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.fuse, ctx.x_dtype = cfg, fuse, xs.dtype
        ctx.params = (wg, wu, wd, bg, bu, bd)
        return y if dt == BF16 else y.to(dt)

    @staticmethod
    def backward(ctx, dy):
        rht, need_dx, need_w, need_b, fuse, dt = ctx.cfg
        g, u, seeds, tiles, segs_d, segs_f, *rest = ctx.saved_tensors
        qs = {}
        for name, flag in (("xtq", need_w), ("wgt", need_dx), ("wut", need_dx), ("atq", need_w),
                           ("wdt", need_dx or need_w or need_b)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_dx, do_wg, do_wu, do_wd = need_dx and nin[0], need_w and nin[1], need_w and nin[2], need_w and nin[3]
        do_bg, do_bu, do_bd = nin[4], nin[5], nin[6]
        grads = [None] * 13
        do_a = do_dx or do_wg or do_wu or do_bg or do_bu
        wg, wu, wd, bg, bu, bd = ctx.params
        doq, dotq, *part = _quantize_grad(dy.contiguous(), do_a, do_wd, seeds[3:6], rht, do_bd)
        if do_bd:
            grads[6] = _bias_grad(part[0], tiles, bd)
        if do_wd:
            grads[3] = _wgrad(ctx, wd, dotq, qs.pop("atq"), segs_d)
        del dotq
        if not do_a:
            return tuple(grads)
        E, f, d = wg.shape
        da = ops.gemm_grouped(doq, qs.pop("wdt"), "m", tiles, group_rows=f)  # gradient of the SwiGLU output
        del doq
        dx = None
        for k, (do_w, do_b, param, bias, row_s, col_s, wt) in enumerate(
                ((do_wg, do_bg, wg, bg, 0, 1, "wgt"), (do_wu, do_bu, wu, bu, 6, 7, "wut"))):
            if not (do_dx or do_w or do_b):
                continue
            row = do_dx or not do_w  # a bias gradient alone still needs one output
            dq, dtq, *part = ops.swiglu_backward_quantize_dual(
                da, g, u, gate=k == 0, row=row, col=do_w, col_pad=ALIGN, bias_grad="tiles" if do_b else False,
                row_sr_seed=seeds[row_s] if row else None, col_sr_seed=seeds[col_s] if do_w else None,
                rht_seed=seeds[2] if do_w and rht else None)
            if do_b:
                grads[4 + k] = _bias_grad(part[0], tiles, bias)
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
    """fc2(gelu(fc1(x))) for every expert's rows of x, biases optional. cfg: (wgrad_rht, need_dx, need_w,
    need_b, fuse, out_dtype, approximate)."""

    @staticmethod
    def forward(ctx, xs, w1, w2, b1, b2, tiles, trows, segs_d, segs_f, caches, cfg):
        rht, need_dx, need_w, need_b, fuse, dt, approximate = cfg
        E, f, d = w1.shape
        need_h = need_dx or need_w or need_b
        seeds = draw_seeds(6, xs.device) if need_h else None  # [dH row SR, dH^T SR, RHT x^T, dOut row, col, RHT]
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN)
        w1q, w1t = _stack(w1, need_dx, caches[0], seeds[0] if need_dx else None)
        w2q, w2t = _stack(w2, need_h, caches[1], seeds[3] if need_h else None)
        h = ops.gemm_grouped(xq, w1q, "m", tiles, group_rows=f, bias=b1, tile_rows=trows)  # h includes b1
        gq, gtq = ops.gelu_quantize_dual(h, approximate=approximate, col=need_w,
                                         rht_seed=seeds[5] if need_w and rht else None, col_pad=ALIGN)
        y = ops.gemm_grouped(gq, w2q, "m", tiles, group_rows=d, bias=b2, tile_rows=trows)
        saved = [h, seeds, tiles, segs_d, segs_f]
        for flag, q in ((need_w, xtq), (need_dx, w1t), (need_w, gtq), (need_h, w2t)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.fuse, ctx.x_dtype = cfg, fuse, xs.dtype
        ctx.params = (w1, w2, b1, b2)
        return y if dt == BF16 else y.to(dt)

    @staticmethod
    def backward(ctx, dy):
        rht, need_dx, need_w, need_b, fuse, dt, approximate = ctx.cfg
        h, seeds, tiles, segs_d, segs_f, *rest = ctx.saved_tensors
        qs = {}
        for name, flag in (("xtq", need_w), ("w1t", need_dx), ("gtq", need_w),
                           ("w2t", need_dx or need_w or need_b)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_dx, do_w1, do_w2 = need_dx and nin[0], need_w and nin[1], need_w and nin[2]
        do_b1, do_b2 = nin[3], nin[4]
        grads = [None] * 11
        do_h = do_dx or do_w1 or do_b1
        w1, w2, b1, b2 = ctx.params
        doq, dotq, *part = _quantize_grad(dy.contiguous(), do_h, do_w2, seeds[3:6], rht, do_b2)
        if do_b2:
            grads[4] = _bias_grad(part[0], tiles, b2)
        if do_w2:
            grads[2] = _wgrad(ctx, w2, dotq, qs.pop("gtq"), segs_d)
        del dotq
        if not do_h:
            return tuple(grads)
        E, f, d = w1.shape
        dg = ops.gemm_grouped(doq, qs.pop("w2t"), "m", tiles, group_rows=f)  # gradient of gelu(h)
        del doq
        row = do_dx or not do_w1
        dhq, dhtq, *part = ops.gelu_backward_quantize_dual(
            dg, h, approximate=approximate, row=row, col=do_w1, col_pad=ALIGN,
            bias_grad="tiles" if do_b1 else False, row_sr_seed=seeds[0] if row else None,
            col_sr_seed=seeds[1] if do_w1 else None, rht_seed=seeds[2] if do_w1 and rht else None)
        del dg
        if do_b1:
            grads[3] = _bias_grad(part[0], tiles, b1)
        if do_w1:
            grads[1] = _wgrad(ctx, w1, dhtq, qs.pop("xtq"), segs_f)
        del dhtq
        if do_dx:
            grads[0] = ops.gemm_grouped(dhq, qs.pop("w1t"), "m", tiles, group_rows=d,
                                        out_dtype=torch.float32).to(ctx.x_dtype)
        return tuple(grads)


class _Layout(NamedTuple):
    """Where the routed slots go in the expert buffer, computed on the GPU (MoE._layout)."""
    tiles: torch.Tensor      # int32 [M / 128]: expert of each 128-row tile, -1 for unused tiles
    tile_rows: torch.Tensor  # int32 [M / 128]: real (not padding) rows of each tile
    segs_d: torch.Tensor     # int32 [E, 3]: (first row, padded rows, e d) per expert (ragged-K GEMMs)
    segs_f: torch.Tensor     # int32 [E, 3]: (first row, padded rows, e d_ff)
    src: torch.Tensor        # int64 [M]: token of each buffer row (T: a zero row)
    pos: torch.Tensor        # int64 [T, k]: buffer row of each (token, choice) slot


class MoE(nn.Module):
    """Dropless top-k mixture-of-experts MLP (see the module docstring).

    d, d_ff: model and expert hidden sizes (multiples of 128 for NVFP4 experts); activation: "swiglu",
    "gelu_tanh" or "gelu"; bias: per-expert biases on every expert projection. After forward, ``aux_loss``
    holds the Switch Transformer load-balancing loss E * sum_e f_e P_e (f_e: fraction of routed slots, P_e:
    mean router probability of expert e), 1.0 when balanced; add ``coef * moe.aux_loss`` to the training loss.
    ``last_counts``: tokens per expert. ``bf16_impl``: how the high-precision experts run, "auto" (grouped
    Triton GEMMs on CUDA when Triton is installed, else the loop), "grouped" or "loop" (one expert after
    another, reading the expert sizes on the host)."""

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int = 2, activation: str = "swiglu",
                 normalize_weights: bool = True, bias: bool = False, device=None, dtype=None):
        super().__init__()
        if activation not in ("swiglu", "gelu_tanh", "gelu"):
            raise ValueError(f"activation must be swiglu, gelu_tanh or gelu, got {activation!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")
        self.d, self.d_ff, self.n_experts, self.top_k = d, d_ff, n_experts, top_k
        self.activation, self.normalize_weights = activation, normalize_weights
        self.router = nn.Linear(d, n_experts, bias=False, device=device, dtype=dtype)
        kw = dict(device=device, dtype=dtype)
        names = (("gate", d_ff, d), ("up", d_ff, d), ("down", d, d_ff)) if activation == "swiglu" else \
            (("fc1", d_ff, d), ("fc2", d, d_ff))
        for name, n, k in names:
            self.register_parameter("w_" + name, nn.Parameter(torch.empty(n_experts, n, k, **kw)))
            self.register_parameter("b_" + name, nn.Parameter(torch.empty(n_experts, n, **kw)) if bias else None)
        self._names = tuple("w_" + n for n, _, _ in names)
        self._bias_names = tuple("b_" + n for n, _, _ in names)
        self.has_bias = bias
        self.reset_parameters()
        self.nvfp4 = False
        self.bf16_impl = "auto"
        self.wgrad_rht = True
        self.fuse_wgrad_accumulation = False
        self.cache_weight_quantization = False
        self._caches = tuple(WeightQuantCache() for _ in self._names)
        self.aux_loss: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        """nn.Linear's default initialization for every expert matrix and bias (uniform, bound 1/sqrt(fan_in))."""
        with torch.no_grad():
            for w, b in zip(self.experts_weights(), self.experts_biases()):
                bound = 1 / math.sqrt(w.shape[-1])
                w.uniform_(-bound, bound)
                if b is not None:
                    b.uniform_(-bound, bound)

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

    def experts_biases(self):
        """The expert biases in the order of experts_weights (None each without ``bias``)."""
        return tuple(getattr(self, n) for n in self._bias_names)

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

    def _bf16_grouped(self, x: torch.Tensor) -> bool:
        if self.bf16_impl not in ("auto", "grouped", "loop"):
            raise ValueError(f"bf16_impl must be auto, grouped or loop, got {self.bf16_impl!r}")
        if self.bf16_impl == "loop" or not x.is_cuda:
            return False
        if not _bf16_grouped.available():
            if self.bf16_impl == "grouped":
                raise RuntimeError("bf16_impl='grouped' needs Triton")
            return False
        return True

    def _experts_loop(self, xs: torch.Tensor, counts: list[int]) -> torch.Tensor:
        """Reference path: each expert's slice of the sorted tokens, one after another."""
        ws = [w.unbind(0) for w in self.experts_weights()]  # unbind: one stacked gradient, not E zero-fills
        bs = [(b.unbind(0) if b is not None else (None,) * self.n_experts) for b in self.experts_biases()]
        outs = []
        for e, part in enumerate(xs.split(counts)):
            if not counts[e]:
                continue
            if self.activation == "swiglu":
                a = F.silu(F.linear(part, ws[0][e], bs[0][e])) * F.linear(part, ws[1][e], bs[1][e])
                outs.append(F.linear(a, ws[2][e], bs[2][e]))
            else:
                approx = "tanh" if self.activation == "gelu_tanh" else "none"
                h = F.gelu(F.linear(part, ws[0][e], bs[0][e]), approximate=approx)
                outs.append(F.linear(h, ws[1][e], bs[1][e]))
        return torch.cat(outs, 0)

    def _layout(self, token, slot_expert, slot_of_sorted, counts: torch.Tensor, T: int) -> _Layout:
        """The padded buffer's layout, computed on the GPU from the device-side counts (no host
        synchronization). The buffer has the worst-case size for T tokens, M = T k + E (128 - 1) rounded up to
        256 (every expert's rows padded to 128; also the column operands' K, a multiple of 256). Tiles past the
        used rows are tagged -1 (the grouped GEMMs store zeros there without loading anything), and the
        ragged-K segments of experts without tokens are empty."""
        dev, E, k = counts.device, self.n_experts, self.top_k
        m_pad = (T * k + E * (_ROWS - 1) + 255) // 256 * 256
        padded = (counts + _ROWS - 1) // _ROWS * _ROWS
        pend = torch.cumsum(padded, 0)
        pstart = pend - padded
        cstart = torch.cumsum(counts, 0) - counts
        tile_row = torch.arange(0, m_pad, _ROWS, device=dev)
        t = torch.searchsorted(pend, tile_row, right=True)
        used = t < E
        te = t.clamp(max=E - 1)
        tile_rows = torch.where(used, (pstart[te] + counts[te] - tile_row).clamp(0, _ROWS), 0).to(torch.int32)
        tiles = torch.where(used, t, -1).to(torch.int32)
        experts = torch.arange(E, device=dev)
        segs_d = torch.stack([pstart, padded, experts * self.d], 1).to(torch.int32)
        segs_f = torch.stack([pstart, padded, experts * self.d_ff], 1).to(torch.int32)
        dest = torch.arange(token.numel(), device=dev) - cstart[slot_expert] + pstart[slot_expert]
        src = torch.full((m_pad,), T, device=dev, dtype=torch.long).index_copy_(0, dest, token)
        pos = torch.empty_like(dest).index_copy_(0, slot_of_sorted, dest).view(T, k)
        return _Layout(tiles, tile_rows, segs_d, segs_f, src, pos)

    def _experts_nvfp4(self, x2d, lay: _Layout):
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x2d.dtype
        xs = _Dispatch.apply(x2d.to(BF16), lay.src, lay.pos)  # bf16 first: the quantizer reads bf16 (as NVFP4Linear)
        grad = torch.is_grad_enabled()
        ws, bs = self.experts_weights(), self.experts_biases()
        need_w = grad and any(w.requires_grad for w in ws)
        need_b = grad and any(b is not None and b.requires_grad for b in bs)
        need_dx = grad and xs.requires_grad
        caches = self._caches if self.cache_weight_quantization else (None,) * len(ws)
        if self.activation == "swiglu":
            return _GroupedSwiGLU.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                        (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt))
        approx = "tanh" if self.activation == "gelu_tanh" else "none"
        return _GroupedGelu.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                  (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt, approx))

    def _experts_bf16_grouped(self, x2d, lay: _Layout):
        """The high-precision experts on the padded buffer with grouped Triton GEMMs (autocast honoured as
        F.linear would: inputs, weights and biases in the autocast dtype)."""
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x2d.dtype
        xs = _Dispatch.apply(x2d, lay.src, lay.pos)
        with torch.autocast("cuda", enabled=False):
            xs = xs.to(dt)
            ws = [w.to(dt) for w in self.experts_weights()]
            bs = [None if b is None else b.to(dt) for b in self.experts_biases()]
            lin = lambda a, i, segs: GroupedLinear.apply(a, ws[i], bs[i], lay.tiles, lay.tile_rows, segs)  # noqa: E731
            if self.activation == "swiglu":
                a = F.silu(lin(xs, 0, lay.segs_f)) * lin(xs, 1, lay.segs_f)
                return lin(a, 2, lay.segs_d)
            approx = "tanh" if self.activation == "gelu_tanh" else "none"
            return lin(F.gelu(lin(xs, 0, lay.segs_f), approximate=approx), 1, lay.segs_d)

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
        nvfp4 = self.nvfp4 and not ops.high_precision() and x2d.is_cuda
        if nvfp4 or self._bf16_grouped(x2d):
            lay = self._layout(token, flat.index_select(0, order), order, counts, T)
            ys = self._experts_nvfp4(x2d, lay) if nvfp4 else self._experts_bf16_grouped(x2d, lay)
            pos = lay.pos
        else:  # the loop splits the sorted rows: one host synchronization for the expert sizes
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
        return r + (", bias=True" if self.has_bias else "") + (", nvfp4=True" if self.nvfp4 else "")


__all__ = ["MoE"]
