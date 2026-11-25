"""Top-k mixture-of-experts MLP whose experts train in NVFP4 with grouped GEMMs.

    from nvfp4.moe import MoE
    moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()
    nvfp4.convert_linears(moe)               # experts -> NVFP4 (moe.to_nvfp4(...)); the router stays BF16
    y = moe(x)                               # x [..., d]; moe.aux_loss: Switch load-balancing loss

Routing: softmax over the router logits (fp32), top-k experts per token, weights renormalized over the k
choices. The (token, expert) pairs are sorted by expert (stable). Dropless by default; with
``capacity_factor`` each expert keeps its first C = ceil(capacity_factor T k / E) slots in token order. Options:
a shared expert (``shared_d_ff``: a dense MLP every token goes through, added to the routed output) and
expert parallelism (``ep_group``: each rank holds E / W experts; slots travel by all-to-alls, of a fixed size
with a capacity; dropless expert parallelism reads the row counts on the host).
Otherwise the layer never reads the expert sizes on the host: its buffers have the worst-case size for the number of
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
import torch.distributed as dist
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
    """Where a set of slots goes in an expert buffer, computed on the GPU (_plan)."""
    tiles: torch.Tensor      # int32 [M / 128]: expert of each 128-row tile, -1 for unused tiles
    tile_rows: torch.Tensor  # int32 [M / 128]: real (not padding) rows of each tile
    segs_d: torch.Tensor     # int32 [E, 3]: (first row, padded rows, e d) per expert (ragged-K GEMMs)
    segs_f: torch.Tensor     # int32 [E, 3]: (first row, padded rows, e f)
    src: torch.Tensor        # int64 [M]: source row of each buffer row (n_src: a zero row)
    pos: torch.Tensor        # int64 [S / k, k]: buffer row of each slot (a dropped slot: the zero row M - 1)


def _plan(slot_expert: torch.Tensor, slot_src: torch.Tensor, n_src: int, n_experts: int, k: int, d: int, f: int,
          max_rows: int, drops: bool) -> tuple[_Layout, torch.Tensor]:
    """(layout, rows per expert) of S slots: slot i reads source row slot_src[i] and goes to expert
    slot_expert[i] (-1: dropped). The slots are grouped by expert (stable: in slot order), each expert's rows
    padded to a multiple of 128. Nothing is read on the host: the buffer has the worst-case size for at most
    ``max_rows`` kept slots, M = max_rows + E (128 - 1) (+ 1 with ``drops``: a row that is always zero, where
    dropped slots point) rounded up to 256 (also the column operands' K, a multiple of 256). Tiles past the
    used rows are tagged -1 (the grouped GEMMs store zeros there without loading anything), and the
    ragged-K segments of experts without rows are empty."""
    dev, E, S = slot_expert.device, n_experts, slot_expert.numel()
    key = torch.where(slot_expert >= 0, slot_expert, E) if drops else slot_expert
    order = torch.argsort(key, stable=True)
    # integer adds commute, so the scatter is deterministic (bincount's output size is data-dependent)
    counts = torch.zeros(E + 1, device=dev, dtype=torch.long).scatter_add_(0, key, torch.ones_like(key))[:E]
    m_pad = (max_rows + E * (_ROWS - 1) + int(drops) + 255) // 256 * 256
    padded = (counts + _ROWS - 1) // _ROWS * _ROWS
    pend = torch.cumsum(padded, 0)
    pstart = pend - padded
    cstart = torch.cumsum(counts, 0) - counts
    ks = key.index_select(0, order)
    kc = ks.clamp(max=E - 1)
    dest = torch.arange(S, device=dev) - cstart[kc] + pstart[kc]
    src_of = slot_src.index_select(0, order)
    if drops:
        kept = ks < E
        dest = torch.where(kept, dest, m_pad - 1)
        src_of = torch.where(kept, src_of, n_src)
    src = torch.full((m_pad,), n_src, device=dev, dtype=torch.long).index_copy_(0, dest, src_of)
    pos = torch.empty_like(dest).index_copy_(0, order, dest).view(-1, k)
    tile_row = torch.arange(0, m_pad, _ROWS, device=dev)
    t = torch.searchsorted(pend, tile_row, right=True)
    used = t < E
    te = t.clamp(max=E - 1)
    tile_rows = torch.where(used, (pstart[te] + counts[te] - tile_row).clamp(0, _ROWS), 0).to(torch.int32)
    tiles = torch.where(used, t, -1).to(torch.int32)
    experts = torch.arange(E, device=dev)
    segs_d = torch.stack([pstart, padded, experts * d], 1).to(torch.int32)
    segs_f = torch.stack([pstart, padded, experts * f], 1).to(torch.int32)
    return _Layout(tiles, tile_rows, segs_d, segs_f, src, pos), counts


class _Pending:
    """An exchange in flight: the forward's work handle, and the backward's (work, output, input kept alive)."""
    __slots__ = ("work", "bwork", "bout", "bin")


def _a2a(x: torch.Tensor, send, recv, group, async_op: bool):
    """(output, work) of all_to_all_single over x's rows: equal splits (send / recv None) or the given row
    counts per rank."""
    rows = x.shape[0] if recv is None else sum(recv)
    out = x.new_empty((rows,) + tuple(x.shape[1:]))
    work = dist.all_to_all_single(out, x.contiguous(), recv, send, group=group, async_op=async_op)
    return out, work


class _ExchangeStart(torch.autograd.Function):
    """Starts all_to_all_single of x's rows (``send`` / ``recv``: rows per rank, None for equal splits) without
    waiting; _ExchangeWait finishes it. Whatever runs between the two overlaps with the exchange, in the forward
    and, in reverse, in the backward: _ExchangeWait.backward starts the reverse exchange and this backward
    waits for it, and autograd runs the nodes created in between (higher sequence numbers) first."""

    @staticmethod
    def forward(ctx, x, group, send, recv, pend: _Pending):
        ctx.pend = pend
        out, pend.work = _a2a(x, send, recv, group, True)
        return out

    @staticmethod
    def backward(ctx, g):
        pend = ctx.pend
        pend.bwork.wait()
        out = pend.bout
        pend.bwork = pend.bout = pend.bin = None
        return out, None, None, None, None


class _ExchangeWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, group, send, recv, pend: _Pending):
        pend.work.wait()
        pend.work = None
        ctx.group, ctx.send, ctx.recv, ctx.pend = group, send, recv, pend
        return out.view_as(out)

    @staticmethod
    def backward(ctx, g):
        pend = ctx.pend
        pend.bin = g.contiguous()
        pend.bout, pend.bwork = _a2a(pend.bin, ctx.recv, ctx.send, ctx.group, True)
        return g, None, None, None, None  # a placeholder: _ExchangeStart.backward returns the exchanged rows


def _exchange(x: torch.Tensor, group, send=None, recv=None, overlap=None):
    """(rows exchanged over ``group``, overlap()'s result): ``overlap`` runs while the rows travel (and its
    backward while their gradients travel back)."""
    pend = _Pending()
    out = _ExchangeStart.apply(x, group, send, recv, pend)
    res = overlap() if overlap is not None else None
    return _ExchangeWait.apply(out, group, send, recv, pend), res


def _combine(ys: torch.Tensor, pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[t] = sum_j w[t, j] ys[pos[t, j]]: a gather and a k-term weighted sum (deterministic; the gather's
    backward writes every row once, except the always-zero row dropped slots point to, which gets zeros)."""
    T, k = pos.shape
    g = ys.index_select(0, pos.reshape(-1)).view(T, k, ys.shape[1])
    return torch.bmm(w.to(g.dtype).unsqueeze(1), g).squeeze(1)


def _names(activation: str, d: int, f: int):
    return (("gate", f, d), ("up", f, d), ("down", d, f)) if activation == "swiglu" else \
        (("fc1", f, d), ("fc2", d, f))


class MoE(nn.Module):
    """Top-k mixture-of-experts MLP (see the module docstring).

    d, d_ff: model and expert hidden sizes (multiples of 128 for NVFP4 experts); activation: "swiglu",
    "gelu_tanh" or "gelu"; bias: biases on every expert projection.

    capacity_factor (None: dropless): each expert takes at most C = ceil(capacity_factor * T k / E) of a
    forward's T k (token, choice) slots, in token order (Switch Transformer); the other slots are dropped (they
    contribute nothing to the token's output).
    shared_d_ff (0: none): a shared expert, a dense MLP of that hidden size (the same activation) that every
    token goes through with weight 1, its output added to the routed experts' (n shared experts of hidden h
    are one of hidden n h). Parameters ``shared_w_*`` [1, ., .] (and ``shared_b_*``).
    ep_group (None: every expert on this process): expert parallelism over a torch.distributed process group
    of W ranks: this rank holds experts [r E / W, (r + 1) E / W) (the stacked parameters have E / W rows), and
    the routed slots travel to their expert's rank and back by two all-to-alls, one per direction. With
    ``capacity_factor`` the exchanges have a fixed size (C slots per expert and source rank) and nothing is read
    on the host; dropless, the row counts are exchanged first and read on the host once per forward (as in
    Megatron-LM's dropless dispatcher), so such a step cannot be captured in a CUDA graph. The shared expert
    runs while the expert outputs travel back (``ep_overlap = False``: after them). The router and the shared
    expert are replicated; gradients of the local experts already include every rank's tokens.

    After forward, ``aux_loss`` holds the Switch Transformer load-balancing loss E * sum_e f_e P_e (f_e:
    fraction of routed slots, P_e: mean router probability of expert e; before dropping; this rank's tokens),
    1.0 when balanced; add ``coef * moe.aux_loss`` to the training loss. ``last_counts``: slots routed to each
    expert (before dropping); ``last_dropped``: slots dropped. ``bf16_impl``: how the high-precision experts
    run, "auto" (grouped Triton GEMMs on CUDA when Triton is installed, else the loop), "grouped" or "loop" (one
    expert after another, reading the expert sizes on the host)."""

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int = 2, activation: str = "swiglu",
                 normalize_weights: bool = True, bias: bool = False, capacity_factor: float | None = None,
                 shared_d_ff: int = 0, ep_group=None, device=None, dtype=None):
        super().__init__()
        if activation not in ("swiglu", "gelu_tanh", "gelu"):
            raise ValueError(f"activation must be swiglu, gelu_tanh or gelu, got {activation!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")
        if capacity_factor is not None and not capacity_factor > 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
        if shared_d_ff < 0:
            raise ValueError(f"shared_d_ff must be >= 0, got {shared_d_ff}")
        self.ep_group, self.ep_size, self.ep_rank = ep_group, 1, 0
        if ep_group is not None:
            self.ep_size, self.ep_rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
            if n_experts % self.ep_size:
                raise ValueError(f"{n_experts} experts do not split over {self.ep_size} ranks")
        self.d, self.d_ff, self.n_experts, self.top_k = d, d_ff, n_experts, top_k
        self.activation, self.normalize_weights = activation, normalize_weights
        self.capacity_factor, self.shared_d_ff = capacity_factor, shared_d_ff
        self.n_local = n_experts // self.ep_size
        self.router = nn.Linear(d, n_experts, bias=False, device=device, dtype=dtype)
        kw = dict(device=device, dtype=dtype)
        groups = [("", self.n_local, d_ff)] + ([("shared_", 1, shared_d_ff)] if shared_d_ff else [])
        for prefix, n_e, f in groups:
            for name, n, k in _names(activation, d, f):
                self.register_parameter(f"{prefix}w_{name}", nn.Parameter(torch.empty(n_e, n, k, **kw)))
                self.register_parameter(f"{prefix}b_{name}", nn.Parameter(torch.empty(n_e, n, **kw)) if bias else None)
        self._names = tuple("w_" + n for n, _, _ in _names(activation, d, d_ff))
        self._bias_names = tuple("b_" + n for n, _, _ in _names(activation, d, d_ff))
        self.has_bias = bias
        self.reset_parameters()
        self.nvfp4 = False
        self.bf16_impl = "auto"
        self.ep_overlap = True
        self.wgrad_rht = True
        self.fuse_wgrad_accumulation = False
        self.cache_weight_quantization = False
        self._caches = tuple(WeightQuantCache() for _ in self._names)
        self._shared_caches = tuple(WeightQuantCache() for _ in self._names)
        self.aux_loss: torch.Tensor | None = None
        self._counts: torch.Tensor | None = None
        self._dropped: torch.Tensor | None = None

    def reset_parameters(self) -> None:
        """nn.Linear's default initialization for every expert matrix and bias (uniform, bound 1/sqrt(fan_in))."""
        with torch.no_grad():
            for w, b in zip(self.experts_weights() + self.shared_weights(),
                            self.experts_biases() + self.shared_biases()):
                bound = 1 / math.sqrt(w.shape[-1])
                w.uniform_(-bound, bound)
                if b is not None:
                    b.uniform_(-bound, bound)

    def to_nvfp4(self, *, wgrad_rht: bool = True, fuse_wgrad_accumulation: bool = False,
                 cache_weight_quantization: bool = False, backend: str = "cuda") -> "MoE":
        """Run the experts (and the shared expert) in NVFP4 (``convert_linears`` calls this with its keyword
        arguments)."""
        if self.d % ALIGN or self.d_ff % ALIGN or self.shared_d_ff % ALIGN:
            raise ValueError(f"NVFP4 experts need d, d_ff and shared_d_ff divisible by {ALIGN}, got {self.d}, "
                             f"{self.d_ff}, {self.shared_d_ff}")
        if backend != "cuda":
            raise ValueError("NVFP4 experts need the CUDA backend")
        self.nvfp4 = True
        self.wgrad_rht, self.fuse_wgrad_accumulation = wgrad_rht, fuse_wgrad_accumulation
        self.cache_weight_quantization = cache_weight_quantization
        return self

    def experts_weights(self):
        """The routed experts' stacked weights ([E / W, ., .] under expert parallelism)."""
        return tuple(getattr(self, n) for n in self._names)

    def experts_biases(self):
        """The expert biases in the order of experts_weights (None each without ``bias``)."""
        return tuple(getattr(self, n) for n in self._bias_names)

    def shared_weights(self):
        """The shared expert's weights [1, ., .] (empty without ``shared_d_ff``)."""
        return tuple(getattr(self, "shared_" + n) for n in self._names) if self.shared_d_ff else ()

    def shared_biases(self):
        return tuple(getattr(self, "shared_" + n) for n in self._bias_names) if self.shared_d_ff else ()

    def __getstate__(self):
        """Copies and pickles leave out the last forward's outputs (``aux_loss`` is a non-leaf tensor, which
        deepcopy refuses; none belongs to the module's state) and the process group (not copyable)."""
        state = self.__dict__.copy()
        state["aux_loss"] = state["_counts"] = state["_dropped"] = None
        state["ep_group"] = None
        return state

    @property
    def last_counts(self) -> list[int] | None:
        """Slots routed to each expert in the last forward, before dropping (read from the GPU: synchronizes)."""
        return None if self._counts is None else self._counts.tolist()

    @property
    def last_dropped(self) -> int | None:
        """Slots dropped for capacity in the last forward (0 when dropless; synchronizes)."""
        return None if self._dropped is None else int(self._dropped)

    def route(self, x2d: torch.Tensor):
        """(weights [T, k] fp32, experts [T, k] int64, router probabilities [T, E] fp32)."""
        probs = torch.softmax(self.router(x2d).float(), dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        if self.normalize_weights and self.top_k > 1:
            w = w / w.sum(-1, keepdim=True)
        return w, idx, probs

    def capacity(self, tokens: int) -> int | None:
        """Slots each expert takes per forward of ``tokens`` tokens (per source rank under expert
        parallelism); None when dropless."""
        if self.capacity_factor is None:
            return None
        return max(1, math.ceil(self.capacity_factor * tokens * self.top_k / self.n_experts))

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

    # --- the experts on a buffer of rows ---------------------------------------------------------------

    def _mlp_loop(self, xs: torch.Tensor, counts: list[int], ws, bs) -> torch.Tensor:
        """Reference path: each expert's slice of the sorted rows, one after another."""
        E = ws[0].shape[0]
        ws = [w.unbind(0) for w in ws]  # unbind: one stacked gradient, not E zero-fills
        bs = [(b.unbind(0) if b is not None else (None,) * E) for b in bs]
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
        return torch.cat(outs, 0) if outs else xs[:0]

    def _mlp_nvfp4(self, x, lay: _Layout, ws, bs, caches):
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        xs = _Dispatch.apply(x.to(BF16), lay.src, lay.pos)  # bf16 first: the quantizer reads bf16 (as NVFP4Linear)
        grad = torch.is_grad_enabled()
        need_w = grad and any(w.requires_grad for w in ws)
        need_b = grad and any(b is not None and b.requires_grad for b in bs)
        need_dx = grad and xs.requires_grad
        caches = caches if self.cache_weight_quantization else (None,) * len(ws)
        if self.activation == "swiglu":
            return _GroupedSwiGLU.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                        (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt))
        approx = "tanh" if self.activation == "gelu_tanh" else "none"
        return _GroupedGelu.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                  (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt, approx))

    def _mlp_bf16_grouped(self, x, lay: _Layout, ws, bs):
        """The high-precision experts on the padded buffer with grouped Triton GEMMs (autocast honoured as
        F.linear would: inputs, weights and biases in the autocast dtype)."""
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        xs = _Dispatch.apply(x, lay.src, lay.pos)
        with torch.autocast("cuda", enabled=False):
            xs = xs.to(dt)
            ws = [w.to(dt) for w in ws]
            bs = [None if b is None else b.to(dt) for b in bs]
            lin = lambda a, i, segs: GroupedLinear.apply(a, ws[i], bs[i], lay.tiles, lay.tile_rows, segs)  # noqa: E731
            if self.activation == "swiglu":
                a = F.silu(lin(xs, 0, lay.segs_f)) * lin(xs, 1, lay.segs_f)
                return lin(a, 2, lay.segs_d)
            approx = "tanh" if self.activation == "gelu_tanh" else "none"
            return lin(F.gelu(lin(xs, 0, lay.segs_f), approximate=approx), 1, lay.segs_d)

    def _mlp(self, x, slot_expert, slot_src, k: int, n_experts: int, f: int, max_rows: int, drops: bool,
             ws, bs, caches):
        """(rows [?, d], pos [S / k, k]): slot i of x's row slot_src[i] through expert slot_expert[i] (-1:
        dropped, a zero row); row pos[s // k, s % k] of the result is slot s's output."""
        nvfp4 = self.nvfp4 and not ops.high_precision() and x.is_cuda
        if nvfp4 or self._bf16_grouped(x):
            lay, _ = _plan(slot_expert, slot_src, x.shape[0], n_experts, k, self.d, f, max_rows, drops)
            ys = self._mlp_nvfp4(x, lay, ws, bs, caches) if nvfp4 else self._mlp_bf16_grouped(x, lay, ws, bs)
            return ys, lay.pos
        # the loop splits the sorted rows: one host synchronization for the expert sizes
        key = torch.where(slot_expert >= 0, slot_expert, n_experts)
        order = torch.argsort(key, stable=True)
        S = key.numel()
        pos = torch.empty_like(order).index_copy_(0, order, torch.arange(S, device=x.device)).view(-1, k)
        counts = torch.zeros(n_experts + 1, device=x.device, dtype=torch.long).scatter_add_(
            0, key, torch.ones_like(key)).tolist()
        src = slot_src.index_select(0, order)
        ys = self._mlp_loop(_Dispatch.apply(x, src, pos)[:S - counts[-1]], counts[:-1], ws, bs)
        return torch.cat([ys, ys.new_zeros(counts[-1], self.d)]), pos  # dropped slots: zero rows

    # --- forward ---------------------------------------------------------------------------------------

    def _shared(self, x2d):
        T = x2d.shape[0]
        zeros = torch.zeros(T, device=x2d.device, dtype=torch.long)
        ys, pos = self._mlp(x2d, zeros, torch.arange(T, device=x2d.device), 1, 1, self.shared_d_ff, T, False,
                            self.shared_weights(), self.shared_biases(), self._shared_caches)
        return ys.index_select(0, pos.view(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2d = x.reshape(-1, self.d)
        T, k, E = x2d.shape[0], self.top_k, self.n_experts
        dev = x2d.device
        w, idx, probs = self.route(x2d)
        flat = idx.reshape(-1)
        S = T * k
        counts = torch.zeros(E, device=dev, dtype=torch.long).scatter_add_(0, flat, torch.ones_like(flat))
        self._counts = counts
        frac = counts.float() / max(1, S)
        self.aux_loss = E * (frac * probs.mean(0)).sum()
        C = self.capacity(T)
        slot_expert, rank = flat, None
        if C is not None:  # keep each expert's first C slots (in token order)
            order = torch.argsort(flat, stable=True)
            cstart = torch.cumsum(counts, 0) - counts
            rank_sorted = torch.arange(S, device=dev) - cstart.index_select(0, flat.index_select(0, order))
            rank = torch.empty_like(rank_sorted).index_copy_(0, order, rank_sorted)
            keep = rank < C
            slot_expert = torch.where(keep, flat, -1)
            w = w * keep.view(T, k)
            self._dropped = (~keep).sum()
        else:
            self._dropped = torch.zeros((), device=dev, dtype=torch.long)
        slot_src = torch.arange(S, device=dev) // k
        if self.ep_group is not None:
            y, sh = self._forward_ep(x2d, slot_expert, slot_src, rank, counts, w, C)
        else:
            max_rows = S if C is None else min(S, E * C)
            ys, pos = self._mlp(x2d, slot_expert, slot_src, k, E, self.d_ff, max_rows, C is not None,
                                self.experts_weights(), self.experts_biases(), self._caches)
            y = _combine(ys, pos, w)
            sh = self._shared(x2d) if self.shared_d_ff else None
        if sh is not None:
            y = y + sh.to(y.dtype)
        return y.reshape(*lead, self.d)

    def _forward_ep(self, x2d, slot_expert, slot_src, rank, counts, w, C: int | None):
        """Expert parallelism: (routed output, shared expert's output or None). The slots go to their expert's
        rank, the local experts run on the received rows, the outputs come back and are combined. The shared
        expert runs while the outputs travel back (and its backward while their gradients go out)."""
        T, k, E, W, El = x2d.shape[0], self.top_k, self.n_experts, self.ep_size, self.n_local
        dev, S = x2d.device, T * k
        nvfp4 = self.nvfp4 and not ops.high_precision() and x2d.is_cuda
        xin = x2d.to(BF16) if nvfp4 else x2d  # NVFP4 experts read bf16: half the bytes on the wire
        shared = (lambda: self._shared(x2d)) if self.shared_d_ff else None
        overlap = shared if self.ep_overlap else None
        group, ws, bs = self.ep_group, self.experts_weights(), self.experts_biases()
        if C is None:
            # dropless: rows sorted by expert (so by rank), variable splits; one host read of the counts
            order = torch.argsort(slot_expert, stable=True)
            pos = torch.empty_like(order).index_copy_(0, order, torch.arange(S, device=dev)).view(T, k)
            xs = _Dispatch.apply(xin, slot_src.index_select(0, order), pos)
            recv_counts = torch.empty_like(counts)
            dist.all_to_all_single(recv_counts, counts, group=group)  # [W, E/W]: per source rank
            host = torch.cat([counts, recv_counts]).tolist()
            send = [sum(host[r * El:(r + 1) * El]) for r in range(W)]
            recv = [sum(host[E + r * El:E + (r + 1) * El]) for r in range(W)]
            R = sum(recv)
            xr, _ = _exchange(xs, group, send, recv)  # [R, d]: by source rank, then by local expert
            e_loc = torch.arange(El, device=dev).repeat(W).repeat_interleave(recv_counts, output_size=R)
            ys, p = self._mlp(xr, e_loc, torch.arange(R, device=dev), 1, El, self.d_ff, R, False, ws, bs,
                              self._caches)
            yb, sh = _exchange(ys.index_select(0, p.view(-1)), group, recv, send, overlap)
            return _combine(yb, pos, w), (sh if overlap is not None or shared is None else shared())
        # capacity: fixed-size buffer [W, E/W, C, d] (expert e's kept slots in rows e C + rank)
        kept = slot_expert >= 0
        send_pos = torch.where(kept, slot_expert * C + rank, E * C)  # E C: a zero row, not sent
        send_src = torch.full((E * C + 1,), T, device=dev, dtype=torch.long).index_copy_(
            0, send_pos, torch.where(kept, slot_src, T))
        xs = _Dispatch.apply(xin, send_src, send_pos.view(T, k))[:E * C]
        kept_counts = torch.minimum(counts, torch.full_like(counts, C))
        recv_counts = torch.empty_like(kept_counts)
        dist.all_to_all_single(recv_counts, kept_counts, group=group)  # [W, E/W]: per source rank
        xr, _ = _exchange(xs, group)  # [W, E/W, C, d] from every source rank
        n = W * El * C
        cpos = torch.arange(n, device=dev)
        e_loc = (cpos // C) % El
        valid = (cpos % C) < recv_counts.view(-1).repeat_interleave(C)
        ys, pos = self._mlp(xr, torch.where(valid, e_loc, -1), cpos, 1, El, self.d_ff, n, True, ws, bs,
                            self._caches)
        # back to the source: rows e C + c (invalid rows are zero)
        yb, sh = _exchange(ys.index_select(0, pos.view(-1)), group, overlap=overlap)
        y = _combine(torch.cat([yb, yb.new_zeros(1, self.d)]), send_pos.view(T, k), w)
        return y, (sh if overlap is not None or shared is None else shared())

    def extra_repr(self) -> str:
        r = f"d={self.d}, d_ff={self.d_ff}, n_experts={self.n_experts}, top_k={self.top_k}, " \
            f"activation={self.activation}"
        if self.has_bias:
            r += ", bias=True"
        if self.capacity_factor is not None:
            r += f", capacity_factor={self.capacity_factor}"
        if self.shared_d_ff:
            r += f", shared_d_ff={self.shared_d_ff}"
        if self.ep_group is not None:
            r += f", expert_parallel={self.ep_size} (local experts {self.n_local})"
        return r + (", nvfp4=True" if self.nvfp4 else "")


__all__ = ["MoE"]
