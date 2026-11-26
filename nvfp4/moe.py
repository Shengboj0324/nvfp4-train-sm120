"""Top-k mixture-of-experts MLP whose experts train in NVFP4 with grouped GEMMs.

    from nvfp4.moe import MoE
    moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu").cuda()
    nvfp4.convert_linears(moe)               # experts -> NVFP4 (moe.to_nvfp4(...)); the router stays BF16
    y = moe(x)                               # x [..., d]; moe.aux_loss: Switch load-balancing loss

Routing: softmax over the router logits (fp32), top-k experts per token, weights renormalized over the k
choices. The (token, expert) pairs are sorted by expert (stable). Dropless by default; with
``capacity_factor`` each expert keeps its first C = ceil(capacity_factor T k / E) slots in token order. Options:
a shared expert (``shared_d_ff``: a dense MLP every token goes through, added to the routed output) and
expert parallelism (``ep_group``: each rank holds E / W experts; slots travel by fixed-size all-to-alls with a
capacity, or tokens once per rank (or node, ``ep_node_size``) by variable-size ones that read the row counts on
the host, optionally chunked to overlap the experts and with NVFP4 messages).
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

import contextlib
import math
import os
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nvfp4 import _bf16_grouped, ops
from nvfp4 import _library as _lib
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


_FUSED_COMBINE = os.environ.get("NVFP4_MOE_FUSED_COMBINE", "1") != "0"  # 0: the index_select + bmm path (ablations)


def _fused(t: torch.Tensor) -> bool:
    """The gather / combine kernels apply: a CUDA tensor on an sm_120 GPU with the extension built (the device
    check is cached by the first eager call; under torch.compile an unchecked device takes the PyTorch path)."""
    if not t.is_cuda or _lib._C is None or not _FUSED_COMBINE:
        return False
    idx = t.device.index if t.device.index is not None else torch.cuda.current_device()
    ok = _lib._device_ok.get(idx)
    if ok is None and not torch.compiler.is_compiling():
        ok = _lib._device_ok[idx] = torch.cuda.get_device_capability(idx) == _lib._SUPPORTED_CAPABILITY
    return bool(ok)


class _TileScope(torch.autograd.Function):
    """Identity whose backward sets ``ops.set_dynamic_tiles(on)``: placed on an expert-parallel layer's output
    (on) and input (the previous setting), it brackets the layer's backward, whose GEMMs then claim their
    tiles at run time while the gradient exchanges' NCCL kernels hold SMs (autograd runs a layer's nodes
    between its output's and its input's)."""

    @staticmethod
    def forward(ctx, x, on: bool):
        ctx.on = on
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        ops.set_dynamic_tiles(ctx.on)
        return g, None


class _GatherCombine(torch.autograd.Function):
    """out[r] = sum_j w[r, j] rows[pos[r, j]] (w None: weight 1; pos outside [0, len(rows)): nothing), in fp32
    with one rounding to rows' dtype, by one kernel that reads each row once (csrc/moe_combine.cu). Backward by
    gathers, deterministic: each row's gradient comes from the one (r, j) that reads it (every row is read by at
    most one choice with a nonzero weight, as in this layer), the weights' from per-choice dot products."""

    @staticmethod
    def forward(ctx, rows, pos, w):
        w32 = None if w is None else w.float().contiguous()
        out = _lib.combine_fwd(rows, pos.contiguous(), w32)
        ctx.save_for_backward(rows if w is not None and ctx.needs_input_grad[2] else None, pos, w32)
        ctx.m, ctx.w_dtype = rows.shape[0], None if w is None else w.dtype
        return out

    @staticmethod
    def backward(ctx, g):
        rows, pos, w32 = ctx.saved_tensors
        R, k = pos.shape
        drows = dw = None
        g = g.contiguous()
        if ctx.needs_input_grad[0]:
            flat = pos.reshape(-1)
            valid = (flat >= 0) & (flat < ctx.m)
            # the slot reading each row (integer max: deterministic; duplicates only among zero-weight slots)
            inv = torch.full((ctx.m + 1,), -1, device=g.device, dtype=torch.long).scatter_reduce_(
                0, torch.where(valid, flat, ctx.m), torch.arange(R * k, device=g.device), "amax")[:ctx.m]
            drows = _lib.combine_bwd_rows(g, inv.contiguous(), w32, k)
        if rows is not None:
            dw = _lib.combine_bwd_w(g, rows if rows.stride(1) == 1 else rows.contiguous(), pos).to(ctx.w_dtype)
        return drows, None, dw


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
        if _fused(g):
            return _lib.combine_fwd(g.contiguous(), pos.contiguous(), None), None, None
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
    def forward(ctx, xs, wg, wu, wd, bg, bu, bd, tiles, trows, segs_d, segs_f, caches, cfg, x_amax):
        rht, need_dx, need_w, need_b, fuse, dt = cfg
        E, f, d = wg.shape
        need_a = need_dx or need_w or need_b  # the gradient flows back through down and the SwiGLU
        # [dG row SR, dG^T SR, RHT of x^T, dOut row SR, dOut^T SR, RHT of a^T, dU row SR, dU^T SR]
        seeds = draw_seeds(8, xs.device) if need_a else None
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN, amax_floor=x_amax)
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
        grads = [None] * 14
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
    def forward(ctx, xs, w1, w2, b1, b2, tiles, trows, segs_d, segs_f, caches, cfg, x_amax):
        rht, need_dx, need_w, need_b, fuse, dt, approximate = cfg
        E, f, d = w1.shape
        need_h = need_dx or need_w or need_b
        seeds = draw_seeds(6, xs.device) if need_h else None  # [dH row SR, dH^T SR, RHT x^T, dOut row, col, RHT]
        xq, xtq = ops.quantize_dual(xs.to(BF16), col=need_w, rht_seed=seeds[2] if need_w and rht else None,
                                    col_pad=ALIGN, amax_floor=x_amax)
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
        grads = [None] * 12
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


_SERIAL_STREAMS: dict = {}


def _serial_stream(dev: torch.device) -> torch.cuda.Stream:
    """The stream every serialized exchange on ``dev`` is issued from (one per device)."""
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _SERIAL_STREAMS:
        _SERIAL_STREAMS[idx] = torch.cuda.Stream(device=idx)
    return _SERIAL_STREAMS[idx]


def _a2a(x: torch.Tensor, send, recv, group, async_op: bool, serial: bool = False):
    """(output, work) of all_to_all_single over x's rows: equal splits (send / recv None) or the given row
    counts per rank. ``serial`` (CUDA): issued from one stream that waits for each collective's end, so
    collectives of different process groups never run at the same time (each still overlaps the compute
    stream, which waits only through the returned work)."""
    rows = x.shape[0] if recv is None else sum(recv)
    out = x.new_empty((rows,) + tuple(x.shape[1:]))
    x = x.contiguous()
    if not (serial and x.is_cuda):
        return out, dist.all_to_all_single(out, x, recv, send, group=group, async_op=async_op)
    s = _serial_stream(x.device)
    s.wait_stream(torch.cuda.current_stream(x.device))  # the payload is ready
    with torch.cuda.stream(s):
        work = dist.all_to_all_single(out, x, recv, send, group=group, async_op=True)
        work.wait()  # the next serialized collective starts after this one ends
    if not async_op:
        work.wait()
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


# --- variable-size expert-parallel exchanges: each token once per rank (or node), payloads in bytes --------------

def _bytes(t: torch.Tensor) -> torch.Tensor:
    """A 2D tensor's rows as bytes, uint8 [R, C * itemsize]."""
    return t.contiguous().view(torch.uint8)


@contextlib.contextmanager
def _plain_rows():
    """Under MS-EDEN on dgrad the dual quantizer rotates its stochastically rounded row output; a message holds
    the plain rows."""
    prev = ops.ms_eden_dgrad()
    if prev:
        ops.set_ms_eden(True, dgrad=False)
    try:
        yield
    finally:
        if prev:
            ops.set_ms_eden(True, dgrad=True)


def _encode(x: torch.Tensor, wire: str, amax=None, sr_seed=None) -> torch.Tensor:
    """x's rows as bytes: raw, or ``wire="nvfp4"``: NVFP4 codes and block scales with the tensor scale of
    ``amax`` (round-to-nearest, or stochastic with ``sr_seed``), then the tensor decode scale and the amax as
    fp32 in every row (9 d / 16 + 8 bytes for d values, 3.6x fewer than bf16 at d = 1024)."""
    if wire == "fp8":
        return _lib.mxfp8_encode(x, sr_seed)
    if wire != "nvfp4":
        return _bytes(x)
    with _plain_rows():
        q, _ = ops.quantize_dual(x.to(BF16).contiguous(), row=True, col=False, row_sr_seed=sr_seed, amax_floor=amax)
    tail = torch.cat([q.gdec.reshape(1), amax.reshape(1).float()]).view(1, 2).expand(x.shape[0], 2)
    return torch.cat([q.data, q.scales, _bytes(tail)], 1)


def _decode(buf: torch.Tensor, o: int, d: int, dtype, wire: str):
    """(rows [R, d] in ``dtype`` from the bytes buf[:, o:], their tensor amax (NVFP4 rows: one fp32 element,
    None without rows; raw rows: None), the offset after them)."""
    if wire == "fp8":
        return _lib.mxfp8_decode(buf[:, o:], d, dtype), None, o + d + d // 32
    if wire != "nvfp4":
        n = d * dtype.itemsize
        return buf[:, o:o + n].view(dtype), None, o + n
    t = o + d // 2 + d // 16
    tail = buf[:, t:t + 8].view(torch.float32)  # per row: (gdec, amax)
    x = ops.dequantize(buf[:, o:o + d // 2], buf[:, o + d // 2:t], tail[:, 0], dtype)
    return x, (tail[:1, 1].contiguous() if buf.shape[0] else None), t + 8


def _global_max(t: torch.Tensor, group, serial: bool = False) -> torch.Tensor:
    """The max over ``group``'s ranks of a one-element tensor (by all-to-all: gloo on PyTorch 2.11 Windows
    crashes reducing CUDA tensors)."""
    W = dist.get_world_size(group)
    out, _ = _a2a(t.reshape(1).float().expand(W), None, None, group, False, serial)
    return out.max().reshape(1)


class _Box:
    """A tensor handed from one autograd node of a forward to another (an NVFP4 gradient message's amax)."""
    __slots__ = ("amax",)

    def __init__(self):
        self.amax = None


class _Xchg:
    """One all-to-all of a row payload over ``group`` (``send`` / ``recv``: rows per rank) and, in the
    backward, of the payload's gradients the other way.

    Payload: x [R, d] (``dtype``; ``wire="nvfp4"``: NVFP4 with the tensor amax ``amax``), optionally routing
    weights w [R, k] (fp32) and ids [R, k] (int32), packed into one byte message per rank. Gradients: x's
    (``wire_grad="nvfp4"``: stochastically rounded NVFP4 with the tensor amax ``grad_amax``, a process group to
    take the max over or a _Box filled by an earlier backward node) and w's (fp32). ``box``: receives the amax
    of NVFP4 gradients arriving here. start() / finish(): whatever runs between them overlaps the transfer, and
    in the backward runs between the reverse transfer's start (finish's backward) and its end."""

    def __init__(self, group, send, recv, d: int, dtype, k: int = 0, w: bool = False, ids: bool = False,
                 wire: str = "bf16", amax=None, wire_grad: str = "bf16", grad_amax=None, box: _Box | None = None,
                 serial: bool = False):
        self.group, self.send, self.recv, self.d, self.dtype, self.k = group, send, recv, d, dtype, k
        self.serial = serial
        self.w, self.ids, self.wire, self.amax = w, ids, wire, amax
        self.wire_grad, self.grad_amax, self.box = wire_grad, grad_amax, box
        self.work = self.out = self.bwork = self.bout = None

    def start(self, x, w=None, ids=None):
        return _XStart.apply(self, x, w, ids)

    def finish(self, handle):
        """(x, w, ids) received (None where not in the payload)."""
        outs = list(_XWait.apply(handle, self))
        x = outs.pop(0)
        w = outs.pop(0) if self.w else None
        return x, w, (outs.pop(0) if self.ids else None)


class _XStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xc: _Xchg, x, w, ids):
        ctx.xc = xc
        parts = [_encode(x, xc.wire, xc.amax)]
        if xc.w:
            parts.append(_bytes(w.float()))
        if xc.ids:
            parts.append(_bytes(ids.to(torch.int32)))
        buf = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        xc.out, xc.work = _a2a(buf, xc.send, xc.recv, xc.group, True, xc.serial)
        return x.new_zeros(())  # a handle: carries the autograd link to _XWait

    @staticmethod
    def backward(ctx, _):
        xc = ctx.xc
        xc.bwork.wait()
        buf, xc.bwork, xc.bout = xc.bout, None, None
        gx, amax, o = _decode(buf, 0, xc.d, xc.dtype, xc.wire_grad)
        if amax is not None and xc.box is not None:
            xc.box.amax = amax
        gw = buf[:, o:o + 4 * xc.k].view(torch.float32) if xc.w else None
        return None, gx, gw, None


class _XWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, handle, xc: _Xchg):
        ctx.xc = xc
        ctx.set_materialize_grads(False)
        xc.work.wait()
        buf, xc.work, xc.out = xc.out, None, None
        x, _, o = _decode(buf, 0, xc.d, xc.dtype, xc.wire)
        outs = [x]
        if xc.w:
            outs.append(buf[:, o:o + 4 * xc.k].view(torch.float32))
            o += 4 * xc.k
        if xc.ids:
            outs.append(buf[:, o:o + 4 * xc.k].view(torch.int32))
        ctx.rows, ctx.dev = buf.shape[0], buf.device
        return tuple(outs)

    @staticmethod
    def backward(ctx, gx, *rest):
        xc, R, dev = ctx.xc, ctx.rows, ctx.dev
        gx = torch.zeros(R, xc.d, device=dev, dtype=xc.dtype) if gx is None else gx.to(xc.dtype)
        amax = seed = None
        if xc.wire_grad == "fp8":
            seed = draw_seeds(1, dev)[0]  # stochastic rounding: unbiased gradients
        elif xc.wire_grad == "nvfp4":
            local = gx.abs().amax().float().reshape(1) if R else torch.zeros(1, device=dev)
            if isinstance(xc.grad_amax, _Box):
                amax = xc.grad_amax.amax if xc.grad_amax.amax is not None else local
            else:
                amax = _global_max(local, xc.grad_amax, xc.serial)
            seed = draw_seeds(1, dev)[0]
        parts = [_encode(gx, xc.wire_grad, amax, seed)]
        if xc.w:
            gw = rest[0] if rest and rest[0] is not None else torch.zeros(R, xc.k, device=dev)
            parts.append(_bytes(gw.float()))
        buf = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        xc.bout, xc.bwork = _a2a(buf, xc.recv, xc.send, xc.group, True, xc.serial)
        return torch.zeros((), device=dev), None


class _Route(NamedTuple):
    """Rows with k choices each, sent once per (row, member) of a group (_route)."""
    src: torch.Tensor   # int64 [n + 1]: the row each sent row copies (the last: R, a zero row that is not sent)
    pos: torch.Tensor   # int64 [R, k]: the sent row of each (row, choice), for the first choice per member; else n
    rows: torch.Tensor  # int64 [n]: src[:n]
    mask: torch.Tensor  # bool [n, k]: the choices of each sent row that its member takes
    ids: torch.Tensor   # int32 [n, k]: those choices' ids within the member, -1 elsewhere


def _first_choices(ids: torch.Tensor, per: int, members: int):
    """(member of each choice [R, k] (``members``: none), the choice is the row's first on its member)."""
    k = ids.shape[1]
    valid = ids >= 0
    mem = torch.where(valid, ids // per, members)
    earlier = torch.ones(k, k, dtype=torch.bool, device=ids.device).tril(-1)  # [j, j']: j' < j
    dup = ((mem.unsqueeze(2) == mem.unsqueeze(1)) & earlier).any(2)
    return mem, valid & ~dup


def _route(ids: torch.Tensor, members: int, per: int, n: int) -> _Route:
    """Routes rows whose k choices are ids [R, k] (-1: none) to ``members`` members holding ``per`` ids each
    (id i on member i // per): one sent row per (row, member) pair, grouped by member, in row order. ``n``: the
    number of pairs (known on the host). All on the GPU."""
    R, k = ids.shape
    dev = ids.device
    mem, first = _first_choices(ids, per, members)
    key = torch.where(first, mem * R + torch.arange(R, device=dev).unsqueeze(1), members * R)
    order = torch.argsort(key.reshape(-1), stable=True)
    sel = order[:n]
    rows = sel // k
    mask = mem.index_select(0, rows) == mem.reshape(-1).index_select(0, sel).unsqueeze(1)
    sub = torch.where(mask, ids.index_select(0, rows) % per, -1).to(torch.int32)
    inv = torch.empty_like(order).index_copy_(0, order, torch.arange(R * k, device=dev))
    pos = torch.where(first.reshape(-1), inv, n).view(R, k)
    return _Route(torch.cat([rows, rows.new_full((1,), R)]), pos, rows, mask, sub)


def _gather_sum(rows: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """out[r] = sum_j rows'[pos[r, j]], rows' = rows plus a zero row at index len(rows) (deterministic: each
    real row is read once)."""
    R, k = pos.shape
    if _fused(rows):
        return _GatherCombine.apply(rows, pos, None)
    rows = torch.cat([rows, rows.new_zeros(1, rows.shape[1])])
    g = rows.index_select(0, pos.reshape(-1)).view(R, k, rows.shape[1])
    return g.float().sum(1).to(rows.dtype) if k > 1 else g[:, 0]


_SUBGROUPS: dict = {}


def _ep_subgroups(group, L: int):
    """(node group, rail group) of this rank for nodes of L consecutive ranks of ``group``: the L ranks of its
    node, and the ranks with its position in every node. Created once per (group, L) (new_group is collective:
    every rank of the default group builds the same layers in the same order)."""
    ranks = dist.get_process_group_ranks(group)
    key = (tuple(ranks), L)
    if key not in _SUBGROUPS:
        N = len(ranks) // L
        nodes = [dist.new_group([ranks[m * L + l] for l in range(L)]) for m in range(N)]
        rails = [dist.new_group([ranks[m * L + l] for m in range(N)]) for l in range(L)]
        _SUBGROUPS[key] = (nodes, rails)
    nodes, rails = _SUBGROUPS[key]
    r = dist.get_rank(group)
    return nodes[r // L], rails[r % L]


def _combine(ys: torch.Tensor, pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """y[t] = sum_j w[t, j] ys[pos[t, j]]: a gather and a k-term weighted sum (deterministic; the gather's
    backward writes every row once, except the always-zero row dropped slots point to, which gets zeros)."""
    T, k = pos.shape
    if _fused(ys):  # weights rounded to the rows' dtype first, as the bmm below takes them
        return _GatherCombine.apply(ys, pos, w.to(ys.dtype))
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
    of W ranks: this rank holds experts [r E / W, (r + 1) E / W) (the stacked parameters have E / W rows). The
    router and the shared expert are replicated; gradients of the local experts already include every rank's
    tokens. Two ways to exchange (``ep_dispatch``, "auto": "fixed" with a capacity factor and none of the
    options below, else "variable"):
      * "fixed" (needs ``capacity_factor``): every slot travels to its expert's rank and back in fixed-size
        all-to-alls (C slots per expert and source rank); nothing is read on the host (CUDA-graph capturable).
      * "variable": each token travels once to every rank holding one of its experts (with those experts' ids
        and routing weights), which returns one row, the weighted sum of its experts' outputs (DeepEP's
        layout: top-k > 1 sends fewer rows than slots). The row counts go first, in one all-to-all read on the
        host once per forward (not capturable). Dropped slots (``capacity_factor``) are not sent.
        ``ep_node_size`` L (constructor; ranks r L .. r L + L - 1 form a node): tokens cross nodes once per
        node, to the rank with the sender's position in that node, which forwards them once per rank of its node
        and sums their rows (two-stage, for clusters whose links between nodes are slower). ``ep_chunks`` n:
        the tokens go in n chunks in a wavefront, the next chunk's rows travelling while this chunk's experts run
        (and the gradients likewise); ``ops.set_sm_margin`` leaves the transfers SMs. With NVFP4 experts,
        ``ep_wire = "nvfp4"`` sends the activations as NVFP4 with the group's tensor scale (3.6x fewer bytes;
        the receiver's re-quantization reproduces the codes, so the forward and the input gradient are
        unchanged, only the weight gradient's token-wise quantization sees the dequantized rows) and
        ``ep_wire_grad = "nvfp4"`` the gradients sent to the experts (stochastically rounded, unbiased).
        ``last_ep_rows``: rows this rank sent to other ranks in the last forward (across nodes, within its node).
        ``ep_wire_return = "fp8"``: the return direction (expert outputs, and the input gradients going back)
        as MXFP8 (e4m3 with a power-of-two scale per 32 values; gradients stochastically rounded), 1.94x fewer
        bytes than bf16 (CUDA and d % 128 == 0; else bf16). ``ep_serial`` ("auto": with chunks and nodes): every exchange
        issued from one side stream, one at a time, so rail and node collectives never run concurrently.
    The shared expert runs while the expert outputs travel back (``ep_overlap = False``: after them).
    ``ep_dynamic_tiles`` ("auto": with chunks or an overlapped shared expert): the layer's GEMMs, forward and
    backward, claim their tiles at run time (``ops.set_dynamic_tiles``) so that SMs held by NCCL do not
    delay them.

    After forward, ``aux_loss`` holds the Switch Transformer load-balancing loss E * sum_e f_e P_e (f_e:
    fraction of routed slots, P_e: mean router probability of expert e; before dropping; this rank's tokens),
    1.0 when balanced; add ``coef * moe.aux_loss`` to the training loss. ``last_counts``: slots routed to each
    expert (before dropping); ``last_dropped``: slots dropped. ``bf16_impl``: how the high-precision experts
    run, "auto" (grouped Triton GEMMs on CUDA when Triton is installed, else the loop), "grouped" or "loop" (one
    expert after another, reading the expert sizes on the host)."""

    def __init__(self, d: int, d_ff: int, n_experts: int, top_k: int = 2, activation: str = "swiglu",
                 normalize_weights: bool = True, bias: bool = False, capacity_factor: float | None = None,
                 shared_d_ff: int = 0, ep_group=None, ep_node_size: int = 1, device=None, dtype=None):
        super().__init__()
        if activation not in ("swiglu", "gelu_tanh", "gelu"):
            raise ValueError(f"activation must be swiglu, gelu_tanh or gelu, got {activation!r}")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in [1, n_experts], got {top_k}")
        if capacity_factor is not None and not capacity_factor > 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}")
        if shared_d_ff < 0:
            raise ValueError(f"shared_d_ff must be >= 0, got {shared_d_ff}")
        if ep_node_size < 1:
            raise ValueError(f"ep_node_size must be >= 1, got {ep_node_size}")
        self.ep_group, self.ep_size, self.ep_rank, self.ep_node_size = ep_group, 1, 0, ep_node_size
        self._node_group = self._rail_group = None
        if ep_group is not None:
            self.ep_size, self.ep_rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
            if n_experts % self.ep_size:
                raise ValueError(f"{n_experts} experts do not split over {self.ep_size} ranks")
            if self.ep_size % ep_node_size:
                raise ValueError(f"{self.ep_size} ranks do not split into nodes of {ep_node_size}")
            self._rail_group = ep_group
            if ep_node_size > 1:
                self._node_group, self._rail_group = _ep_subgroups(ep_group, ep_node_size)
        elif ep_node_size != 1:
            raise ValueError("ep_node_size needs ep_group")
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
        self.ep_dispatch = "auto"
        self.ep_chunks = 1
        self.ep_wire = "bf16"
        self.ep_wire_grad = "bf16"
        self.ep_wire_return = "bf16"
        self.ep_serial = "auto"
        self.ep_dynamic_tiles = "auto"
        self.wgrad_rht = True
        self.fuse_wgrad_accumulation = False
        self.cache_weight_quantization = False
        self._caches = tuple(WeightQuantCache() for _ in self._names)
        self._shared_caches = tuple(WeightQuantCache() for _ in self._names)
        self.aux_loss: torch.Tensor | None = None
        self.last_ep_rows: tuple[int, int] | None = None
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
        state["ep_group"] = state["_node_group"] = state["_rail_group"] = None
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

    def _mlp_nvfp4(self, x, lay: _Layout, ws, bs, caches, x_amax=None):
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        xs = _Dispatch.apply(x.to(BF16), lay.src, lay.pos)  # bf16 first: the quantizer reads bf16 (as NVFP4Linear)
        grad = torch.is_grad_enabled()
        need_w = grad and any(w.requires_grad for w in ws)
        need_b = grad and any(b is not None and b.requires_grad for b in bs)
        need_dx = grad and xs.requires_grad
        caches = caches if self.cache_weight_quantization else (None,) * len(ws)
        if self.activation == "swiglu":
            return _GroupedSwiGLU.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                        (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt),
                                        x_amax)
        approx = "tanh" if self.activation == "gelu_tanh" else "none"
        return _GroupedGelu.apply(xs, *ws, *bs, lay.tiles, lay.tile_rows, lay.segs_d, lay.segs_f, caches,
                                  (self.wgrad_rht, need_dx, need_w, need_b, self.fuse_wgrad_accumulation, dt, approx),
                                  x_amax)

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
             ws, bs, caches, x_amax=None):
        """(rows [?, d], pos [S / k, k]): slot i of x's row slot_src[i] through expert slot_expert[i] (-1:
        dropped, a zero row); row pos[s // k, s % k] of the result is slot s's output. ``x_amax``: a floor for
        the NVFP4 experts' input tensor scale (expert parallelism: the group's amax)."""
        nvfp4 = self.nvfp4 and not ops.high_precision() and x.is_cuda
        if nvfp4 or self._bf16_grouped(x):
            lay, _ = _plan(slot_expert, slot_src, x.shape[0], n_experts, k, self.d, f, max_rows, drops)
            ys = self._mlp_nvfp4(x, lay, ws, bs, caches, x_amax) if nvfp4 else \
                self._mlp_bf16_grouped(x, lay, ws, bs)
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
        dyn = self._dynamic_tiles()
        if not dyn:
            return self._forward(x)
        prev = ops.dynamic_tiles()
        ops.set_dynamic_tiles(True)
        # the backward is bracketed only when the input takes a gradient: its node restores the setting last
        scoped = x.requires_grad and torch.is_grad_enabled()
        try:
            y = self._forward(_TileScope.apply(x, prev) if scoped else x)
        finally:
            ops.set_dynamic_tiles(prev)
        return _TileScope.apply(y, True) if scoped else y

    def _dynamic_tiles(self) -> bool:
        """Whether this forward (and its backward) runs its GEMMs with dynamic tile claiming
        (``ops.set_dynamic_tiles``): ``ep_dynamic_tiles`` "auto" turns it on while expert-parallel exchanges
        overlap GEMMs (``ep_chunks`` > 1, or the shared expert during the exchanges), where NCCL's kernels hold
        SMs that the fixed tile assignment would wait for (the GEMMs ran 1.7x slower; RESULTS.md §8.6)."""
        v = self.ep_dynamic_tiles
        if v not in ("auto", True, False):
            raise ValueError(f"ep_dynamic_tiles must be 'auto', True or False, got {v!r}")
        if self.ep_group is None or not torch.cuda.is_available() or not ops.available():
            return False
        if v != "auto":
            return bool(v)
        return self.ep_chunks > 1 or bool(self.shared_d_ff and self.ep_overlap)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
        if self.ep_group is not None and self._ep_mode(C) == "variable":
            y, sh = self._forward_ep_variable(x2d, slot_expert.view(T, k), w)
        elif self.ep_group is not None:
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

    def _ep_mode(self, C: int | None) -> str:
        """"fixed" (capacity, fixed-size per-slot exchanges, nothing read on the host) or "variable"."""
        if self.ep_dispatch not in ("auto", "fixed", "variable"):
            raise ValueError(f"ep_dispatch must be auto, fixed or variable, got {self.ep_dispatch!r}")
        for name in ("ep_wire", "ep_wire_grad"):
            if getattr(self, name) not in ("bf16", "nvfp4"):
                raise ValueError(f"{name} must be 'bf16' or 'nvfp4', got {getattr(self, name)!r}")
        if self.ep_wire_return not in ("bf16", "fp8"):
            raise ValueError(f"ep_wire_return must be 'bf16' or 'fp8', got {self.ep_wire_return!r}")
        if self.ep_serial not in ("auto", True, False):
            raise ValueError(f"ep_serial must be 'auto', True or False, got {self.ep_serial!r}")
        if not (isinstance(self.ep_chunks, int) and self.ep_chunks >= 1):
            raise ValueError(f"ep_chunks must be a positive int, got {self.ep_chunks!r}")
        extra = self.ep_chunks > 1 or self.ep_node_size > 1 or self.ep_wire != "bf16" or             self.ep_wire_grad != "bf16" or self.ep_wire_return != "bf16"
        if self.ep_dispatch == "variable" or (self.ep_dispatch == "auto" and (C is None or extra)):
            return "variable"
        if C is None:
            raise ValueError("ep_dispatch='fixed' needs a capacity_factor")
        if extra:
            raise ValueError("ep_chunks, ep_node_size and the NVFP4 wire need ep_dispatch='variable'")
        return "fixed"

    def _forward_ep_variable(self, x2d, idx, w):
        """Expert parallelism with variable-size exchanges: (routed output, shared expert's output or None).
        idx [T, k]: each token's experts (-1: dropped for capacity); w [T, k]: their weights.

        Every token goes once to each rank holding one of its experts (with the ids and weights of the experts
        it takes there), and that rank returns one row, the weighted sum of its experts' outputs. With nodes
        (``ep_node_size`` L > 1) the token goes once to each node, to the rank with this rank's position there
        (its "rail"), which forwards it once to each rank of its node that holds one of its experts and returns
        the sum of their rows. One all-to-all of row counts (with the input's amax) and one host read per
        forward size every message. With ``ep_chunks`` n the tokens go in n chunks, in a wavefront: the next
        chunk's rows travel while this chunk's experts run, and (by autograd's order) the gradients likewise."""
        T, k, E, W, El, L, d = x2d.shape[0], self.top_k, self.n_experts, self.ep_size, self.n_local, \
            self.ep_node_size, self.d
        N, dev, group = W // L, x2d.device, self.ep_group
        me_l = self.ep_rank % L
        nvfp4 = self.nvfp4 and not ops.high_precision() and x2d.is_cuda
        wire = self.ep_wire if nvfp4 else "bf16"
        wire_grad = self.ep_wire_grad if nvfp4 else "bf16"
        wret = self.ep_wire_return if x2d.is_cuda and self.d % 128 == 0 else "bf16"  # outputs, input gradients
        if nvfp4:
            xin = x2d.to(BF16)  # NVFP4 experts read bf16
        elif torch.is_autocast_enabled("cuda") and x2d.is_cuda:
            xin = x2d.to(torch.get_autocast_dtype("cuda"))  # the experts compute in it anyway
        else:
            xin = x2d
        n = max(1, min(self.ep_chunks, T))
        ser = (L > 1 and n > 1) if self.ep_serial == "auto" else bool(self.ep_serial)
        starts = [(c * T + n - 1) // n for c in range(n + 1)]
        sizes = [starts[c + 1] - starts[c] for c in range(n)]
        # --- what every rank sends where, per chunk: one all-to-all, one host read --------------------------
        chunk = ((torch.arange(T, device=dev) * n) // T).unsqueeze(1)  # [T, 1]
        valid = idx >= 0
        a = torch.zeros(n * E + 1, device=dev, dtype=torch.long).scatter_add_(
            0, torch.where(valid, chunk * E + idx, n * E).reshape(-1), torch.ones(T * k, device=dev, dtype=torch.long))

        def pairs(per, members):  # (token, member) pairs per chunk [n, members]
            mem, first = _first_choices(idx, per, members)
            key = torch.where(first, chunk * members + mem, n * members).reshape(-1)
            return torch.zeros(n * members + 1, device=dev, dtype=torch.long).scatter_add_(
                0, key, torch.ones_like(key))[:-1].view(n, members)
        b = pairs(El, W)           # rows to each rank
        cn = pairs(El * L, N)      # rows to each node
        F = El + 2 + L             # per (chunk, destination rank (m, l)): slots per expert there, rows to it,
        cnt = torch.cat([a[:-1].view(n, W, El), b.unsqueeze(2),   # rows to its node, rows to each rank of its node
                         cn.repeat_interleave(L, 1).unsqueeze(2), b.view(n, N, L).repeat_interleave(L, 1)], 2)
        amax = xin.abs().amax().float() if nvfp4 and T else torch.zeros((), device=dev)
        bits = amax.reshape(1).view(torch.int32).to(torch.long).view(1, 1).expand(W, 1)
        msg = torch.cat([cnt.permute(1, 0, 2).reshape(W, n * F), bits], 1).contiguous()
        got = torch.empty_like(msg)
        dist.all_to_all_single(got, msg, group=group)
        x_amax = got[:, -1].max().to(torch.int32).reshape(1).view(torch.float32) if nvfp4 else None
        host = torch.cat([msg, got]).tolist()
        sent, recvd = host[:W], host[W:]
        rail = [m * L + me_l for m in range(N)]  # the EP ranks of this rank's rail (rail group order)
        plans = []
        for c in range(n):
            o = c * F
            p = dict(send1=[sent[r][o + El + 1] for r in rail], recv1=[recvd[r][o + El + 1] for r in rail],
                     slots=sum(sum(recvd[r][o:o + El]) for r in range(W)))
            if L > 1:
                p["send2"] = [sum(recvd[r][o + El + 2 + l] for r in rail) for l in range(L)]
                p["recv2"] = [sum(recvd[r][o + El] for r in range(W) if r % L == l) for l in range(L)]
            plans.append(p)
        # rows this rank sends to other ranks: over the rails (to other nodes) and within the node
        self.last_ep_rows = (sum(v for c in range(n) for m, v in enumerate(plans[c]["send1"]) if m != self.ep_rank // L),
                             sum(v for c in range(n) for l, v in enumerate(plans[c].get("send2", ())) if l != me_l))
        ws, bs = self.experts_weights(), self.experts_biases()
        xs_c, ids_c, w_c = xin.split(sizes), idx.split(sizes), w.split(sizes)

        def run(c):  # one chunk; yields after starting each exchange but the last, returns the chunk's output
            p = plans[c]
            r1 = _route(ids_c[c], N, El * L, sum(p["send1"]))
            d1 = _Xchg(self._rail_group, p["send1"], p["recv1"], d, xin.dtype, k, w=True, ids=True, wire=wire,
                       amax=x_amax, wire_grad=wret, serial=ser)
            h = d1.start(_Dispatch.apply(xs_c[c], r1.src, r1.pos)[:r1.rows.shape[0]],
                         w_c[c].index_select(0, r1.rows) * r1.mask, r1.ids)
            yield
            xe, we, ie = d1.finish(h)
            box = _Box()
            if L > 1:  # forward within the node
                r2 = _route(ie, L, El, sum(p["send2"]))
                d2 = _Xchg(self._node_group, p["send2"], p["recv2"], d, xin.dtype, k, w=True, ids=True, wire=wire,
                           amax=x_amax, wire_grad=wret, serial=ser)
                h = d2.start(_Dispatch.apply(xe, r2.src, r2.pos)[:r2.rows.shape[0]],
                             we.index_select(0, r2.rows) * r2.mask, r2.ids)
                yield
                xe, we, ie = d2.finish(h)
            R = xe.shape[0]
            ys, pos = self._mlp(xe, ie.reshape(-1).long(), torch.arange(R * k, device=dev) // k, k, El, self.d_ff,
                                p["slots"], True, ws, bs, self._caches, x_amax)
            part = _combine(ys, pos, we)  # this rank's experts' weighted sum for every row
            if L > 1:
                back = _Xchg(self._node_group, p["recv2"], p["send2"], d, part.dtype, wire=wret,
                             wire_grad=wire_grad, grad_amax=box, serial=ser)
                h = back.start(part)
                yield
                part = _gather_sum(back.finish(h)[0], r2.pos)  # the node's sum for every row
            back = _Xchg(self._rail_group, p["recv1"], p["send1"], d, part.dtype, wire=wret,
                         wire_grad=wire_grad, grad_amax=group, box=box, serial=ser)
            h = back.start(part)
            yield
            return _gather_sum(back.finish(h)[0], r1.pos)

        gens = [run(c) for c in range(n)]
        steps = 2 if L == 1 else 4  # yields per chunk
        for tick in range(n + steps - 1):  # wavefront: later chunks start their exchanges first
            for c in reversed(range(n)):
                if 0 <= tick - c < steps:
                    next(gens[c])
        sh = self._shared(x2d) if self.shared_d_ff and self.ep_overlap else None  # during the last returns
        ys = []
        for g in gens:
            try:
                next(g)
            except StopIteration as e:
                ys.append(e.value)
        y = torch.cat(ys) if n > 1 else ys[0]
        if self.shared_d_ff and sh is None:
            sh = self._shared(x2d)
        return y, sh

    def _forward_ep(self, x2d, slot_expert, slot_src, rank, counts, w, C: int):
        """Expert parallelism with a capacity and fixed-size exchanges: (routed output, shared expert's output or
        None). The slots go to their expert's rank, the local experts run on the received rows, the outputs come
        back and are combined. The shared expert runs while the outputs travel back (and its backward while
        their gradients go out)."""
        T, k, E, W, El = x2d.shape[0], self.top_k, self.n_experts, self.ep_size, self.n_local
        dev = x2d.device
        nvfp4 = self.nvfp4 and not ops.high_precision() and x2d.is_cuda
        xin = x2d.to(BF16) if nvfp4 else x2d  # NVFP4 experts read bf16: half the bytes on the wire
        shared = (lambda: self._shared(x2d)) if self.shared_d_ff else None
        overlap = shared if self.ep_overlap else None
        group, ws, bs = self.ep_group, self.experts_weights(), self.experts_biases()
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
            r += f", expert_parallel={self.ep_size} (local experts {self.n_local}" + \
                (f", nodes of {self.ep_node_size})" if self.ep_node_size > 1 else ")")
        return r + (", nvfp4=True" if self.nvfp4 else "")


def expert_parallel_parameters(module: nn.Module) -> list[str]:
    """Names (as in ``module.named_parameters()``) of the expert weights and biases of the expert-parallel MoE
    layers in ``module``: they differ per rank, so data parallelism over the same ranks must leave them alone
    (``DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(module, names)``, see
    ``ignore_expert_parameters_in_ddp``), and their gradients already sum every rank's tokens."""
    names = []
    for prefix, m in module.named_modules():
        if isinstance(m, MoE) and m.ep_group is not None:
            for n, p in m.named_parameters(recurse=False):
                if p is not None and n.startswith(("w_", "b_")):
                    names.append(f"{prefix}.{n}" if prefix else n)
    return names


def ignore_expert_parameters_in_ddp(module: nn.Module) -> list[str]:
    """Before wrapping ``module`` in DistributedDataParallel: exclude the expert-parallel experts (their
    gradients are not to be averaged over ranks); returns their names."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    names = expert_parallel_parameters(module)
    DDP._set_params_and_buffers_to_ignore_for_model(module, names)
    return names


@torch.no_grad()
def scale_expert_grads(module: nn.Module, factor: float | None = None) -> None:
    """Multiply the expert-parallel experts' gradients by ``factor`` (default 1 / the expert-parallel group's
    size). Data parallelism averages the other gradients over ranks (the objective is the mean of the ranks'
    losses); an expert's gradient from expert parallelism is the sum of every rank's contribution. Call before
    the optimizer step (and before gradient clipping)."""
    for m in module.modules():
        if isinstance(m, MoE) and m.ep_group is not None:
            f = 1.0 / m.ep_size if factor is None else factor
            for n, p in m.named_parameters(recurse=False):
                if p is not None and p.grad is not None and n.startswith(("w_", "b_")):
                    p.grad.mul_(f)


__all__ = ["MoE", "expert_parallel_parameters", "ignore_expert_parameters_in_ddp", "scale_expert_grads"]
