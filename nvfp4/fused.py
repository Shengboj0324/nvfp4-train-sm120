"""Transformer sub-blocks with the producer ops fused into NVFP4 quantization (CUDA backend).

    layernorm_linear(x, ln, lin)        lin(ln(x))                      e.g. pre-LN attention qkv
    layernorm_mlp(x, ln, fc1, fc2)      fc2(gelu_tanh(fc1(ln(x))))      pre-LN MLP

Same recipe as NVFP4Linear, but LayerNorm, GELU and the GELU backward run inside the quantizers
(ops.layernorm_quantize_dual / gelu_quantize_dual / gelu_backward_quantize_dual), so the LayerNorm
output, the GELU output and the GELU input gradient are never written to memory, and each costs no
separate elementwise pass. What is saved for backward: x, the LayerNorm statistics, the fc1 output h
(for the GELU gradient) and the NVFP4 transposed operands, as before.

Numerics differ from the unfused composition only in rounding: the fused path quantizes the fp32
LayerNorm / GELU outputs directly instead of first rounding them to bf16, and uses its own LayerNorm
statistics (fp32, tree-reduced), which the LayerNorm backward then also uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4 import ops
from nvfp4.linear import ALIGN, NVFP4Linear, _seed, _weight_grad

BF16 = torch.bfloat16


def _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask):
    """LayerNorm backward with our statistics (aten kernel, fp32; dln comes from an fp32-output GEMM)."""
    dx, dw, db = torch.ops.aten.native_layer_norm_backward(
        dln, x2d.float(), [x2d.shape[1]], mean[:, None], rstd[:, None], ln_w.float(), ln_b.float(), mask)
    return (dx.to(x2d.dtype) if dx is not None else None, dw, db)


def _sr_seeds(row: bool, col: bool):
    return (_seed() if row else None), (_seed() if col else None)


class _LayerNormLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, ln_w, ln_b, eps, weight, wgrad_rht, need_dln, need_wgrad, fuse_wgrad):
        rht = _seed() if need_wgrad and wgrad_rht else None
        mean, rstd, xq, xtq = ops.layernorm_quantize_dual(x2d, ln_w, ln_b, eps, col=need_wgrad, rht_seed=rht,
                                                          col_pad=ALIGN)
        wq, wtq = ops.quantize_dual(weight.detach(), col=need_dln, block_2d=True)
        y = ops.gemm(xq, wq, BF16)
        ctx.save_for_backward(x2d, ln_w, ln_b, mean, rstd,
                              *(xtq.as_storage() if need_wgrad else ()), *(wtq.as_storage() if need_dln else ()))
        ctx.rht, ctx.w_dtype, ctx.need_dln, ctx.need_wgrad = rht, weight.dtype, need_dln, need_wgrad
        ctx.param = weight if fuse_wgrad else None
        return y

    @staticmethod
    def backward(ctx, dy):
        x2d, ln_w, ln_b, mean, rstd, *rest = ctx.saved_tensors
        xtq = ops.NVFP4Tensor(*rest[:3]) if ctx.need_wgrad else None
        wtq = ops.NVFP4Tensor(*rest[-3:]) if ctx.need_dln else None
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]]
        do_dln = ctx.need_dln and any(mask)
        do_dw = ctx.need_wgrad and ctx.needs_input_grad[4]
        grads = [None] * 9
        if not (do_dln or do_dw):
            return tuple(grads)
        row_seed, col_seed = _sr_seeds(do_dln, do_dw)
        dyq, dytq = ops.quantize_dual(dy.contiguous().to(BF16), row=do_dln, col=do_dw, row_sr_seed=row_seed,
                                      col_sr_seed=col_seed, rht_seed=ctx.rht, col_pad=ALIGN)
        # Temporaries are dropped as soon as they are consumed: everything referenced inside one backward
        # call stays alive until it returns, unlike between separate autograd nodes.
        if do_dw:
            grads[4] = _weight_grad(ctx.param, dytq, xtq, ctx.w_dtype, "cuda")
        del dytq, xtq
        if do_dln:
            dln = ops.gemm(dyq, wtq, torch.float32)  # fp32 straight into the LayerNorm backward
            del dyq
            grads[0], grads[1], grads[2] = _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask)
        return tuple(grads)


class _LayerNormMLPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, ln_w, ln_b, eps, w1, w2, wgrad_rht, need_dln, need_w1, need_w2, fuse_wgrad):
        rht1 = _seed() if need_w1 and wgrad_rht else None
        rht2 = _seed() if need_w2 and wgrad_rht else None
        need_dh = need_dln or need_w1  # the gradient has to flow back through fc2 and the GELU
        mean, rstd, xq, xtq = ops.layernorm_quantize_dual(x2d, ln_w, ln_b, eps, col=need_w1, rht_seed=rht1,
                                                          col_pad=ALIGN)
        w1q, w1tq = ops.quantize_dual(w1.detach(), col=need_dln, block_2d=True)
        h = ops.gemm(xq, w1q, BF16)                                       # fc1 output (GELU input)
        gq, gtq = ops.gelu_quantize_dual(h, col=need_w2, rht_seed=rht2, col_pad=ALIGN)
        w2q, w2tq = ops.quantize_dual(w2.detach(), col=need_dh, block_2d=True)
        out = ops.gemm(gq, w2q, BF16)
        saved = [x2d, ln_w, ln_b, mean, rstd, h]
        for flag, q in ((need_w1, xtq), (need_dln, w1tq), (need_w2, gtq), (need_dh, w2tq)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.flags = (need_w1, need_dln, need_w2, need_dh)
        ctx.rht1, ctx.rht2 = rht1, rht2
        ctx.dtypes = (w1.dtype, w2.dtype)
        ctx.params = (w1, w2) if fuse_wgrad else (None, None)
        return out

    @staticmethod
    def backward(ctx, dout):
        x2d, ln_w, ln_b, mean, rstd, h, *rest = ctx.saved_tensors
        need_w1, need_dln, need_w2, need_dh = ctx.flags
        qs = {}
        for name, flag in (("xtq", need_w1), ("w1tq", need_dln), ("gtq", need_w2), ("w2tq", need_dh)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]]
        do_dln = need_dln and any(mask)
        do_dw1 = need_w1 and ctx.needs_input_grad[4]
        do_dw2 = need_w2 and ctx.needs_input_grad[5]
        do_dg = do_dln or do_dw1
        grads = [None] * 11
        if not (do_dg or do_dw2):
            return tuple(grads)
        row_seed, col_seed = _sr_seeds(do_dg, do_dw2)
        doq, dotq = ops.quantize_dual(dout.contiguous().to(BF16), row=do_dg, col=do_dw2, row_sr_seed=row_seed,
                                      col_sr_seed=col_seed, rht_seed=ctx.rht2, col_pad=ALIGN)
        # Temporaries are dropped as soon as they are consumed (see _LayerNormLinearFn.backward).
        if do_dw2:
            grads[5] = _weight_grad(ctx.params[1], dotq, qs.pop("gtq"), ctx.dtypes[1], "cuda")
        del dotq
        if do_dg:
            dg = ops.gemm(doq, qs.pop("w2tq"), BF16)                     # gradient w.r.t. gelu(h)
            del doq
            row_seed, col_seed = _sr_seeds(do_dln, do_dw1)
            dhq, dhtq = ops.gelu_backward_quantize_dual(dg, h, row=do_dln, col=do_dw1, row_sr_seed=row_seed,
                                                        col_sr_seed=col_seed, rht_seed=ctx.rht1, col_pad=ALIGN)
            del dg, h
            if do_dw1:
                grads[4] = _weight_grad(ctx.params[0], dhtq, qs.pop("xtq"), ctx.dtypes[0], "cuda")
            del dhtq
            if do_dln:
                dln = ops.gemm(dhq, qs.pop("w1tq"), torch.float32)      # fp32 straight into the LayerNorm backward
                del dhq
                grads[0], grads[1], grads[2] = _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask)
        return tuple(grads)


def _ln_params(ln: torch.nn.LayerNorm, d: int, device):
    if ln.normalized_shape != (d,):
        raise ValueError(f"LayerNorm must normalize the last dim ({d}), got {ln.normalized_shape}")
    w = ln.weight if ln.weight is not None else torch.ones(d, device=device)
    b = ln.bias if ln.bias is not None else torch.zeros(d, device=device)
    return w, b


def _check_linear(lin, d_in: int):
    if not isinstance(lin, NVFP4Linear) or lin.backend != "cuda":
        raise TypeError("fused sub-blocks need NVFP4Linear modules with backend='cuda'")
    if lin.bias is not None:
        raise ValueError("fused sub-blocks support bias-free linears only")
    if lin.in_features != d_in:
        raise ValueError(f"expected in_features={d_in}, got {lin.in_features}")


def _out_dtype(x):
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype


def _needs(x, *params):
    g = torch.is_grad_enabled()
    return [g and t.requires_grad for t in (x, *params)]


def layernorm_linear(x: torch.Tensor, ln: torch.nn.LayerNorm, lin: NVFP4Linear) -> torch.Tensor:
    """``lin(ln(x))`` with the LayerNorm fused into the quantization of the linear's input."""
    d = x.shape[-1]
    _check_linear(lin, d)
    ln_w, ln_b = _ln_params(ln, d, x.device)
    x2d = x.reshape(-1, d)
    if x2d.dtype not in (torch.float32, BF16):
        x2d = x2d.float()
    gx, gw, gb, gl = _needs(x2d, ln_w, ln_b, lin.weight)
    y = _LayerNormLinearFn.apply(x2d.contiguous(), ln_w, ln_b, ln.eps, lin.weight, lin.wgrad_rht, gx or gw or gb, gl,
                                 lin.fuse_wgrad_accumulation)
    return y.to(_out_dtype(x)).reshape(*x.shape[:-1], lin.out_features)


def layernorm_mlp(x: torch.Tensor, ln: torch.nn.LayerNorm, fc1: NVFP4Linear, fc2: NVFP4Linear) -> torch.Tensor:
    """``fc2(gelu_tanh(fc1(ln(x))))`` with LayerNorm, GELU and the GELU backward fused into quantization."""
    d = x.shape[-1]
    _check_linear(fc1, d)
    _check_linear(fc2, fc1.out_features)
    ln_w, ln_b = _ln_params(ln, d, x.device)
    x2d = x.reshape(-1, d)
    if x2d.dtype not in (torch.float32, BF16):
        x2d = x2d.float()
    gx, gw, gb, g1, g2 = _needs(x2d, ln_w, ln_b, fc1.weight, fc2.weight)
    out = _LayerNormMLPFn.apply(x2d.contiguous(), ln_w, ln_b, ln.eps, fc1.weight, fc2.weight, fc1.wgrad_rht,
                                gx or gw or gb, g1, g2, fc1.fuse_wgrad_accumulation)
    return out.to(_out_dtype(x)).reshape(*x.shape[:-1], fc2.out_features)


__all__ = ["layernorm_linear", "layernorm_mlp"]
