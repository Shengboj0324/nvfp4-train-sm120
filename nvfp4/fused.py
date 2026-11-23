"""Transformer sub-blocks with the producer ops fused into NVFP4 quantization (CUDA backend).

    layernorm_linear(x, ln, lin)                    lin(ln(x))                       GPT / BERT-style qkv
    layernorm_mlp(x, ln, fc1, fc2, approximate)     fc2(gelu(fc1(ln(x))))            GPT / BERT-style MLP
    rmsnorm_linear(x, norm, lin)                    lin(rmsnorm(x))                  Llama-style qkv
    rmsnorm_swiglu_mlp(x, norm, gate, up, down)     down(silu(gate(n)) * up(n)),     Llama-style MLP
                                                    n = rmsnorm(x)

Linears may have biases (except the SwiGLU gate / up projections). A bias after a GEMM is added as in
NVFP4Linear; fc1's bias is added inside the GELU producer (in fp32), and its gradient comes from column
sums the GELU-backward quantizer computes from the same read. GELU is the tanh approximation
(``approximate="tanh"``, GPT-2) or exact (``"none"``, BERT / ViT).

Same recipe as NVFP4Linear, but LayerNorm, GELU and the GELU backward run inside the quantizers
(ops.layernorm_quantize_dual / gelu_quantize_dual / gelu_backward_quantize_dual), so the LayerNorm
output, the GELU output and the GELU input gradient are never written to memory, and each costs no
separate elementwise pass. What is saved for backward: x, the LayerNorm statistics, the fc1 output h
(for the GELU gradient) and the NVFP4 transposed operands, as before.

The Llama-style blocks do the same with RMSNorm and SwiGLU: the normalized input is quantized once
for both the gate and the up projection (and its transposed copy serves both weight gradients), and
silu(gate) * up as well as both halves of its gradient are computed inside the quantizers.

Numerics differ from the unfused composition only in rounding: the fused path quantizes the fp32
LayerNorm / RMSNorm / GELU / SwiGLU outputs directly instead of first rounding them to bf16, and uses
its own normalization statistics (fp32, tree-reduced), which the backward then also uses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4 import ops
from nvfp4.linear import (ALIGN, NVFP4Linear, _weight_grad, draw_seeds, quantized_weight, unflatten_output,
                          weight_cache)

BF16 = torch.bfloat16


def _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask):
    """LayerNorm backward with our statistics (aten kernel, fp32; dln comes from an fp32-output GEMM)."""
    dx, dw, db = torch.ops.aten.native_layer_norm_backward(
        dln, x2d.float(), [x2d.shape[1]], mean[:, None], rstd[:, None], ln_w.float(), ln_b.float(), mask)
    return (dx.to(x2d.dtype) if dx is not None else None, dw, db)


_fused_rms_bwd = getattr(torch.ops.aten, "_fused_rms_norm_backward", None)


def _rms_backward(dn, x2d, rstd, w, mask):
    """RMSNorm backward (fp32) with our statistic: n = x * rstd, y = n * w,
    dx = rstd * (dn * w - n * mean(dn * w * n)), dw = sum over rows of dn * n.
    PyTorch's fused kernel when it has one (>= 2.9); one pass instead of ~10 elementwise ops."""
    if _fused_rms_bwd is not None:
        dx, dw = _fused_rms_bwd(dn, x2d.float(), [x2d.shape[1]], rstd[:, None], w.float(), mask)
        return (dx.to(x2d.dtype) if dx is not None else None), dw
    xf = x2d.float()
    n = xf * rstd[:, None]
    dw = (dn * n).sum(0) if mask[1] else None
    dx = None
    if mask[0]:
        gn = dn * w.float()
        dx = (rstd[:, None] * (gn - n * (gn * n).mean(-1, keepdim=True))).to(x2d.dtype)
    return dx, dw


def _sr(seeds, row: bool, col: bool, rht: bool):
    """Seed keyword arguments from a [row SR, col SR, RHT] seed tensor."""
    return dict(row_sr_seed=seeds[0] if row else None, col_sr_seed=seeds[1] if col else None,
                rht_seed=seeds[2] if rht else None)


class _LayerNormLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, ln_w, ln_b, eps, weight, wgrad_rht, need_dln, need_wgrad, fuse_wgrad, wcache):
        seeds = draw_seeds(3, x2d.device) if need_dln or need_wgrad else None  # [dY row SR, dY^T SR, RHT]
        use_rht = need_wgrad and wgrad_rht
        mean, rstd, xq, xtq = ops.layernorm_quantize_dual(x2d, ln_w, ln_b, eps, col=need_wgrad,
                                                          rht_seed=seeds[2] if use_rht else None, col_pad=ALIGN)
        wq, wtq = quantized_weight(weight, need_dln, "cuda", wcache)
        y = ops.gemm(xq, wq, BF16)
        ctx.save_for_backward(x2d, ln_w, ln_b, mean, rstd, seeds,
                              *(xtq.as_storage() if need_wgrad else ()), *(wtq.as_storage() if need_dln else ()))
        ctx.use_rht, ctx.w_dtype, ctx.need_dln, ctx.need_wgrad = use_rht, weight.dtype, need_dln, need_wgrad
        ctx.param = weight if fuse_wgrad else None
        return y

    @staticmethod
    def backward(ctx, dy):
        x2d, ln_w, ln_b, mean, rstd, seeds, *rest = ctx.saved_tensors
        xtq = ops.NVFP4Tensor(*rest[:3]) if ctx.need_wgrad else None
        wtq = ops.NVFP4Tensor(*rest[-3:]) if ctx.need_dln else None
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]]
        do_dln = ctx.need_dln and any(mask)
        do_dw = ctx.need_wgrad and ctx.needs_input_grad[4]
        grads = [None] * 10
        if not (do_dln or do_dw):
            return tuple(grads)
        dyq, dytq = ops.quantize_dual(dy.contiguous().to(BF16), row=do_dln, col=do_dw, col_pad=ALIGN,
                                      **_sr(seeds, do_dln, do_dw, ctx.use_rht))
        # Temporaries are dropped as soon as they are consumed: everything referenced inside one backward
        # call stays alive until it returns, unlike between separate autograd nodes.
        if do_dw:
            grads[4] = _weight_grad(ctx, ctx.param, dytq, xtq, ctx.w_dtype, "cuda")
        del dytq, xtq
        if do_dln:
            dln = ops.gemm(dyq, wtq, torch.float32)  # fp32 straight into the LayerNorm backward
            del dyq
            grads[0], grads[1], grads[2] = _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask)
        return tuple(grads)


class _LayerNormMLPFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, ln_w, ln_b, eps, w1, w2, b1, wgrad_rht, need_dln, need_w1, need_w2, need_b1, fuse_wgrad,
                wc1, wc2, approximate):
        # [dH row SR, dH^T SR, RHT of fc1's wgrad, dOut row SR, dOut^T SR, RHT of fc2's wgrad]
        seeds = draw_seeds(6, x2d.device) if need_dln or need_w1 or need_w2 or need_b1 else None
        rht1, rht2 = need_w1 and wgrad_rht, need_w2 and wgrad_rht
        need_dh = need_dln or need_w1 or need_b1  # the gradient has to flow back through fc2 and the GELU
        mean, rstd, xq, xtq = ops.layernorm_quantize_dual(x2d, ln_w, ln_b, eps, col=need_w1,
                                                          rht_seed=seeds[2] if rht1 else None,
                                                          col_pad=ALIGN)
        w1q, w1tq = quantized_weight(w1, need_dln, "cuda", wc1)
        h = ops.gemm(xq, w1q, BF16)                                       # fc1 output (GELU input, without bias)
        gq, gtq = ops.gelu_quantize_dual(h, bias=b1, approximate=approximate, col=need_w2,
                                         rht_seed=seeds[5] if rht2 else None, col_pad=ALIGN)
        w2q, w2tq = quantized_weight(w2, need_dh, "cuda", wc2)
        out = ops.gemm(gq, w2q, BF16)
        saved = [x2d, ln_w, ln_b, mean, rstd, h, seeds, b1]
        for flag, q in ((need_w1, xtq), (need_dln, w1tq), (need_w2, gtq), (need_dh, w2tq)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.flags = (need_w1, need_dln, need_w2, need_dh)
        ctx.rht1, ctx.rht2, ctx.approximate = rht1, rht2, approximate
        ctx.dtypes = (w1.dtype, w2.dtype)
        ctx.params = (w1, w2) if fuse_wgrad else (None, None)
        return out

    @staticmethod
    def backward(ctx, dout):
        x2d, ln_w, ln_b, mean, rstd, h, seeds, b1, *rest = ctx.saved_tensors
        need_w1, need_dln, need_w2, need_dh = ctx.flags
        qs = {}
        for name, flag in (("xtq", need_w1), ("w1tq", need_dln), ("gtq", need_w2), ("w2tq", need_dh)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]]
        do_dln = need_dln and any(mask)
        do_dw1 = need_w1 and ctx.needs_input_grad[4]
        do_dw2 = need_w2 and ctx.needs_input_grad[5]
        do_db1 = b1 is not None and ctx.needs_input_grad[6]
        do_dg = do_dln or do_dw1 or do_db1
        grads = [None] * 17
        if not (do_dg or do_dw2):
            return tuple(grads)
        doq, dotq = ops.quantize_dual(dout.contiguous().to(BF16), row=do_dg, col=do_dw2, col_pad=ALIGN,
                                      **_sr(seeds[3:], do_dg, do_dw2, ctx.rht2))
        # Temporaries are dropped as soon as they are consumed (see _LayerNormLinearFn.backward).
        if do_dw2:
            grads[5] = _weight_grad(ctx, ctx.params[1], dotq, qs.pop("gtq"), ctx.dtypes[1], "cuda")
        del dotq
        if do_dg:
            dg = ops.gemm(doq, qs.pop("w2tq"), BF16)                     # gradient w.r.t. gelu(h)
            del doq
            outs = ops.gelu_backward_quantize_dual(dg, h, bias=b1, approximate=ctx.approximate, bias_grad=do_db1,
                                                   row=do_dln or not do_dw1, col=do_dw1, col_pad=ALIGN,
                                                   **_sr(seeds, do_dln, do_dw1, ctx.rht1))
            dhq, dhtq = outs[0], outs[1]
            if do_db1:
                grads[6] = outs[2].to(b1.dtype)
            del dg, h, outs
            if do_dw1:
                grads[4] = _weight_grad(ctx, ctx.params[0], dhtq, qs.pop("xtq"), ctx.dtypes[0], "cuda")
            del dhtq
            if do_dln:
                dln = ops.gemm(dhq, qs.pop("w1tq"), torch.float32)      # fp32 straight into the LayerNorm backward
                del dhq
                grads[0], grads[1], grads[2] = _ln_backward(dln, x2d, mean, rstd, ln_w, ln_b, mask)
        return tuple(grads)


class _RMSNormLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, norm_w, eps, weight, wgrad_rht, need_dn, need_wgrad, fuse_wgrad, wcache):
        seeds = draw_seeds(3, x2d.device) if need_dn or need_wgrad else None  # [dY row SR, dY^T SR, RHT]
        use_rht = need_wgrad and wgrad_rht
        rstd, xq, xtq = ops.rmsnorm_quantize_dual(x2d, norm_w, eps, col=need_wgrad,
                                                  rht_seed=seeds[2] if use_rht else None, col_pad=ALIGN)
        wq, wtq = quantized_weight(weight, need_dn, "cuda", wcache)
        y = ops.gemm(xq, wq, BF16)
        ctx.save_for_backward(x2d, norm_w, rstd, seeds,
                              *(xtq.as_storage() if need_wgrad else ()), *(wtq.as_storage() if need_dn else ()))
        ctx.use_rht, ctx.w_dtype, ctx.need_dn, ctx.need_wgrad = use_rht, weight.dtype, need_dn, need_wgrad
        ctx.param = weight if fuse_wgrad else None
        return y

    @staticmethod
    def backward(ctx, dy):
        x2d, norm_w, rstd, seeds, *rest = ctx.saved_tensors
        xtq = ops.NVFP4Tensor(*rest[:3]) if ctx.need_wgrad else None
        wtq = ops.NVFP4Tensor(*rest[-3:]) if ctx.need_dn else None
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1]]
        do_dn = ctx.need_dn and any(mask)
        do_dw = ctx.need_wgrad and ctx.needs_input_grad[3]
        grads = [None] * 9
        if not (do_dn or do_dw):
            return tuple(grads)
        dyq, dytq = ops.quantize_dual(dy.contiguous().to(BF16), row=do_dn, col=do_dw, col_pad=ALIGN,
                                      **_sr(seeds, do_dn, do_dw, ctx.use_rht))
        if do_dw:
            grads[3] = _weight_grad(ctx, ctx.param, dytq, xtq, ctx.w_dtype, "cuda")
        del dytq, xtq
        if do_dn:
            dn = ops.gemm(dyq, wtq, torch.float32)
            del dyq
            grads[0], grads[1] = _rms_backward(dn, x2d, rstd, norm_w, mask)
        return tuple(grads)


class _RMSNormSwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, norm_w, eps, wg, wu, wd, wgrad_rht, need_dn, need_wg, need_wu, need_wd, fuse_wgrad,
                wcg, wcu, wcd):
        # [dG row SR, dG^T SR, RHT of x^T (gate and up wgrad), dOut row SR, dOut^T SR, RHT of the down
        #  projection's wgrad, dU row SR, dU^T SR]
        need_any = need_dn or need_wg or need_wu or need_wd
        seeds = draw_seeds(8, x2d.device) if need_any else None
        need_xt = need_wg or need_wu
        rht_x, rht_a = need_xt and wgrad_rht, need_wd and wgrad_rht
        need_da = need_dn or need_wg or need_wu  # the gradient has to flow back through down and SwiGLU
        rstd, xq, xtq = ops.rmsnorm_quantize_dual(x2d, norm_w, eps, col=need_xt,
                                                  rht_seed=seeds[2] if rht_x else None, col_pad=ALIGN)
        wgq, wgtq = quantized_weight(wg, need_dn, "cuda", wcg)
        wuq, wutq = quantized_weight(wu, need_dn, "cuda", wcu)
        g = ops.gemm(xq, wgq, BF16)                                    # gate projection
        u = ops.gemm(xq, wuq, BF16)                                    # up projection
        aq, atq = ops.swiglu_quantize_dual(g, u, col=need_wd, rht_seed=seeds[5] if rht_a else None, col_pad=ALIGN)
        wdq, wdtq = quantized_weight(wd, need_da, "cuda", wcd)
        out = ops.gemm(aq, wdq, BF16)
        saved = [x2d, norm_w, rstd, g, u, seeds]
        for flag, q in ((need_xt, xtq), (need_dn, wgtq), (need_dn, wutq), (need_wd, atq), (need_da, wdtq)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.flags = (need_xt, need_dn, need_wd, need_da)
        ctx.rht_x, ctx.rht_a = rht_x, rht_a
        ctx.dtypes = (wg.dtype, wu.dtype, wd.dtype)
        ctx.params = (wg, wu, wd) if fuse_wgrad else (None, None, None)
        return out

    @staticmethod
    def backward(ctx, dout):
        x2d, norm_w, rstd, g, u, seeds, *rest = ctx.saved_tensors
        need_xt, need_dn, need_wd, need_da = ctx.flags
        qs = {}
        for name, flag in (("xtq", need_xt), ("wgtq", need_dn), ("wutq", need_dn), ("atq", need_wd),
                           ("wdtq", need_da)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[1]]
        do_dn = need_dn and any(mask)
        do_wg = need_xt and ctx.needs_input_grad[3]
        do_wu = need_xt and ctx.needs_input_grad[4]
        do_wd = need_wd and ctx.needs_input_grad[5]
        do_da = do_dn or do_wg or do_wu
        grads = [None] * 15
        if not (do_da or do_wd):
            return tuple(grads)
        doq, dotq = ops.quantize_dual(dout.contiguous().to(BF16), row=do_da, col=do_wd, col_pad=ALIGN,
                                      **_sr(seeds[3:6], do_da, do_wd, ctx.rht_a))
        # Temporaries are dropped as soon as they are consumed (see _LayerNormLinearFn.backward).
        if do_wd:
            grads[5] = _weight_grad(ctx, ctx.params[2], dotq, qs.pop("atq"), ctx.dtypes[2], "cuda")
        del dotq
        if not do_da:
            return tuple(grads)
        da = ops.gemm(doq, qs.pop("wdtq"), BF16)                       # gradient w.r.t. silu(g) * u
        del doq
        dn = None
        for k, (do_w, row_s, col_s) in enumerate(((do_wg, 0, 1), (do_wu, 6, 7))):  # gate, then up
            if not (do_dn or do_w):
                continue
            dq, dtq = ops.swiglu_backward_quantize_dual(
                da, g, u, gate=k == 0, row=do_dn, col=do_w, col_pad=ALIGN,
                row_sr_seed=seeds[row_s] if do_dn else None, col_sr_seed=seeds[col_s] if do_w else None,
                rht_seed=seeds[2] if do_w and ctx.rht_x else None)
            if do_w:  # dW = dG^T x (dU^T x): both operands carry x^T's RHT
                grads[3 + k] = _weight_grad(ctx, ctx.params[k], dtq, qs["xtq"], ctx.dtypes[k], "cuda")
            del dtq
            if do_dn:  # d(normalized x) = dG @ Wg + dU @ Wu, accumulated in fp32
                wt = qs.pop("wgtq" if k == 0 else "wutq")
                dn = ops.gemm(dq, wt, torch.float32) if dn is None else ops.gemm(dq, wt, out=dn, accumulate=True)
            del dq
        del da, g, u
        qs.pop("xtq", None)
        if do_dn:
            grads[0], grads[1] = _rms_backward(dn, x2d, rstd, norm_w, mask)
        return tuple(grads)


def _rms_params(norm: torch.nn.Module, d: int, x: torch.Tensor):
    """(weight, eps) of a torch.nn.RMSNorm (or a module with ``weight`` and ``variance_epsilon``, like
    Hugging Face's LlamaRMSNorm)."""
    shape = getattr(norm, "normalized_shape", None)
    if shape is not None and tuple(shape) != (d,):
        raise ValueError(f"RMSNorm must normalize the last dim ({d}), got {tuple(shape)}")
    w = getattr(norm, "weight", None)
    if w is None:
        w = torch.ones(d, device=x.device)
    elif w.shape != (d,):
        raise ValueError(f"RMSNorm weight must have shape ({d},), got {tuple(w.shape)}")
    eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", None))
    if eps is None:  # torch.nn.RMSNorm's default
        eps = torch.finfo(x.dtype).eps
    return w, float(eps)


def _ln_params(ln: torch.nn.LayerNorm, d: int, device):
    if ln.normalized_shape != (d,):
        raise ValueError(f"LayerNorm must normalize the last dim ({d}), got {ln.normalized_shape}")
    w = ln.weight if ln.weight is not None else torch.ones(d, device=device)
    b = ln.bias if ln.bias is not None else torch.zeros(d, device=device)
    return w, b


def _check_linear(lin, d_in: int, bias_ok: bool = True):
    if not isinstance(lin, NVFP4Linear) or lin.backend != "cuda":
        raise TypeError("fused sub-blocks need NVFP4Linear modules with backend='cuda'")
    if lin.bias is not None and not bias_ok:
        raise ValueError("the SwiGLU gate / up projections must be bias-free")
    if lin.in_features != d_in:
        raise ValueError(f"expected in_features={d_in}, got {lin.in_features}")


def _out_dtype(x):
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype


def _finish(y, x, lin):
    """Cast to the caller's dtype and add the bias as NVFP4Linear does; [*lead, out] without a view."""
    dt = _out_dtype(x)
    y = y.to(dt)
    if lin.bias is not None:
        y = y + lin.bias.to(dt)
    return unflatten_output(y, x.shape[:-1], lin.out_features)


def _fuse(lin: NVFP4Linear) -> bool:
    return lin.fuse_wgrad_accumulation and not torch.compiler.is_compiling()


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
                                 _fuse(lin), weight_cache(lin))
    return _finish(y, x, lin)


def layernorm_mlp(x: torch.Tensor, ln: torch.nn.LayerNorm, fc1: NVFP4Linear, fc2: NVFP4Linear,
                  approximate: str = "tanh") -> torch.Tensor:
    """``fc2(gelu(fc1(ln(x))))`` with LayerNorm, GELU (``approximate="tanh"`` or ``"none"``, as in
    torch.nn.functional.gelu) and the GELU backward fused into quantization."""
    if approximate not in ("tanh", "none"):
        raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")
    d = x.shape[-1]
    _check_linear(fc1, d)
    _check_linear(fc2, fc1.out_features)
    ln_w, ln_b = _ln_params(ln, d, x.device)
    x2d = x.reshape(-1, d)
    if x2d.dtype not in (torch.float32, BF16):
        x2d = x2d.float()
    b1 = fc1.bias
    gx, gw, gb, g1, g2 = _needs(x2d, ln_w, ln_b, fc1.weight, fc2.weight)
    gb1 = b1 is not None and _needs(b1)[0]
    out = _LayerNormMLPFn.apply(x2d.contiguous(), ln_w, ln_b, ln.eps, fc1.weight, fc2.weight, b1, fc1.wgrad_rht,
                                gx or gw or gb, g1, g2, gb1, _fuse(fc1), weight_cache(fc1), weight_cache(fc2),
                                approximate)
    return _finish(out, x, fc2)


def rmsnorm_linear(x: torch.Tensor, norm: torch.nn.Module, lin: NVFP4Linear) -> torch.Tensor:
    """``lin(norm(x))`` with the RMSNorm fused into the quantization of the linear's input."""
    d = x.shape[-1]
    _check_linear(lin, d)
    x2d = x.reshape(-1, d)
    if x2d.dtype not in (torch.float32, BF16):
        x2d = x2d.float()
    norm_w, eps = _rms_params(norm, d, x2d)
    gx, gw, gl = _needs(x2d, norm_w, lin.weight)
    y = _RMSNormLinearFn.apply(x2d.contiguous(), norm_w, eps, lin.weight, lin.wgrad_rht, gx or gw, gl, _fuse(lin),
                               weight_cache(lin))
    return _finish(y, x, lin)


def rmsnorm_swiglu_mlp(x: torch.Tensor, norm: torch.nn.Module, gate: NVFP4Linear, up: NVFP4Linear,
                       down: NVFP4Linear) -> torch.Tensor:
    """``down(silu(gate(n)) * up(n))`` with ``n = norm(x)`` (Llama's MLP), with the RMSNorm, the SwiGLU and
    its backward fused into quantization."""
    d = x.shape[-1]
    _check_linear(gate, d, bias_ok=False)
    _check_linear(up, d, bias_ok=False)
    if gate.out_features != up.out_features:
        raise ValueError(f"gate and up must have the same out_features, got {gate.out_features} and {up.out_features}")
    _check_linear(down, gate.out_features)
    x2d = x.reshape(-1, d)
    if x2d.dtype not in (torch.float32, BF16):
        x2d = x2d.float()
    norm_w, eps = _rms_params(norm, d, x2d)
    gx, gw, gg, gu, gd = _needs(x2d, norm_w, gate.weight, up.weight, down.weight)
    out = _RMSNormSwiGLUFn.apply(x2d.contiguous(), norm_w, eps, gate.weight, up.weight, down.weight, gate.wgrad_rht,
                                 gx or gw, gg, gu, gd, _fuse(gate), weight_cache(gate), weight_cache(up),
                                 weight_cache(down))
    return _finish(out, x, down)


__all__ = ["layernorm_linear", "layernorm_mlp", "rmsnorm_linear", "rmsnorm_swiglu_mlp"]
