"""Transformer sub-blocks with the producer ops fused into NVFP4 quantization (CUDA backend).

    layernorm_linear(x, ln, lin)                    lin(ln(x))                       GPT / BERT-style qkv
    layernorm_mlp(x, ln, fc1, fc2, approximate)     fc2(gelu(fc1(ln(x))))            GPT / BERT-style MLP
    rmsnorm_linear(x, norm, lin)                    lin(rmsnorm(x))                  Llama-style qkv
    rmsnorm_swiglu_mlp(x, norm, gate, up, down)     down(silu(gate(n)) * up(n)),     Llama-style MLP
                                                    n = rmsnorm(x)
    gelu_mlp(x, fc1, fc2) / swiglu_mlp(x, ...)      the MLPs without a norm          post-norm blocks

The norm argument may be None (no normalization: post-norm blocks, where the norm follows the residual
add, or any plain MLP). Every linear may have a bias.

Same recipe as NVFP4Linear, but LayerNorm / RMSNorm, GELU / SwiGLU and their backwards run inside the
quantizers (ops.*_quantize_dual), so the normalized input, the activation and its input gradient are
never written to memory, and each costs no separate elementwise pass. The normalized input is
quantized once for every linear that reads it (gate and up share it). Biases of the first linear of an
MLP are added inside the activation producer (in fp32), with their gradients from column sums the
backward quantizer computes from the same read; the last linear's bias, and the residual add, run in
the GEMM's epilogue (ops.gemm(bias=, residual=)), bit-identical to adding them afterwards.

Residual connections (``x + sublayer(x)``):
  * ``residual=True`` on the MLPs returns ``x + mlp(x)``: the add happens in the down projection's
    epilogue, and in backward the residual gradient is added inside the norm backward kernel.
  * ``return_residual=True`` on the qkv functions also returns a handle of x to pass as the residual of
    the attention output projection, ``NVFP4Linear.forward(attn, residual=handle)``; its gradient then
    reaches the norm backward kernel the same way (no separate gradient-sum pass).
  * ``chunks=3`` on the qkv functions returns q, k and v as separate views of the GEMM output; in
    backward their gradients are quantized in place (no concatenation of dq, dk, dv).

Numerics differ from the unfused composition only in rounding: the fused path quantizes the fp32
norm / activation outputs directly instead of first rounding them to bf16, uses its own normalization
statistics (fp32, tree-reduced), which the backward then also uses, and quantizes an fp32 output
gradient (a residual stream in fp32) without first rounding it to bf16. Forward outputs equal the
explicit composition of the reference ops bit for bit.
"""
from __future__ import annotations

import torch

from nvfp4 import ops
from nvfp4.linear import (ALIGN, NVFP4Linear, _weight_grad, dgrad_weight_operand, draw_seeds, unflatten_output,
                          weight_cache)

BF16 = torch.bfloat16
_EPI_DTYPES = (torch.float32, BF16)


def _sr(seeds, row: bool, col: bool, rht: bool):
    """Seed keyword arguments from a [row SR, col SR, RHT] seed tensor."""
    return dict(row_sr_seed=seeds[0] if row else None, col_sr_seed=seeds[1] if col else None,
                rht_seed=seeds[2] if rht else None)


# --- the norm stage: "ln" (LayerNorm), "rms" (RMSNorm) or "none" ----------------------------------------

def _norm_quantize(kind, x2d, w, b, eps, col: bool, rht_seed, terms: int = 1):
    """(mean, rstd, Q(norm(x)), Q(norm(x)^T)) with the norm fused into the quantizer (the row output in ``terms``
    NVFP4 terms, see ops.set_fprop_terms)."""
    if kind == "ln":
        return ops.layernorm_quantize_dual(x2d, w, b, eps, col=col, rht_seed=rht_seed, col_pad=ALIGN, row_terms=terms)
    if kind == "rms":
        rstd, xq, xtq = ops.rmsnorm_quantize_dual(x2d, w, eps, col=col, rht_seed=rht_seed, col_pad=ALIGN,
                                                  row_terms=terms)
        return None, rstd, xq, xtq
    xq, xtq = ops.quantize_dual(x2d.to(BF16), col=col, rht_seed=rht_seed, col_pad=ALIGN,
                                row_terms=terms)  # as NVFP4Linear
    return None, None, xq, xtq


def _norm_backward(kind, dn, x2d, mean, rstd, w, dres, nin):
    """(dx, dw, db) of the norm stage from dn, the fp32 gradient of its output; dres (the gradient reaching x
    through a residual connection) is added to dx. nin: needs_input_grad of (x, norm weight, norm bias)."""
    if kind == "none":
        dx = dn if dres is None else dn + dres.float()
        return dx, None, None
    return ops.norm_backward(dn, x2d, mean, rstd, w, dres=dres, need_dx=nin[0], need_dw=nin[1],
                             need_db=nin[2] and kind == "ln")


def _dgrad_to_input(kind, dq, wtq, x2d, mean, rstd, w, dres, nin):
    """Input-side gradients (dx, dw, db) of a norm stage followed by one linear, from that linear's quantized
    output gradient dq: no norm -> dx straight from the dgrad GEMM with dres added in its epilogue."""
    if kind == "none":
        return (ops.gemm(dq, wtq, BF16, residual=dres) if dres is not None else ops.gemm(dq, wtq, BF16)), None, None
    return _norm_backward(kind, ops.gemm(dq, wtq, torch.float32), x2d, mean, rstd, w, dres, nin)


def _out_gemm(aq, wq, dt, bias, residual):
    """The last GEMM of a sub-block in the output dtype dt: the bf16 GEMM output cast to dt, plus the bias
    and the residual, both in the epilogue (bit-identical to adding them afterwards)."""
    if bias is None and residual is None:
        y = ops.gemm(aq, wq, BF16)
        return y if dt == BF16 else y.to(dt)
    return ops.gemm(aq, wq, dt, bias=bias, residual=residual)


def _quantize_out_grad(dout, row: bool, col: bool, bias_grad: bool, seeds, rht: bool):
    """(dQ, dQ^T, bias grad) of an output gradient (one tensor or a list of chunks, bf16 or fp32)."""
    parts = [g.reshape(-1, g.shape[-1]) for g in (dout if isinstance(dout, (list, tuple)) else [dout])]
    parts = [p if p.dtype in _EPI_DTYPES else p.float() for p in parts]
    if not (row or col):  # only the bias gradient
        return None, None, torch.cat([p.float().sum(0) for p in parts]) if bias_grad else None
    if len({p.dtype for p in parts}) > 1:
        parts = [p.float() for p in parts]
    outs = ops.quantize_dual(parts if len(parts) > 1 else parts[0], row=row, col=col, bias_grad=bias_grad,
                             col_pad=ALIGN, **_sr(seeds, row, col, rht))
    return (outs[0], outs[1], outs[2] if bias_grad else None)


def _cast(t, like):
    return None if t is None or like is None else t.to(like.dtype)


class _NormLinearFn(torch.autograd.Function):
    """lin(norm(x)) as `chunks` views (+ x itself as a residual handle). cfg: (kind, eps, wgrad_rht, need_dn,
    need_w, need_b, fuse_wgrad, chunks, out_dtype, return_residual)."""

    @staticmethod
    def forward(ctx, x2d, norm_w, norm_b, weight, bias, wcache, cfg):
        kind, eps, rht_on, need_dn, need_w, need_b, fuse, chunks, dt, ret_res = cfg
        seeds = draw_seeds(3, x2d.device) if need_dn or need_w or need_b else None  # [dY row SR, dY^T SR, RHT]
        use_rht = need_w and rht_on
        T = ops.fprop_terms()
        mean, rstd, xq, xtq = _norm_quantize(kind, x2d, norm_w, norm_b, eps, need_w, seeds[2] if use_rht else None, T)
        wq, wtq = dgrad_weight_operand("cuda", weight, need_dn, wcache, seeds[0] if need_dn else None, terms=T)
        y = _out_gemm(xq, wq, dt, bias, None)
        ctx.save_for_backward(x2d if kind != "none" else None, norm_w, mean, rstd, seeds,
                              *(xtq.as_storage() if need_w else ()), *(wtq.as_storage() if need_dn else ()))
        ctx.cfg, ctx.use_rht, ctx.w_dtype, ctx.has_bias = cfg, use_rht, weight.dtype, bias is not None
        ctx.param = weight if fuse else None
        outs = y.split(y.shape[1] // chunks, 1) if chunks > 1 else (y,)
        return (*outs, x2d) if ret_res else tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        kind, eps, rht_on, need_dn, need_w, need_b, fuse, chunks, dt, ret_res = ctx.cfg
        x2d, norm_w, mean, rstd, seeds, *rest = ctx.saved_tensors
        xtq = ops.NVFP4Tensor(*rest[:3]) if need_w else None
        wtq = ops.NVFP4Tensor(*rest[-3:]) if need_dn else None
        dys, dres = grads[:chunks], (grads[chunks] if ret_res else None)
        nin = ctx.needs_input_grad
        do_norm = need_dn and any(nin[:3])
        do_w = need_w and nin[3]
        do_b = ctx.has_bias and nin[4]
        out = [None] * 7
        if dres is not None and nin[0] and not do_norm:
            out[0] = dres
        if not (do_norm or do_w or do_b):
            return tuple(out)
        dyq, dytq, db = _quantize_out_grad(list(dys), do_norm, do_w, do_b, seeds, ctx.use_rht)
        del dys
        if do_b:
            out[4] = db
        if do_w:
            out[3] = _weight_grad(ctx, ctx.param, dytq, xtq, ctx.w_dtype, "cuda")
        del dytq, xtq
        if do_norm:
            out[0], out[1], out[2] = _dgrad_to_input(kind, dyq, wtq, x2d, mean, rstd, norm_w,
                                                     dres if nin[0] else None, nin)
        return tuple(out)


class _GeluMLPFn(torch.autograd.Function):
    """[x +] fc2(gelu(fc1(norm(x)) + b1) + b2). cfg: (kind, eps, wgrad_rht, approximate, need_dn, need_w1,
    need_b1, need_w2, need_b2, fuse_wgrad, out_dtype, residual)."""

    @staticmethod
    def forward(ctx, x2d, norm_w, norm_b, w1, b1, w2, b2, wc1, wc2, cfg):
        kind, eps, rht_on, approximate, need_dn, need_w1, need_b1, need_w2, need_b2, fuse, dt, residual = cfg
        # [dH row SR, dH^T SR, RHT of fc1's wgrad, dOut row SR, dOut^T SR, RHT of fc2's wgrad]
        need_dh = need_dn or need_w1 or need_b1  # the gradient has to flow back through fc2 and the GELU
        seeds = draw_seeds(6, x2d.device) if need_dh or need_w2 or need_b2 else None
        rht1, rht2 = need_w1 and rht_on, need_w2 and rht_on
        T = ops.fprop_terms()
        mean, rstd, xq, xtq = _norm_quantize(kind, x2d, norm_w, norm_b, eps, need_w1, seeds[2] if rht1 else None, T)
        w1q, w1tq = dgrad_weight_operand("cuda", w1, need_dn, wc1, seeds[0] if need_dn else None, terms=T)
        h = ops.gemm(xq, w1q, BF16)                                       # fc1 output (GELU input, without bias)
        gq, gtq = ops.gelu_quantize_dual(h, bias=b1, approximate=approximate, col=need_w2,
                                         rht_seed=seeds[5] if rht2 else None, col_pad=ALIGN, row_terms=T)
        w2q, w2tq = dgrad_weight_operand("cuda", w2, need_dh, wc2, seeds[3] if need_dh else None, terms=T)
        out = _out_gemm(gq, w2q, dt, b2, x2d if residual else None)
        saved = [x2d if kind != "none" else None, norm_w, mean, rstd, h, seeds, b1]
        for flag, q in ((need_w1, xtq), (need_dn, w1tq), (need_w2, gtq), (need_dh, w2tq)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.rht1, ctx.rht2 = cfg, rht1, rht2
        ctx.dtypes = (w1.dtype, w2.dtype)
        ctx.params = (w1, w2) if fuse else (None, None)
        ctx.has_b2 = b2 is not None
        return out

    @staticmethod
    def backward(ctx, dout):
        kind, eps, rht_on, approximate, need_dn, need_w1, need_b1, need_w2, need_b2, fuse, dt, residual = ctx.cfg
        x2d, norm_w, mean, rstd, h, seeds, b1, *rest = ctx.saved_tensors
        need_dh = need_dn or need_w1 or need_b1
        qs = {}
        for name, flag in (("xtq", need_w1), ("w1tq", need_dn), ("gtq", need_w2), ("w2tq", need_dh)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_norm = need_dn and any(nin[:3])
        do_w1, do_b1 = need_w1 and nin[3], b1 is not None and nin[4]
        do_w2, do_b2 = need_w2 and nin[5], ctx.has_b2 and nin[6]
        do_dg = do_norm or do_w1 or do_b1
        dout = dout.contiguous()  # read by the quantizer and (as dres) by the norm backward
        dres = dout if residual and nin[0] else None
        grads = [None] * 10
        if dres is not None and not do_norm:
            grads[0] = dres
        if not (do_dg or do_w2 or do_b2):
            return tuple(grads)
        doq, dotq, db2 = _quantize_out_grad(dout, do_dg, do_w2, do_b2, seeds[3:], ctx.rht2)
        # Temporaries are dropped as soon as they are consumed: everything referenced inside one backward
        # call stays alive until it returns, unlike between separate autograd nodes.
        if do_b2:
            grads[6] = db2
        if do_w2:
            grads[5] = _weight_grad(ctx, ctx.params[1], dotq, qs.pop("gtq"), ctx.dtypes[1], "cuda")
        del dotq
        if do_dg:
            dg = ops.gemm(doq, qs.pop("w2tq"), BF16)                      # gradient w.r.t. gelu(h + b1)
            del doq
            outs = ops.gelu_backward_quantize_dual(dg, h, bias=b1, approximate=approximate, bias_grad=do_b1,
                                                   row=do_norm or not do_w1, col=do_w1, col_pad=ALIGN,
                                                   **_sr(seeds, do_norm, do_w1, ctx.rht1))
            dhq, dhtq = outs[0], outs[1]
            if do_b1:
                grads[4] = outs[2]
            del dg, h, outs
            if do_w1:
                grads[3] = _weight_grad(ctx, ctx.params[0], dhtq, qs.pop("xtq"), ctx.dtypes[0], "cuda")
            del dhtq
            if do_norm:
                grads[0], grads[1], grads[2] = _dgrad_to_input(kind, dhq, qs.pop("w1tq"), x2d, mean, rstd, norm_w,
                                                               dres, nin)
        return tuple(grads)


class _SwiGLUMLPFn(torch.autograd.Function):
    """[x +] down(silu(gate(n) + bg) * (up(n) + bu)) + bd, n = norm(x). cfg: (kind, eps, wgrad_rht, need_dn,
    need_wg, need_wu, need_wd, need_bg, need_bu, need_bd, fuse_wgrad, out_dtype, residual)."""

    @staticmethod
    def forward(ctx, x2d, norm_w, norm_b, wg, bg, wu, bu, wd, bd, wcg, wcu, wcd, cfg):
        kind, eps, rht_on, need_dn, need_wg, need_wu, need_wd, need_bg, need_bu, need_bd, fuse, dt, residual = cfg
        # [dG row SR, dG^T SR, RHT of x^T (gate and up wgrad), dOut row SR, dOut^T SR, RHT of the down
        #  projection's wgrad, dU row SR, dU^T SR]
        need_xt = need_wg or need_wu
        need_da = need_dn or need_xt or need_bg or need_bu  # the gradient flows back through down and SwiGLU
        seeds = draw_seeds(8, x2d.device) if need_da or need_wd or need_bd else None
        rht_x, rht_a = need_xt and rht_on, need_wd and rht_on
        T = ops.fprop_terms()
        mean, rstd, xq, xtq = _norm_quantize(kind, x2d, norm_w, norm_b, eps, need_xt, seeds[2] if rht_x else None, T)
        wgq, wgtq = dgrad_weight_operand("cuda", wg, need_dn, wcg, seeds[0] if need_dn else None, terms=T)
        wuq, wutq = dgrad_weight_operand("cuda", wu, need_dn, wcu, seeds[6] if need_dn else None, terms=T)
        g = ops.gemm(xq, wgq, BF16)                                    # gate projection (without bias)
        u = ops.gemm(xq, wuq, BF16)                                    # up projection (without bias)
        aq, atq = ops.swiglu_quantize_dual(g, u, bias_g=bg, bias_u=bu, col=need_wd,
                                           rht_seed=seeds[5] if rht_a else None, col_pad=ALIGN, row_terms=T)
        wdq, wdtq = dgrad_weight_operand("cuda", wd, need_da, wcd, seeds[3] if need_da else None, terms=T)
        out = _out_gemm(aq, wdq, dt, bd, x2d if residual else None)
        saved = [x2d if kind != "none" else None, norm_w, mean, rstd, g, u, seeds, bg, bu]
        for flag, q in ((need_xt, xtq), (need_dn, wgtq), (need_dn, wutq), (need_wd, atq), (need_da, wdtq)):
            if flag:
                saved += q.as_storage()
        ctx.save_for_backward(*saved)
        ctx.cfg, ctx.rht_x, ctx.rht_a = cfg, rht_x, rht_a
        ctx.dtypes = (wg.dtype, wu.dtype, wd.dtype)
        ctx.params = (wg, wu, wd) if fuse else (None, None, None)
        ctx.has_bd, ctx.x_dtype = bd is not None, x2d.dtype
        return out

    @staticmethod
    def backward(ctx, dout):
        kind, eps, rht_on, need_dn, need_wg, need_wu, need_wd, need_bg, need_bu, need_bd, fuse, dt, residual = ctx.cfg
        x2d, norm_w, mean, rstd, g, u, seeds, bg, bu, *rest = ctx.saved_tensors
        need_xt = need_wg or need_wu
        need_da = need_dn or need_xt or need_bg or need_bu
        qs = {}
        for name, flag in (("xtq", need_xt), ("wgtq", need_dn), ("wutq", need_dn), ("atq", need_wd),
                           ("wdtq", need_da)):
            if flag:
                qs[name], rest = ops.NVFP4Tensor(*rest[:3]), rest[3:]
        nin = ctx.needs_input_grad
        do_norm = need_dn and any(nin[:3])
        do_wg, do_bg = need_xt and nin[3], bg is not None and nin[4]
        do_wu, do_bu = need_xt and nin[5], bu is not None and nin[6]
        do_wd, do_bd = need_wd and nin[7], ctx.has_bd and nin[8]
        do_da = do_norm or do_wg or do_wu or do_bg or do_bu
        dout = dout.contiguous()  # read by the quantizer and (as dres) by the norm backward
        dres = dout if residual and nin[0] else None
        grads = [None] * 13
        if dres is not None and not do_norm:
            grads[0] = dres
        if not (do_da or do_wd or do_bd):
            return tuple(grads)
        doq, dotq, dbd = _quantize_out_grad(dout, do_da, do_wd, do_bd, seeds[3:6], ctx.rht_a)
        # Temporaries are dropped as soon as they are consumed (see _GeluMLPFn.backward).
        if do_bd:
            grads[8] = dbd
        if do_wd:
            grads[7] = _weight_grad(ctx, ctx.params[2], dotq, qs.pop("atq"), ctx.dtypes[2], "cuda")
        del dotq
        if not do_da:
            return tuple(grads)
        da = ops.gemm(doq, qs.pop("wdtq"), BF16)                       # gradient w.r.t. the SwiGLU output
        del doq
        dn = None
        for k, (do_w, do_b, row_s, col_s) in enumerate(((do_wg, do_bg, 0, 1), (do_wu, do_bu, 6, 7))):  # gate, up
            if not (do_norm or do_w or do_b):
                continue
            outs = ops.swiglu_backward_quantize_dual(
                da, g, u, gate=k == 0, bias_g=bg, bias_u=bu, bias_grad=do_b, row=do_norm or not do_w, col=do_w,
                col_pad=ALIGN, row_sr_seed=seeds[row_s] if do_norm else None,
                col_sr_seed=seeds[col_s] if do_w else None, rht_seed=seeds[2] if do_w and ctx.rht_x else None)
            dq, dtq = outs[0], outs[1]
            if do_b:
                grads[4 + 2 * k] = outs[2]
            if do_w:  # dW = dG^T x (dU^T x): both operands carry x^T's RHT
                grads[3 + 2 * k] = _weight_grad(ctx, ctx.params[k], dtq, qs["xtq"], ctx.dtypes[k], "cuda")
            del dtq, outs
            if do_norm:  # d(normalized x) = dG @ Wg + dU @ Wu, accumulated in fp32
                wt = qs.pop("wgtq" if k == 0 else "wutq")
                dn = ops.gemm(dq, wt, torch.float32) if dn is None else ops.gemm(dq, wt, out=dn, accumulate=True)
            del dq
        del da, g, u
        qs.pop("xtq", None)
        if do_norm:
            grads[0], grads[1], grads[2] = _norm_backward(kind, dn, x2d, mean, rstd, norm_w, dres, nin)
            if kind == "none" and grads[0] is not None:
                grads[0] = grads[0].to(ctx.x_dtype)
        return tuple(grads)


# --- module plumbing -----------------------------------------------------------------------------------

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


def _norm_args(kind, norm, d, x2d):
    """(kind, weight, bias, eps) of the norm stage; kind 'none' if norm is None."""
    if norm is None:
        return "none", None, None, 0.0
    if kind == "ln":
        w, b = _ln_params(norm, d, x2d.device)
        return "ln", w, b, float(norm.eps)
    w, eps = _rms_params(norm, d, x2d)
    return "rms", w, None, eps


def _check_linear(lin, d_in: int):
    if not isinstance(lin, NVFP4Linear) or lin.backend != "cuda":
        raise TypeError("fused sub-blocks need NVFP4Linear modules with backend='cuda'")
    if lin.in_features != d_in:
        raise ValueError(f"expected in_features={d_in}, got {lin.in_features}")


def _out_dtype(x):
    return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype


def _fuse(lin: NVFP4Linear) -> bool:
    return lin.fuse_wgrad_accumulation and not torch.compiler.is_compiling()


def _req(t) -> bool:
    return t is not None and torch.is_grad_enabled() and t.requires_grad


def _flat(x: torch.Tensor):
    d = x.shape[-1]
    x2d = x.reshape(-1, d)
    if x2d.dtype not in _EPI_DTYPES:
        x2d = x2d.float()
    return d, x2d.contiguous()


def _epilogue(x2d, dt, residual: bool):
    """Whether the last GEMM can take the bias (and the residual x2d) in its epilogue."""
    return dt in _EPI_DTYPES and (not residual or x2d.dtype in _EPI_DTYPES)


def _finish(y, x, lin, bias_done: bool, residual):
    """[M, N] (already in the output dtype) -> [*lead, N], adding the bias / residual where the epilogue
    could not (fp16 outputs)."""
    if not bias_done:
        y = y + lin.bias.to(y.dtype)
    y = unflatten_output(y, x.shape[:-1], lin.out_features)
    return y if residual is None else residual + y


def _norm(kind, x, norm):
    """The unfused normalization (ops.set_high_precision): what the module computes in the caller's dtype."""
    return x if norm is None else norm(x)


def _norm_linear(kind, x, norm, lin, chunks, return_residual):
    d, x2d = _flat(x)
    _check_linear(lin, d)
    if chunks < 1 or lin.out_features % (chunks * 16):
        raise ValueError(f"out_features ({lin.out_features}) must split into {chunks} chunks of multiples of 16")
    if ops.high_precision():  # residual handle = x itself (NVFP4Linear adds it outside)
        y = torch.nn.Linear.forward(lin, _norm(kind, x, norm))
        ys = (y,) if chunks == 1 else y.split(lin.out_features // chunks, dim=-1)
        if chunks == 1:
            return (y, x) if return_residual else y
        return (*ys, x) if return_residual else ys
    kind, w, b, eps = _norm_args(kind, norm, d, x2d)
    dt = _out_dtype(x)
    bias = lin.bias if _epilogue(x2d, dt, False) else None  # else (fp16 output) added below
    need_dn = _req(x2d) or _req(w) or _req(b)
    cfg = (kind, eps, lin.wgrad_rht, need_dn, _req(lin.weight), _req(bias), _fuse(lin), chunks, dt,
           return_residual)
    outs = _NormLinearFn.apply(x2d, w, b, lin.weight, bias, weight_cache(lin), cfg)
    ys, res = outs[:chunks], (outs[chunks].view(x.shape) if return_residual else None)
    lead, n = x.shape[:-1], lin.out_features // chunks
    if bias is None and lin.bias is not None:
        ys = [y + lin.bias[i * n:(i + 1) * n].to(dt) for i, y in enumerate(ys)]
    if chunks == 1:
        y = unflatten_output(ys[0], lead, n) if ys[0]._base is None else ys[0].view(*lead, n)
        return (y, res) if return_residual else y
    ys = tuple(y.view(*lead, n) for y in ys)
    return (*ys, res) if return_residual else ys


def layernorm_linear(x: torch.Tensor, ln: torch.nn.LayerNorm | None, lin: NVFP4Linear, *, chunks: int = 1,
                     return_residual: bool = False):
    """``lin(ln(x))`` with the LayerNorm fused into the quantization of the linear's input.

    ``chunks=k`` returns the output split into k views along the last dim (q, k, v of a fused qkv
    projection; their gradients are quantized in place, without being concatenated). With
    ``return_residual=True`` the result is followed by a handle of x: pass it as the residual of the block's
    next residual add (``NVFP4Linear.forward(..., residual=handle)``) and the residual gradient is added
    inside this function's norm backward. ``ln=None``: no normalization.
    """
    return _norm_linear("ln", x, ln, lin, chunks, return_residual)


def rmsnorm_linear(x: torch.Tensor, norm: torch.nn.Module | None, lin: NVFP4Linear, *, chunks: int = 1,
                   return_residual: bool = False):
    """``lin(norm(x))`` with the RMSNorm fused into the quantization of the linear's input (``chunks`` and
    ``return_residual`` as in layernorm_linear)."""
    return _norm_linear("rms", x, norm, lin, chunks, return_residual)


def _mlp_gelu(kind, x, norm, fc1, fc2, approximate, residual):
    if approximate not in ("tanh", "none"):
        raise ValueError(f"approximate must be 'tanh' or 'none', got {approximate!r}")
    d, x2d = _flat(x)
    _check_linear(fc1, d)
    _check_linear(fc2, fc1.out_features)
    if residual and fc2.out_features != d:
        raise ValueError(f"residual=True needs fc2.out_features == {d}")
    if ops.high_precision():
        h = torch.nn.functional.gelu(torch.nn.Linear.forward(fc1, _norm(kind, x, norm)), approximate=approximate)
        y = torch.nn.Linear.forward(fc2, h)
        return x + y if residual else y
    kind, w, b, eps = _norm_args(kind, norm, d, x2d)
    dt = _out_dtype(x)
    epi = _epilogue(x2d, dt, residual)
    b2 = fc2.bias if epi else None
    need_dn = _req(x2d) or _req(w) or _req(b)
    cfg = (kind, eps, fc1.wgrad_rht, approximate, need_dn, _req(fc1.weight), _req(fc1.bias), _req(fc2.weight),
           _req(b2), _fuse(fc1), dt, residual and epi)
    out = _GeluMLPFn.apply(x2d, w, b, fc1.weight, fc1.bias, fc2.weight, b2, weight_cache(fc1), weight_cache(fc2),
                           cfg)
    return _finish(out, x, fc2, b2 is not None or fc2.bias is None, None if epi or not residual else x)


def layernorm_mlp(x: torch.Tensor, ln: torch.nn.LayerNorm | None, fc1: NVFP4Linear, fc2: NVFP4Linear,
                  approximate: str = "tanh", *, residual: bool = False) -> torch.Tensor:
    """``fc2(gelu(fc1(ln(x))))`` with LayerNorm, GELU (``approximate="tanh"`` or ``"none"``, as in
    torch.nn.functional.gelu) and the GELU backward fused into quantization. ``residual=True`` returns
    ``x + fc2(...)`` with the add fused (see the module docstring); ``ln=None``: no normalization."""
    return _mlp_gelu("ln", x, ln, fc1, fc2, approximate, residual)


def gelu_mlp(x: torch.Tensor, fc1: NVFP4Linear, fc2: NVFP4Linear, approximate: str = "tanh", *,
             residual: bool = False) -> torch.Tensor:
    """``fc2(gelu(fc1(x)))`` (``[x +]``): layernorm_mlp without the norm, e.g. for post-norm blocks."""
    return _mlp_gelu("ln", x, None, fc1, fc2, approximate, residual)


def _mlp_swiglu(kind, x, norm, gate, up, down, residual):
    d, x2d = _flat(x)
    _check_linear(gate, d)
    _check_linear(up, d)
    if gate.out_features != up.out_features:
        raise ValueError(f"gate and up must have the same out_features, got {gate.out_features} and {up.out_features}")
    _check_linear(down, gate.out_features)
    if residual and down.out_features != d:
        raise ValueError(f"residual=True needs down.out_features == {d}")
    if ops.high_precision():
        n = _norm(kind, x, norm)
        h = torch.nn.functional.silu(torch.nn.Linear.forward(gate, n)) * torch.nn.Linear.forward(up, n)
        y = torch.nn.Linear.forward(down, h)
        return x + y if residual else y
    kind, w, b, eps = _norm_args(kind, norm, d, x2d)
    dt = _out_dtype(x)
    epi = _epilogue(x2d, dt, residual)
    bd = down.bias if epi else None
    need_dn = _req(x2d) or _req(w) or _req(b)
    cfg = (kind, eps, gate.wgrad_rht, need_dn, _req(gate.weight), _req(up.weight), _req(down.weight),
           _req(gate.bias), _req(up.bias), _req(bd), _fuse(gate), dt, residual and epi)
    out = _SwiGLUMLPFn.apply(x2d, w, b, gate.weight, gate.bias, up.weight, up.bias, down.weight, bd,
                             weight_cache(gate), weight_cache(up), weight_cache(down), cfg)
    return _finish(out, x, down, bd is not None or down.bias is None, None if epi or not residual else x)


def rmsnorm_swiglu_mlp(x: torch.Tensor, norm: torch.nn.Module | None, gate: NVFP4Linear, up: NVFP4Linear,
                       down: NVFP4Linear, *, residual: bool = False) -> torch.Tensor:
    """``down(silu(gate(n)) * up(n))`` with ``n = norm(x)`` (Llama's MLP), with the RMSNorm, the SwiGLU and
    its backward fused into quantization. ``residual=True`` returns ``x + down(...)`` with the add fused;
    ``norm=None``: no normalization."""
    return _mlp_swiglu("rms", x, norm, gate, up, down, residual)


def swiglu_mlp(x: torch.Tensor, gate: NVFP4Linear, up: NVFP4Linear, down: NVFP4Linear, *,
               residual: bool = False) -> torch.Tensor:
    """``down(silu(gate(x)) * up(x))`` (``[x +]``): rmsnorm_swiglu_mlp without the norm."""
    return _mlp_swiglu("rms", x, None, gate, up, down, residual)


__all__ = ["layernorm_linear", "layernorm_mlp", "rmsnorm_linear", "rmsnorm_swiglu_mlp", "gelu_mlp", "swiglu_mlp"]
