"""NVFP4Linear: a drop-in nn.Linear whose three training GEMMs run on NVFP4 tensor cores.

Recipe (follows NVIDIA's published NVFP4 pretraining recipe):

    fprop  Y  = X  W^T     X: 1x16 blocks, RNE        W: 16x16 2D blocks, RNE
    dgrad  dX = dY W       dY: 1x16 blocks, stochastic rounding   W^T: same 2D blocks (transpose-exact)
    wgrad  dW = dY^T X     dY^T, X^T: 16-point random Hadamard transform along tokens, 1x16 blocks;
                           dY^T stochastically rounded

All three are mapped onto one "TN" kernel (D = A @ B^T, both operands quantized along the
contraction dim) by letting the quantizer emit transposed outputs. Accumulation is fp32 inside the
MMA; outputs are bf16 (activations/grad-inputs) or the weight's dtype (weight grads).

Each operand is read once by the fused dual quantizer, which emits both layouts it is needed in:
X -> (X for fprop, RHT(X^T) for wgrad), W -> (W, W^T), dY -> (dY for dgrad, RHT(dY^T) for wgrad).
Only the NVFP4 copies of X^T and W^T are saved for backward (~3.5x less than the bf16 activation).

``backend="emulate"`` runs the same recipe with nvfp4.reference (any device, slow): used to verify
the CUDA path and for numerics studies on machines without an sm_120 GPU.
"""
from __future__ import annotations

import torch

from nvfp4 import ops
from nvfp4 import reference as ref

ALIGN = 128  # kernel requires the contraction dim to be a multiple of 128


def _seed() -> int:
    # drawn from torch's CPU generator: reproducible under torch.manual_seed
    return int(torch.randint(0, 2**62, (1,)).item())


class _QuantizerCUDA:
    @staticmethod
    def dual(x, **kw):
        return ops.quantize_dual(x, **kw)

    @staticmethod
    def mm(a, b, out_dtype):
        return ops.gemm(a, b, out_dtype=out_dtype)


class _QuantizerEmulate:
    """Same API, implemented with the pure-PyTorch reference."""

    @staticmethod
    def dual(x, **kw):
        outs = ref.quantize_dual(x, **kw)
        return tuple(None if o is None else ops.NVFP4Tensor(o[0], o[1], o[2].reshape(1)) for o in outs)

    @staticmethod
    def mm(a, b, out_dtype):
        return ref.gemm(a.as_tuple(), b.as_tuple()).to(out_dtype)


_BACKENDS = {"cuda": _QuantizerCUDA, "emulate": _QuantizerEmulate}


def _weight_grad(param, dytq, xtq, w_dtype, backend_name: str):
    """dW = dY^T X from the NVFP4 operands. With ``param`` given (fused accumulation) and an fp32
    ``param.grad`` present, the GEMM adds into ``.grad`` in place and None is returned to autograd."""
    g = param.grad if param is not None else None
    if g is not None and g.dtype == torch.float32 and g.is_contiguous() and g.shape == param.shape:
        ops.gemm(dytq, xtq, out=g, accumulate=True)
        return None
    dw = _BACKENDS[backend_name].mm(dytq, xtq, torch.float32 if w_dtype == torch.float32 else torch.bfloat16)
    return dw.to(w_dtype)


class _NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, weight, backend_name: str, wgrad_rht: bool, need_dgrad: bool, need_wgrad: bool,
                fuse_wgrad: bool):
        be = _BACKENDS[backend_name]
        rht = _seed() if need_wgrad and wgrad_rht else None
        # X  -> rowwise for fprop, and (if wgrad) X^T along tokens, RHT, zero-padded to ALIGN tokens
        xq, xtq = be.dual(x2d, col=need_wgrad, rht_seed=rht, col_pad=ALIGN)
        # W  -> 16x16 tiles; W^T (exact transpose) for dgrad
        wq, wtq = be.dual(weight.detach(), col=need_dgrad, block_2d=True)
        y = be.mm(xq, wq, torch.bfloat16)                                   # [M, out]
        saved = []
        if need_wgrad:
            saved += [xtq.data, xtq.scales, xtq.gdec]
        if need_dgrad:
            saved += [wtq.data, wtq.scales, wtq.gdec]
        ctx.save_for_backward(*saved)
        ctx.backend_name, ctx.rht, ctx.w_dtype = backend_name, rht, weight.dtype
        ctx.need_dgrad, ctx.need_wgrad = need_dgrad, need_wgrad
        ctx.param = weight if fuse_wgrad and backend_name == "cuda" else None
        return y

    @staticmethod
    def backward(ctx, dy):
        be = _BACKENDS[ctx.backend_name]
        saved = list(ctx.saved_tensors)
        xtq = ops.NVFP4Tensor(*saved[:3]) if ctx.need_wgrad else None
        wtq = ops.NVFP4Tensor(*saved[-3:]) if ctx.need_dgrad else None
        do_dx = ctx.need_dgrad and ctx.needs_input_grad[0]
        do_dw = ctx.need_wgrad and ctx.needs_input_grad[1]
        if not (do_dx or do_dw):
            return (None,) * 7
        dy = dy.contiguous()
        if dy.dtype not in (torch.bfloat16, torch.float32):
            dy = dy.float()
        # dY -> rowwise with SR for dgrad, and dY^T with RHT + SR for wgrad, from one read
        row_seed = _seed() if do_dx else None
        col_seed = _seed() if do_dw else None
        dyq, dytq = be.dual(dy, row=do_dx, col=do_dw, row_sr_seed=row_seed, col_sr_seed=col_seed,
                            rht_seed=ctx.rht, col_pad=ALIGN)
        dx = be.mm(dyq, wtq, torch.bfloat16) if do_dx else None              # [M, in]
        # [out, in]; with fused accumulation the GEMM adds into .grad and autograd sees no gradient
        dw = _weight_grad(ctx.param, dytq, xtq, ctx.w_dtype, ctx.backend_name) if do_dw else None
        return dx, dw, None, None, None, None, None


class NVFP4Linear(torch.nn.Linear):
    """nn.Linear with NVFP4 fprop/dgrad/wgrad. in/out features must be multiples of 128.

    The bias (if any) is added in bf16 after the GEMM. Inputs are cast to bf16.

    ``fuse_wgrad_accumulation``: when ``weight.grad`` already exists (fp32), add the new weight gradient
    to it inside the wgrad GEMM instead of returning it to autograd. Saves a full read-modify-write of
    the gradient per micro-batch under gradient accumulation. Autograd then does not see that gradient,
    so per-parameter gradient hooks (e.g. DDP's) do not fire and ``torch.autograd.grad`` must not be
    used for these weights: leave it off unless the training loop only reads ``.grad``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
                 backend: str = "cuda", wgrad_rht: bool = True, fuse_wgrad_accumulation: bool = False):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if in_features % ALIGN or out_features % ALIGN:
            raise ValueError(f"NVFP4Linear needs in/out features divisible by {ALIGN}, got "
                             f"{in_features}->{out_features}")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {list(_BACKENDS)}")
        self.backend = backend
        self.wgrad_rht = wgrad_rht
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation

    @classmethod
    def from_linear(cls, lin: torch.nn.Linear, **kw) -> "NVFP4Linear":
        new = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, device=lin.weight.device,
                  dtype=lin.weight.dtype, **kw)
        with torch.no_grad():
            new.weight.copy_(lin.weight)
            if lin.bias is not None:
                new.bias.copy_(lin.bias)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous()
        grad = torch.is_grad_enabled()
        y = _NVFP4LinearFn.apply(x2d, self.weight, self.backend, self.wgrad_rht,
                                 grad and x2d.requires_grad, grad and self.weight.requires_grad,
                                 self.fuse_wgrad_accumulation)
        # Drop-in semantics: return the caller's dtype (bf16 under autocast / bf16 models).
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        y = y.to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        r = super().extra_repr() + f", backend={self.backend}, wgrad_rht={self.wgrad_rht}"
        return r + (", fuse_wgrad_accumulation=True" if self.fuse_wgrad_accumulation else "")


def convert_linears(module: torch.nn.Module, *, skip=(), **kw) -> list[str]:
    """Replace every eligible nn.Linear (dims % 128 == 0, name not starting with any ``skip`` prefix).

    Returns the names of converted modules.
    """
    converted = []
    for name, child in list(module.named_modules()):
        for cname, sub in list(child.named_children()):
            full = f"{name}.{cname}" if name else cname
            if (type(sub) is torch.nn.Linear and sub.in_features % ALIGN == 0 and sub.out_features % ALIGN == 0
                    and not any(full.startswith(s) for s in skip)):
                setattr(child, cname, NVFP4Linear.from_linear(sub, **kw))
                converted.append(full)
    return converted


def flops_per_token(in_f: int, out_f: int) -> int:
    """Training FLOPs per token for one linear layer (fprop + dgrad + wgrad)."""
    return 6 * in_f * out_f


__all__ = ["NVFP4Linear", "convert_linears", "flops_per_token"]
