"""NVFP4Linear: a drop-in nn.Linear whose three training GEMMs run on NVFP4 tensor cores.

Recipe (follows NVIDIA's published NVFP4 pretraining recipe):

    fprop  Y  = X  W^T     X: 1x16 blocks, RNE        W: 16x16 2D blocks, RNE
    dgrad  dX = dY W       dY: 1x16 blocks, stochastic rounding   W^T: same 2D blocks (transpose-exact)
    wgrad  dW = dY^T X     dY^T, X^T: 16-point random Hadamard transform along tokens, 1x16 blocks;
                           dY^T stochastically rounded

All three are mapped onto one "TN" kernel (D = A @ B^T, both operands quantized along the
contraction dim) by letting the quantizer emit transposed outputs. Accumulation is fp32 inside the
MMA; outputs are bf16 (activations/grad-inputs) or the weight's dtype (weight grads).

``backend="emulate"`` runs the same recipe with nvfp4.reference (any device, slow): used to verify
the CUDA path and for numerics studies on machines without an sm_120 GPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nvfp4 import ops
from nvfp4 import reference as ref

ALIGN = 128  # kernel requires the contraction dim to be a multiple of 128


def _seed() -> int:
    # drawn from torch's CPU generator: reproducible under torch.manual_seed
    return int(torch.randint(0, 2**62, (1,)).item())


class _QuantizerCUDA:
    @staticmethod
    def q(x, **kw):
        return ops.quantize(x, **kw)

    @staticmethod
    def mm(a, b, out_dtype):
        return ops.gemm(a, b, out_dtype=out_dtype)


class _QuantizerEmulate:
    """Same API, implemented with the pure-PyTorch reference."""

    @staticmethod
    def q(x, *, transpose=False, block_2d=False, stochastic=False, sr_seed=0, rht_seed=None):
        y = (x.T if transpose else x).float().contiguous()
        if rht_seed is not None:
            y = ref.apply_rht(y, rht_seed)
        d, s, g = ref.quantize(y, block_2d=block_2d, stochastic=stochastic, seed=sr_seed)
        return ops.NVFP4Tensor(d, s, g.reshape(1))

    @staticmethod
    def mm(a, b, out_dtype):
        return ref.gemm(a.as_tuple(), b.as_tuple()).to(out_dtype)


_BACKENDS = {"cuda": _QuantizerCUDA, "emulate": _QuantizerEmulate}


def _pad_rows(x: torch.Tensor, mult: int) -> torch.Tensor:
    r = x.shape[0]
    pad = (-r) % mult
    return F.pad(x, (0, 0, 0, pad)) if pad else x


class _NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, weight, backend_name: str, wgrad_rht: bool):
        be = _BACKENDS[backend_name]
        m = x2d.shape[0]
        xq = be.q(x2d)                                    # [M, in]   along in
        wq = be.q(weight, block_2d=True)                  # [out, in] along in, 16x16 tiles
        y = be.mm(xq, wq, torch.bfloat16)                 # [M, out]
        ctx.save_for_backward(x2d, weight)
        ctx.backend_name, ctx.wgrad_rht, ctx.m = backend_name, wgrad_rht, m
        return y

    @staticmethod
    def backward(ctx, dy):
        x2d, weight = ctx.saved_tensors
        be = _BACKENDS[ctx.backend_name]
        dy = dy.contiguous()
        if dy.dtype not in (torch.bfloat16, torch.float32):
            dy = dy.float()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            dyq = be.q(dy, stochastic=True, sr_seed=_seed())          # [M, out]  along out
            wtq = be.q(weight, transpose=True, block_2d=True)          # [in, out] along out
            dx = be.mm(dyq, wtq, torch.bfloat16)                       # [M, in]
        if ctx.needs_input_grad[1]:
            rht = _seed() if ctx.wgrad_rht else None
            dyp, xp = _pad_rows(dy, ALIGN), _pad_rows(x2d, ALIGN)      # zero rows add nothing to dW
            dytq = be.q(dyp, transpose=True, stochastic=True, sr_seed=_seed(), rht_seed=rht)  # [out, M]
            xtq = be.q(xp, transpose=True, rht_seed=rht)                                       # [in, M]
            dw = be.mm(dytq, xtq, torch.float32 if weight.dtype == torch.float32 else torch.bfloat16)
            dw = dw.to(weight.dtype)
        return dx, dw, None, None


class NVFP4Linear(torch.nn.Linear):
    """nn.Linear with NVFP4 fprop/dgrad/wgrad. in/out features must be multiples of 128.

    The bias (if any) is added in bf16 after the GEMM. Inputs are cast to bf16.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
                 backend: str = "cuda", wgrad_rht: bool = True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if in_features % ALIGN or out_features % ALIGN:
            raise ValueError(f"NVFP4Linear needs in/out features divisible by {ALIGN}, got "
                             f"{in_features}->{out_features}")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {list(_BACKENDS)}")
        self.backend = backend
        self.wgrad_rht = wgrad_rht

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
        y = _NVFP4LinearFn.apply(x2d, self.weight, self.backend, self.wgrad_rht)
        # Drop-in semantics: return the caller's dtype (bf16 under autocast / bf16 models).
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        y = y.to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", backend={self.backend}, wgrad_rht={self.wgrad_rht}"


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
