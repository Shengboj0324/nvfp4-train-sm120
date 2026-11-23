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
from torch.optim.optimizer import register_optimizer_step_post_hook

from nvfp4 import ops
from nvfp4 import reference as ref

ALIGN = 128  # kernel requires the contraction dim to be a multiple of 128


def unflatten_output(y: torch.Tensor, lead, n: int) -> torch.Tensor:
    """[M, n] -> [*lead, n] without making the result a view, like nn.Linear's output (F.linear does
    the same through matmul). FSDP2 warns about modules returning views: an in-place op on one would
    bypass its pre-backward hook. ``y`` is a fresh contiguous tensor nothing else references."""
    return torch.ops.aten._unsafe_view(y, (*lead, n))


def draw_seeds(n: int, device) -> torch.Tensor:
    """``n`` random seeds as an int64 tensor on ``device``, from torch's generator for that device.

    The kernels read them from device memory, so drawing them needs no host synchronization, they are
    reproducible under ``torch.manual_seed`` (and restored by activation checkpointing), a captured
    CUDA graph draws fresh ones on every replay, and torch.compile traces them like any random op.
    """
    return torch.randint(0, 2**62, (n,), device=device, dtype=torch.int64)


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
        kw = {k: int(v) if isinstance(v, torch.Tensor) else v for k, v in kw.items()}  # seed tensors -> ints
        outs = ref.quantize_dual(x, **kw)
        return tuple(None if o is None else ops.NVFP4Tensor(o[0], o[1], o[2].reshape(1)) for o in outs)

    @staticmethod
    def mm(a, b, out_dtype):
        return ref.gemm(a.as_tuple(), b.as_tuple()).to(out_dtype)


_BACKENDS = {"cuda": _QuantizerCUDA, "emulate": _QuantizerEmulate}


class WeightQuantCache:
    """The NVFP4 copies (W, and W^T for dgrad) of one weight, reused while the weight is unchanged.

    Weights are quantized with deterministic 16x16 blocks (no RHT, no stochastic rounding), so a cached
    copy is bit-identical to quantizing again. Under gradient accumulation the weights don't change
    between the micro-batches of an optimizer step, and every micro-batch after the first skips the
    weight quantizer.

    A cached copy is used only if, since it was made,
      * no ``torch.optim`` optimizer has stepped (any optimizer, any subclass: a global step hook
        counts steps; needed because fused optimizers such as ``AdamW(fused=True)`` update parameters
        without bumping their version counter), and
      * the weight's storage and autograd version counter are unchanged (catches ``copy_`` / in-place
        ops under ``no_grad``, ``load_state_dict``, ``.to()``, assigning ``param.data``).
    Not noticed: in-place writes through ``param.data`` (``p.data.add_(...)``) outside an optimizer
    step, and weights changed by replaying a CUDA graph that a forward of this layer was not part of.
    After those, call ``invalidate_weight_caches(model)``. Never used for parameters that are not
    leaves (FSDP1's flat-parameter views) or under torch.compile; a layer seen during CUDA-graph
    capture turns its cache off for good (replays change weights without running any Python).
    """

    __slots__ = ("key", "wq", "wtq", "hits", "misses", "bypass")

    def __init__(self):
        self.key = self.wq = self.wtq = None
        self.hits = self.misses = 0
        self.bypass = False

    def clear(self) -> None:
        self.key = self.wq = self.wtq = None

    # copies / pickles of a module start with an empty cache (no tensors serialized)
    def __deepcopy__(self, memo):
        return WeightQuantCache()

    def __reduce__(self):
        return (WeightQuantCache, ())


_optimizer_steps = [0]  # bumped after every torch.optim optimizer step (see WeightQuantCache)


def _count_optimizer_step(optimizer, args, kwargs):
    _optimizer_steps[0] += 1


register_optimizer_step_post_hook(_count_optimizer_step)


def quantized_weight(weight: torch.Tensor, need_t: bool, backend_name: str, cache: WeightQuantCache | None):
    """(W, W^T or None) in NVFP4 with 16x16 blocks, from ``cache`` when it is still valid."""
    be = _BACKENDS[backend_name]
    if cache is not None and not cache.bypass and torch.cuda.is_current_stream_capturing():
        cache.clear()
        cache.bypass = True
    if cache is None or cache.bypass or backend_name != "cuda" or not weight.is_leaf:
        return be.dual(weight.detach(), col=need_t, block_2d=True)
    key = (_optimizer_steps[0], weight.data_ptr(), weight._version, weight.dtype, tuple(weight.shape),
           weight.device)
    if cache.key == key and (cache.wtq is not None or not need_t):
        cache.hits += 1
        return cache.wq, (cache.wtq if need_t else None)
    wq, wtq = be.dual(weight.detach(), col=need_t, block_2d=True)
    cache.key, cache.wq, cache.wtq = key, wq, wtq
    cache.misses += 1
    return wq, wtq


def weight_cache(lin) -> WeightQuantCache | None:
    """The layer's cache if caching is enabled (never while torch.compile traces)."""
    if not lin.cache_weight_quantization or torch.compiler.is_compiling():
        return None
    return lin.weight_quant_cache


def invalidate_weight_caches(module: torch.nn.Module) -> int:
    """Drop the cached NVFP4 weights of every NVFP4Linear in ``module`` (needed only after in-place writes
    through ``param.data``; see WeightQuantCache). Returns the number of layers."""
    n = 0
    for m in module.modules():
        if isinstance(m, NVFP4Linear):
            m.weight_quant_cache.clear()
            n += 1
    return n


_will_execute = getattr(torch._C, "_will_engine_execute_node", None)


def _accumulation_target(ctx, param):
    """The existing fp32 ``param.grad`` that the wgrad GEMM may add into, or None.

    Fused accumulation is only used when this backward pass will run the parameter's AccumulateGrad
    node (``loss.backward()``, not ``torch.autograd.grad`` or ``backward(inputs=...)`` excluding it) and
    the parameter is a leaf, so the result can be handed to that node (see _weight_grad).
    """
    if param is None or _will_execute is None or not param.is_leaf:
        return None  # e.g. FSDP1's views into its flat parameter: autograd accumulates into the flat one
    g = param.grad
    if (g is None or g.dtype != torch.float32 or not g.is_contiguous() or g.shape != param.shape
            or g.device != param.device):
        return None
    # the parameter's AccumulateGrad node among this node's inputs (absent if param is not a leaf)
    node = next((n for n, _ in ctx.next_functions if getattr(n, "variable", None) is param), None)
    if node is None:
        return None
    try:
        if not _will_execute(node):
            return None
    except RuntimeError:  # raised inside torch.autograd.grad(), which never accumulates into .grad
        return None
    return g


def _weight_grad(ctx, param, dytq, xtq, w_dtype, backend_name: str):
    """dW = dY^T X from the NVFP4 operands, as the weight gradient ``ctx``'s backward returns.

    Fused accumulation (``param`` given and an fp32 ``param.grad`` present): the GEMM adds the new
    gradient into ``.grad`` in place (no separate read-modify-write), ``.grad`` is detached from the
    parameter and the accumulated tensor is returned. Autograd's AccumulateGrad then finds no
    ``.grad`` and adopts the returned tensor as is (no copy), so every gradient hook fires as usual:
    DDP / FSDP bucketing, ``register_post_accumulate_grad_hook``, etc. (A tensor hook on the weight sees
    the accumulated gradient rather than this micro-batch's contribution.)
    """
    g = _accumulation_target(ctx, param)
    if g is not None:
        ops.gemm(dytq, xtq, out=g, accumulate=True)
        param.grad = None
        return g
    dw = _BACKENDS[backend_name].mm(dytq, xtq, torch.float32 if w_dtype == torch.float32 else torch.bfloat16)
    return dw.to(w_dtype)


class _NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d, weight, backend_name: str, wgrad_rht: bool, need_dgrad: bool, need_wgrad: bool,
                fuse_wgrad: bool, wcache):
        be = _BACKENDS[backend_name]
        # [dY row SR, dY^T SR, RHT]: drawn once per call; the backward reuses the RHT seed of X^T
        seeds = draw_seeds(3, x2d.device) if need_dgrad or need_wgrad else None
        rht = seeds[2] if need_wgrad and wgrad_rht else None
        # X  -> rowwise for fprop, and (if wgrad) X^T along tokens, RHT, zero-padded to ALIGN tokens
        xq, xtq = be.dual(x2d, col=need_wgrad, rht_seed=rht, col_pad=ALIGN)
        # W  -> 16x16 tiles; W^T (exact transpose) for dgrad
        wq, wtq = quantized_weight(weight, need_dgrad, backend_name, wcache)
        y = be.mm(xq, wq, torch.bfloat16)                                   # [M, out]
        saved = [seeds] if seeds is not None else []
        if need_wgrad:
            saved += xtq.as_storage()
        if need_dgrad:
            saved += wtq.as_storage()
        ctx.save_for_backward(*saved)
        ctx.backend_name, ctx.use_rht, ctx.w_dtype = backend_name, rht is not None, weight.dtype
        ctx.need_dgrad, ctx.need_wgrad = need_dgrad, need_wgrad
        ctx.param = weight if fuse_wgrad and backend_name == "cuda" else None
        return y

    @staticmethod
    def backward(ctx, dy):
        be = _BACKENDS[ctx.backend_name]
        saved = list(ctx.saved_tensors)
        do_dx = ctx.need_dgrad and ctx.needs_input_grad[0]
        do_dw = ctx.need_wgrad and ctx.needs_input_grad[1]
        if not (do_dx or do_dw):
            return (None,) * 8
        seeds = saved[0]
        xtq = ops.NVFP4Tensor(*saved[1:4]) if ctx.need_wgrad else None
        wtq = ops.NVFP4Tensor(*saved[-3:]) if ctx.need_dgrad else None
        dy = dy.contiguous()
        if dy.dtype not in (torch.bfloat16, torch.float32):
            dy = dy.float()
        # dY -> rowwise with SR for dgrad, and dY^T with RHT + SR for wgrad, from one read
        dyq, dytq = be.dual(dy, row=do_dx, col=do_dw, row_sr_seed=seeds[0] if do_dx else None,
                            col_sr_seed=seeds[1] if do_dw else None, rht_seed=seeds[2] if ctx.use_rht else None,
                            col_pad=ALIGN)
        dx = be.mm(dyq, wtq, torch.bfloat16) if do_dx else None              # [M, in]
        dw = _weight_grad(ctx, ctx.param, dytq, xtq, ctx.w_dtype, ctx.backend_name) if do_dw else None
        return dx, dw, None, None, None, None, None, None


class NVFP4Linear(torch.nn.Linear):
    """nn.Linear with NVFP4 fprop/dgrad/wgrad. in/out features must be multiples of 128.

    The bias (if any) is added in bf16 after the GEMM. Inputs are cast to bf16.

    ``fuse_wgrad_accumulation``: when ``weight.grad`` already exists (fp32), add the new weight gradient
    to it inside the wgrad GEMM. Saves a full read-modify-write of the gradient per micro-batch under
    gradient accumulation. Gradient hooks, DDP and FSDP keep working (see _weight_grad); it is skipped
    automatically where it cannot apply (``torch.autograd.grad``, non-fp32 gradients, inside
    torch.compile).

    ``cache_weight_quantization``: keep the NVFP4 copies of the weight and reuse them while the weight
    is unchanged (bit-identical results; saves the weight quantizer on every micro-batch after the first
    under gradient accumulation, at ~1.1 bytes per weight element of memory). See WeightQuantCache for
    exactly when it notices weight updates.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None,
                 backend: str = "cuda", wgrad_rht: bool = True, fuse_wgrad_accumulation: bool = False,
                 cache_weight_quantization: bool = False):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if in_features % ALIGN or out_features % ALIGN:
            raise ValueError(f"NVFP4Linear needs in/out features divisible by {ALIGN}, got "
                             f"{in_features}->{out_features}")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {list(_BACKENDS)}")
        self.backend = backend
        self.wgrad_rht = wgrad_rht
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.cache_weight_quantization = cache_weight_quantization
        self.weight_quant_cache = WeightQuantCache()

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
        fuse = self.fuse_wgrad_accumulation and not torch.compiler.is_compiling()
        y = _NVFP4LinearFn.apply(x2d, self.weight, self.backend, self.wgrad_rht,
                                 grad and x2d.requires_grad, grad and self.weight.requires_grad, fuse,
                                 weight_cache(self))
        # Drop-in semantics: return the caller's dtype (bf16 under autocast / bf16 models).
        out_dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        y = y.to(out_dtype)
        if self.bias is not None:
            y = y + self.bias.to(out_dtype)
        return unflatten_output(y, lead, self.out_features)

    def extra_repr(self) -> str:
        r = super().extra_repr() + f", backend={self.backend}, wgrad_rht={self.wgrad_rht}"
        r += ", fuse_wgrad_accumulation=True" if self.fuse_wgrad_accumulation else ""
        return r + (", cache_weight_quantization=True" if self.cache_weight_quantization else "")


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


__all__ = ["NVFP4Linear", "WeightQuantCache", "convert_linears", "flops_per_token", "invalidate_weight_caches"]
