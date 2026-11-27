"""Where the NVFP4 loss gap comes from: train_gpt.py with chosen GEMMs of every NVFP4Linear run in BF16 instead.

    python experiments/gap_ablate.py <train_gpt.py args> --gap-bf16 fprop,dgrad,wgrad --gap-rtn

--gap-bf16: comma-separated subset of {fprop, dgrad, wgrad, fprop_x, fprop_w}: those GEMMs (or, for fprop_x /
fprop_w, only that fprop operand) skip the NVFP4 quantization (BF16 operands, FP32 accumulation).
--gap-rtn: round the output gradient to nearest instead of stochastically (dgrad and wgrad operands).
--gap-only: comma-separated subset of {qkv, proj, fc1, fc2}: apply --gap-bf16 / --gap-rtn to those layers only
(told apart by weight shape, d_model 768 / MLP 3072 of GPT-124M); default all.
Every evaluation also runs with all NVFP4Linear layers in BF16 (printed as {"eval_nvfp4": ..., "eval_bf16": ...}): how much of
the gap is the quantized forward at evaluation rather than worse weights.
(train_gpt's --no-rht still switches the wgrad Hadamard transform off.)

Runs the unfused NVFP4Linear path (no --fuse-producers / --fuse-wgrad: the fused blocks have their own kernels),
so compare against a baseline from this script with --gap-bf16 '' (the recipe as is, same path).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nvfp4.linear as L  # noqa: E402
from nvfp4 import ops  # noqa: E402
from nvfp4 import reference as ref  # noqa: E402


def _pop(argv, name, default, flag=False):
    if name not in argv:
        return default
    i = argv.index(name)
    if flag:
        del argv[i]
        return True
    val = argv[i + 1]
    del argv[i:i + 2]
    return val


argv = sys.argv[1:]
_BF = {s for s in _pop(argv, "--gap-bf16", "").split(",") if s}
RTN = _pop(argv, "--gap-rtn", False, flag=True)
_SHAPES = {"qkv": (2304, 768), "proj": (768, 768), "fc1": (3072, 768), "fc2": (768, 3072)}
ONLY = {s for s in _pop(argv, "--gap-only", "").split(",") if s}
if ONLY - set(_SHAPES):
    raise SystemExit(f"unknown --gap-only entries {ONLY - set(_SHAPES)}")
ONLY_SHAPES = {_SHAPES[m] for m in ONLY}
_bad = _BF - {"fprop", "dgrad", "wgrad", "fprop_x", "fprop_w"}
if _bad:
    raise SystemExit(f"unknown --gap-bf16 entries {_bad}")
if "fprop" in _BF:
    _BF |= {"fprop_x", "fprop_w"}


class _GapLinearFn(torch.autograd.Function):
    """_NVFP4LinearFn's recipe (CUDA backend, no fused accumulation) with per-GEMM BF16 switches."""

    @staticmethod
    def forward(ctx, x2d, weight, backend_name, wgrad_rht, need_dgrad, need_wgrad, fuse_wgrad, wcache, bias=None,
                residual=None, out_dtype=torch.bfloat16):
        if _EVAL_BF16[0]:
            y = x2d @ weight.to(torch.bfloat16).t()
            if bias is not None or residual is not None or out_dtype != torch.bfloat16:
                y = ref.linear_epilogue(y, out_dtype, bias, residual)
            return y
        seeds = L.draw_seeds(3, x2d.device)
        BF = _BF if not ONLY_SHAPES or tuple(weight.shape) in ONLY_SHAPES else set()
        ctx.rtn = RTN and (not ONLY_SHAPES or tuple(weight.shape) in ONLY_SHAPES)
        q_wgrad = need_wgrad and "wgrad" not in BF
        rht = seeds[2] if q_wgrad and wgrad_rht else None
        q_x = "fprop_x" not in BF
        xq = xtq = None
        if q_x or q_wgrad:
            xq, xtq = ops.quantize_dual(x2d, row=q_x, col=q_wgrad, rht_seed=rht, col_pad=L.ALIGN)
        q_dgrad = need_dgrad and "dgrad" not in BF
        wq = wtq = None
        if "fprop_w" not in BF or q_dgrad:
            wq, wtq = L.quantized_weight(weight, q_dgrad, "cuda", wcache)
        if "fprop_x" in BF and "fprop_w" in BF:
            y = x2d @ weight.to(torch.bfloat16).t()
        elif "fprop_x" in BF or "fprop_w" in BF:
            xa = x2d if "fprop_x" in BF else ops.dequantize(xq.data, xq.scales, xq.gdec)
            wa = weight.to(torch.bfloat16) if "fprop_w" in BF else ops.dequantize(wq.data, wq.scales, wq.gdec)
            y = xa @ wa.t()
        else:
            y = ops.gemm(xq, wq, torch.bfloat16)
        if bias is not None or residual is not None or out_dtype != torch.bfloat16:
            y = ref.linear_epilogue(y, out_dtype, bias, residual)
        ctx.save_for_backward(seeds, x2d if (need_wgrad and not q_wgrad) else None,
                              weight if (need_dgrad and not q_dgrad) else None,
                              *(xtq.as_storage() if q_wgrad else (None,) * 3),
                              *(wtq.as_storage() if q_dgrad else (None,) * 3))
        ctx.flags = (need_dgrad, need_wgrad, q_dgrad, q_wgrad, rht is not None, bias is not None, weight.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        seeds, x2d, weight, *st = ctx.saved_tensors
        need_dgrad, need_wgrad, q_dgrad, q_wgrad, use_rht, has_bias, w_dtype = ctx.flags
        nin = ctx.needs_input_grad
        grads = [None] * 11
        if nin[9]:
            grads[9] = dy
        dy = dy.contiguous()
        if dy.dtype not in (torch.bfloat16, torch.float32):
            dy = dy.float()
        do_dx, do_dw = need_dgrad and nin[0], need_wgrad and nin[1]
        dyq = dytq = None
        if (do_dx and q_dgrad) or (do_dw and q_wgrad):
            dyq, dytq = ops.quantize_dual(
                dy, row=do_dx and q_dgrad, col=do_dw and q_wgrad,
                row_sr_seed=None if ctx.rtn else seeds[0], col_sr_seed=None if ctx.rtn else seeds[1],
                rht_seed=seeds[2] if use_rht else None, col_pad=L.ALIGN)
        if do_dx:
            if q_dgrad:
                grads[0] = ops.gemm(dyq, ops.NVFP4Tensor(*st[3:6]), out_dtype=torch.bfloat16)
            else:
                grads[0] = (dy.to(torch.bfloat16) @ weight.to(torch.bfloat16))
        if do_dw:
            if q_wgrad:
                dw = ops.gemm(dytq, ops.NVFP4Tensor(*st[0:3]), out_dtype=torch.float32)
            else:
                dw = dy.float().t() @ x2d.float()
            grads[1] = dw.to(w_dtype)
        if has_bias and nin[8]:
            grads[8] = dy.reshape(-1, dy.shape[-1]).float().sum(0)
        return tuple(grads)


L._NVFP4LinearFn = _GapLinearFn
_EVAL_BF16 = [False]


def _patch_evaluate(tg):
    orig = tg.evaluate

    def evaluate(model, val, b, t, n_batches=20):
        v = orig(model, val, b, t, n_batches)
        _EVAL_BF16[0] = True
        try:
            vb = orig(model, val, b, t, n_batches)
        finally:
            _EVAL_BF16[0] = False
        print({"eval_nvfp4": v, "eval_bf16": vb}, flush=True)
        return v

    tg.evaluate = evaluate
if __name__ == "__main__":
    for flag in ("--fuse-producers", "--fuse-wgrad"):
        if flag in argv:
            raise SystemExit(f"{flag}: gap_ablate.py patches the unfused NVFP4Linear path only")
    import train_gpt

    _patch_evaluate(train_gpt)
    sys.argv = [sys.argv[0]] + argv
    print({"gap_bf16": sorted(_BF), "gap_rtn": RTN, "gap_only": sorted(ONLY)}, flush=True)
    train_gpt.main()
