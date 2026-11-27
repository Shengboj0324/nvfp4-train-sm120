"""Which forward format would close the NVFP4 loss gap: train_gpt.py with the fprop GEMM of every NVFP4Linear emulated
in another format (dequantized operands, BF16 GEMM), the backward the recipe as is (gap_ablate.py's unfused path).

    python experiments/gap_fwd.py <train_gpt.py args> --fwd-x FMT --fwd-w FMT [--fwd-rot N] [--gap-only LAYERS]

FMT, along the reduction dim (in_features):
  nvfp4   the recipe's quantizer (activations 1x16 blocks, weights 16x16; RTN), via ops.quantize + ops.dequantize
  nvfp4_1d  weights in 1x16 blocks too
  nvfp4x2 two NVFP4 terms (1x16 blocks): q1 = Q(x), q2 = Q(x - q1), each with its own tensor scale (one GEMM of
          twice the depth, [q1 | q2] [W | W]^T, on the NVFP4 kernel: the cost of an MXFP8 GEMM on RTX 50)
  mxfp8   E4M3 with one UE8M0 scale per 32 elements
  fp8row  E4M3 with one fp32 scale per row (amax / 448)
  bf16    unquantized
--fwd-rot N (16, 32, ..., 256): both operands rotated by the same random orthogonal diag(s) H_N / sqrt(N) along the
reduction dim before quantization (x W^T is unchanged), one sign vector per layer shape, fixed for the run.
--gap-only as in gap_ablate.py. Evaluations also run in BF16 (gap_ablate.py).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gap_ablate as G  # noqa: E402  (patches nvfp4.linear._NVFP4LinearFn)
from nvfp4 import ops  # noqa: E402
from nvfp4 import reference as ref  # noqa: E402

argv = sys.argv[1:]
FX = G._pop(argv, "--fwd-x", "nvfp4")
FW = G._pop(argv, "--fwd-w", "nvfp4")
ROT = int(G._pop(argv, "--fwd-rot", "0"))
_ONLY = {s for s in G._pop(argv, "--gap-only", "").split(",") if s}
ONLY_SHAPES = {G._SHAPES[m] for m in _ONLY}
_FMTS = ("nvfp4", "nvfp4_1d", "nvfp4x2", "mxfp8", "fp8row", "bf16")
if FX not in _FMTS or FW not in _FMTS:
    raise SystemExit(f"--fwd-x / --fwd-w: one of {_FMTS}")
if ROT and (ROT & (ROT - 1) or ROT < 2):
    raise SystemExit("--fwd-rot: a power of two")
# the parent computes the recipe's backward operands and a BF16 forward we replace
G._BF.clear()
G._BF.update({"fprop_x", "fprop_w"})
G.ONLY_SHAPES.clear()

_rot_cache = {}


def _rotation(k, device):
    if (k, device) not in _rot_cache:
        h = torch.ones(1, 1)
        while h.shape[0] < ROT:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        g = torch.Generator().manual_seed(1000 + k)
        s = (torch.randint(0, 2, (ROT,), generator=g) * 2 - 1).float()
        _rot_cache[(k, device)] = (s.unsqueeze(1) * h / ROT ** 0.5).to(device)
    return _rot_cache[(k, device)]


def _rotate(t):
    k = t.shape[-1]
    m = _rotation(k, t.device)
    return (t.float().reshape(-1, k // ROT, ROT) @ m).reshape(t.shape)


def _fake(t, fmt, weight):
    if fmt == "bf16":
        return t.to(torch.bfloat16)
    if fmt == "mxfp8":
        return ref.mxfp8_fake(t.float(), dim=-1).to(torch.bfloat16)
    if fmt == "fp8row":
        tf = t.float()
        sc = (tf.abs().amax(-1, keepdim=True) / 448.0).clamp(min=1e-30)
        return ((tf / sc).to(torch.float8_e4m3fn).float() * sc).to(torch.bfloat16)
    tb = t.to(torch.bfloat16).contiguous()
    q = ops.quantize(tb, block_2d=weight and fmt == "nvfp4")
    d = ops.dequantize(q.data, q.scales, q.gdec, torch.float32)
    if fmt == "nvfp4x2":
        r = (tb.float() - d).to(torch.bfloat16).contiguous()
        q2 = ops.quantize(r)
        d = d + ops.dequantize(q2.data, q2.scales, q2.gdec, torch.float32)
    return d.to(torch.bfloat16)


class _FwdLinearFn(G._GapLinearFn):
    @staticmethod
    def forward(ctx, x2d, weight, backend_name, wgrad_rht, need_dgrad, need_wgrad, fuse_wgrad, wcache, bias=None,
                residual=None, out_dtype=torch.bfloat16):
        y = G._GapLinearFn.forward(ctx, x2d, weight, backend_name, wgrad_rht, need_dgrad, need_wgrad, fuse_wgrad,
                                   wcache, bias, residual, out_dtype)
        if G._EVAL_BF16[0] or (ONLY_SHAPES and tuple(weight.shape) not in ONLY_SHAPES):
            if ONLY_SHAPES and not G._EVAL_BF16[0]:  # the recipe's NVFP4 forward for the other layers
                xq = ops.quantize(x2d.contiguous())
                wq, _ = G.L.quantized_weight(weight, False, "cuda", wcache)
                y = ops.gemm(xq, wq, torch.bfloat16)
                if bias is not None or residual is not None or out_dtype != torch.bfloat16:
                    y = ref.linear_epilogue(y, out_dtype, bias, residual)
            return y
        xa, wa = (x2d, weight) if not ROT else (_rotate(x2d), _rotate(weight.detach()))
        y = _fake(xa, FX, False) @ _fake(wa, FW, True).t()
        if bias is not None or residual is not None or out_dtype != torch.bfloat16:
            y = ref.linear_epilogue(y, out_dtype, bias, residual)
        return y


G.L._NVFP4LinearFn = _FwdLinearFn
if __name__ == "__main__":
    for flag in ("--fuse-producers", "--fuse-wgrad"):
        if flag in argv:
            raise SystemExit(f"{flag}: gap_fwd.py patches the unfused NVFP4Linear path only")
    import train_gpt

    G._patch_evaluate(train_gpt)
    sys.argv = [sys.argv[0]] + argv
    print({"fwd_x": FX, "fwd_w": FW, "fwd_rot": ROT, "gap_only": sorted(_ONLY)}, flush=True)
    train_gpt.main()
