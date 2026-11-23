"""Exercise every kernel variant once with small, ragged shapes (driver for tools/sanitize.sh)."""
import torch
from nvfp4 import ops
torch.manual_seed(0)
d = "cuda"
for shape in [(100, 48), (130, 256), (112, 64)]:
    for dt in (torch.bfloat16, torch.float32):
        x = torch.randn(shape, device=d, dtype=dt)
        ops.quantize(x)
        if shape[0] % 16 == 0:
            ops.quantize(x, transpose=True, stochastic=True, sr_seed=1, rht_seed=2)
        ops.quantize_dual(x, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)
        ops.quantize_dual(x, col=False); ops.quantize_dual(x, row=False, col_pad=64)
        w, b = torch.ones(shape[1], device=d), torch.zeros(shape[1], device=d)
        ops.layernorm_quantize_dual(x, w, b, 1e-5, rht_seed=1, col_pad=128)
    xb = torch.randn(shape, device=d, dtype=torch.bfloat16)
    ops.gelu_quantize_dual(xb, rht_seed=1, col_pad=128)
    ops.gelu_backward_quantize_dual(xb, xb, row_sr_seed=1, col_sr_seed=2, rht_seed=1, col_pad=128)
    # device-resident seeds (the path the layers use): shared-memory broadcast + warp-built RHT mask
    s = torch.randint(0, 2**62, (3,), device=d, dtype=torch.int64)
    ops.quantize_dual(xb, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2], col_pad=128)
    ops.layernorm_quantize_dual(xb, torch.ones(shape[1], device=d), torch.zeros(shape[1], device=d), 1e-5,
                                row_sr_seed=s[0], rht_seed=s[2], col_pad=128)
    ops.gelu_backward_quantize_dual(xb, xb, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2], col_pad=128)
ops.quantize_dual(torch.randn(64, 48, device=d), block_2d=True)
from nvfp4.fused import layernorm_mlp  # noqa: E402
from nvfp4.linear import NVFP4Linear  # noqa: E402
ln = torch.nn.LayerNorm(128).to(d)
fc1, fc2 = NVFP4Linear(128, 256, bias=False).to(d), NVFP4Linear(256, 128, bias=False).to(d)
xg = torch.randn(3, 50, 128, device=d, requires_grad=True)
(layernorm_mlp(xg, ln, fc1, fc2) + fc2(fc1(xg))).float().square().mean().backward()
for (m, n, k) in [(200, 72, 256), (130, 260, 1280), (64, 64, 2048)]:
    qa = ops.quantize(torch.randn(m, k, device=d)); qb = ops.quantize(torch.randn(n, k, device=d))
    for impl in ("tma", "cp.async"):
        for s in (1, 3):
            if impl == "tma" and k % 256:
                continue
            ops.gemm(qa, qb, impl=impl, splits=s)
            out = torch.zeros(m, n, device=d)
            ops.gemm(qa, qb, out=out, accumulate=True, impl=impl, splits=s)
torch.cuda.synchronize()
print("DRIVE_OK")
