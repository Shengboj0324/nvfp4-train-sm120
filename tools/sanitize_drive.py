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
    for dt in (torch.bfloat16, torch.float32):
        ops.rmsnorm_quantize_dual(torch.randn(shape, device=d, dtype=dt), torch.ones(shape[1], device=d), 1e-6,
                                  rht_seed=1, col_pad=128)
    ops.swiglu_quantize_dual(xb, xb, rht_seed=1, col_pad=128)
    for gate in (True, False):
        ops.swiglu_backward_quantize_dual(xb, xb, xb, gate=gate, row_sr_seed=1, col_sr_seed=2, rht_seed=1,
                                          col_pad=128)
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
from nvfp4.fused import rmsnorm_linear, rmsnorm_swiglu_mlp  # noqa: E402
norm = torch.nn.RMSNorm(128).to(d)
up, qkv = NVFP4Linear(128, 256, bias=False).to(d), NVFP4Linear(128, 384, bias=False).to(d)
(rmsnorm_swiglu_mlp(xg, norm, fc1, up, fc2).float().square().mean()
 + rmsnorm_linear(xg, norm, qkv).float().mean()).backward()
# v0.8: multi-part quantization with column sums, SwiGLU biases, norm backward (register rows / warp path)
for R in (100, 130):
    for dt in (torch.bfloat16, torch.float32):
        parts = [torch.randn(R, 64, device=d, dtype=dt) for _ in range(3)]
        ops.quantize_dual(parts, bias_grad=True, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)
        ops.quantize_dual(parts[0], bias_grad=True, row=False, col_pad=128)
    xb = torch.randn(R, 64, device=d, dtype=torch.bfloat16)
    bg, bu = torch.randn(64, device=d), torch.randn(64, device=d)
    ops.swiglu_quantize_dual(xb, xb, bias_g=bg, bias_u=bu, rht_seed=1, col_pad=128)
    for gate in (True, False):
        ops.swiglu_backward_quantize_dual(xb, xb, xb, gate=gate, bias_g=bg, bias_u=bu, bias_grad=True,
                                          row_sr_seed=1, col_sr_seed=2, rht_seed=1, col_pad=128)
for (R, C) in [(100, 48), (37, 2048), (20, 6144), (5, 16384)]:
    for dt in (torch.bfloat16, torch.float32):
        xn = torch.randn(R, C, device=d, dtype=dt)
        w, rstd, mean = torch.rand(C, device=d), torch.rand(R, device=d), torch.randn(R, device=d)
        ops.norm_backward(torch.randn(R, C, device=d), xn, mean, rstd, w, dres=torch.randn_like(xn), need_db=True)
        ops.norm_backward(torch.randn(R, C, device=d), xn, None, rstd, w, need_dx=False)
xg2 = torch.randn(2, 50, 128, device=d, requires_grad=True)
lin3, o = NVFP4Linear(128, 384).to(d), NVFP4Linear(128, 128).to(d)
g2, u2, d2 = NVFP4Linear(128, 256).to(d), NVFP4Linear(128, 256).to(d), NVFP4Linear(256, 128).to(d)
with torch.autocast("cuda", dtype=torch.bfloat16):
    q, k, v, res = rmsnorm_linear(xg2, norm, lin3, chunks=3, return_residual=True)
    h = o(q * k + v, residual=res)
    out = rmsnorm_swiglu_mlp(h, norm, g2, u2, d2, residual=True)
out.float().square().mean().backward()
# GEMMs: both main loops, split-K, accumulate, epilogue; the persistent TMA loop with more work units than
# SMs (2048x2048: 256 tiles; with splits 3: 768 units) and ragged edges
for (m, n, k) in [(200, 72, 256), (130, 260, 1280), (64, 64, 2048), (2048, 2000, 512)]:
    qa = ops.quantize(torch.randn(m, k, device=d)); qb = ops.quantize(torch.randn(n, k, device=d))
    bias, res = torch.randn(n, device=d), torch.randn(m, n, device=d)
    for impl in ("tma", "cp.async"):
        for s in (1, 3):
            if impl == "tma" and k % 256:
                continue
            ops.gemm(qa, qb, impl=impl, splits=s)
            out = torch.zeros(m, n, device=d)
            ops.gemm(qa, qb, out=out, accumulate=True, impl=impl, splits=s)
            ops.gemm(qa, qb, torch.bfloat16, bias=bias, residual=res, impl=impl, splits=s)
            ops.gemm(qa, qb, torch.float32, bias=bias, residual=res.bfloat16(), impl=impl, splits=s)
# 8-bit attention (nvfp4.attention): forward + backward, ragged lengths, GQA, causal / full, both head dims
from nvfp4.attention import attention  # noqa: E402
for (B, H, Hk, N, D, causal) in [(1, 2, 1, 77, 64, True), (1, 2, 2, 200, 128, False), (2, 4, 2, 130, 128, True)]:
    qa_, ka_, va_ = (torch.randn(B, h, N, D, device=d, dtype=torch.bfloat16, requires_grad=True) for h in (H, Hk, Hk))
    for bwd in ("bf16", "8bit"):
        attention(qa_, ka_, va_, causal=causal, backward=bwd).float().square().mean().backward()
# v0.11 recipe options: Four Over Six (row / col / 2D), MS-EDEN (RHT64 column phase, EDEN scales; host and
# device seeds), 1x16 weights with re-quantization, the BF16 tail, MoE (sorted per-expert fused MLPs)
for R in (80, 130):
    for dt in (torch.bfloat16, torch.float32):
        xr = torch.randn(R, 128, device=d, dtype=dt)
        with ops.four_six_scales(True):
            ops.quantize(xr)
            ops.quantize_dual(xr, rht_seed=3, col_pad=128)
            ops.quantize_dual(torch.randn(64, 128, device=d), block_2d=True)
        ops.set_ms_eden(True)
        ops.quantize_dual(xr, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)
        ops.quantize_dual(xr, rht_seed=3, col_pad=64)
        s = torch.randint(0, 2**62, (3,), device=d, dtype=torch.int64)
        ops.quantize_dual(xr, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2], col_pad=128)
        ops.set_ms_eden(False)
# MS-EDEN on dgrad: row RHT64 + EDEN (with and without the col output; host / device seeds), rotated weights
ops.set_ms_eden(True, dgrad=True)
for R in (80, 130):
    xr = torch.randn(R, 128, device=d, dtype=torch.bfloat16)
    ops.quantize_dual(xr, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128)
    ops.quantize_dual(xr, col=False, row_sr_seed=4)
    s = torch.randint(0, 2**62, (3,), device=d, dtype=torch.int64)
    ops.quantize_dual(xr, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2], col_pad=128)
    ops.gelu_backward_quantize_dual(xr, xr, row_sr_seed=s[0], col_sr_seed=s[1], rht_seed=s[2], col_pad=128)
xg.grad = None
(layernorm_mlp(xg, ln, fc1, fc2) + fc2(fc1(xg))).float().square().mean().backward()
ops.set_ms_eden(False)
# grouped GEMMs: ragged M (partial last k-tile: K = 384) and ragged K (half k-tiles at both ends)
qa, _ = ops.quantize_dual(torch.randn(384, 384, device=d), col=False)
qb, _ = ops.quantize_dual(torch.randn(3 * 128, 384, device=d), col=False)
ops.gemm_grouped(qa, qb, "m", torch.tensor([0, 2, 1], device=d, dtype=torch.int32), group_rows=128)
qa, _ = ops.quantize_dual(torch.randn(128, 1024, device=d), col=False)
qb, _ = ops.quantize_dual(torch.randn(256, 1024, device=d), col=False)
seg = torch.tensor([[0, 384, 0], [384, 512, 128], [896, 128, 256]], device=d, dtype=torch.int32)
ops.gemm_grouped(qa, qb, "k", seg, out_dtype=torch.float32, out=torch.zeros(384, 256, device=d), accumulate=True)
ops.set_weight_blocks("1d")
ops.set_four_six(True)
lw = NVFP4Linear(128, 256, bias=False).to(d)
lw(xg).float().square().mean().backward()
ops.set_four_six(False)
ops.set_weight_blocks("2d")
ops.set_high_precision(True)
layernorm_mlp(xg, ln, fc1, fc2).float().mean().backward()
ops.set_high_precision(False)
from nvfp4 import convert_linears  # noqa: E402
from nvfp4.moe import MoE  # noqa: E402
for act in ("swiglu", "gelu_tanh"):
    moe = MoE(128, 256, 4, top_k=2, activation=act).to(d)
    convert_linears(moe)
    xm = torch.randn(150, 128, device=d, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ym = moe(xm)
    (ym.float().square().mean() + moe.aux_loss).backward()
torch.cuda.synchronize()
print("DRIVE_OK")
