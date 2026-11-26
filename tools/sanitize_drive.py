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
# v0.13: units without k-tiles (unused tiles -1, empty segments at k0 % 256 == 0 and 128) between real ones
qa, _ = ops.quantize_dual(torch.randn(1024, 256, device=d), col=False)
qb, _ = ops.quantize_dual(torch.randn(256, 256, device=d), col=False)
ops.gemm_grouped(qa, qb, "m", torch.tensor([0, -1, 1, -1, -1, 0, 1, -1], device=d, dtype=torch.int32), group_rows=128)
qa, _ = ops.quantize_dual(torch.randn(128, 1024, device=d), col=False)
qb, _ = ops.quantize_dual(torch.randn(256, 1024, device=d), col=False)
seg = torch.tensor([[0, 0, 0], [0, 384, 128], [384, 0, 256], [512, 512, 384], [1024, 0, 512]], device=d,
                   dtype=torch.int32)
ops.gemm_grouped(qa, qb, "k", seg, out_dtype=torch.float32, out=torch.zeros(640, 256, device=d))
# v0.14: per-group bias in the store, limited to each tile's real rows, next to unused tiles
qa, _ = ops.quantize_dual(torch.randn(1024, 256, device=d), col=False)
qb, _ = ops.quantize_dual(torch.randn(256, 256, device=d), col=False)
tg = torch.tensor([0, -1, 1, -1, -1, 0, 1, 1], device=d, dtype=torch.int32)
tr = torch.tensor([128, 0, 5, 0, 0, 77, 128, 0], device=d, dtype=torch.int32)
for odt in (torch.bfloat16, torch.float32):
    ops.gemm_grouped(qa, qb, "m", tg, group_rows=128, out_dtype=odt, bias=torch.randn(2, 128, device=d),
                     tile_rows=tr)
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
for act, bias, fp4, cf, sh in (("swiglu", False, True, None, 0), ("gelu_tanh", False, True, None, 0),
                               ("swiglu", True, True, None, 0), ("gelu_tanh", True, True, None, 0),
                               ("swiglu", True, False, None, 0),  # (fp4=False: the Triton BF16 path)
                               ("swiglu", True, True, 0.5, 256), ("gelu_tanh", False, False, 0.5, 128)):
    moe = MoE(128, 256, 4, top_k=2, activation=act, bias=bias, capacity_factor=cf, shared_d_ff=sh).to(d)
    if fp4:
        convert_linears(moe)
    xm = torch.randn(150, 128, device=d, requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ym = moe(xm)
    (ym.float().square().mean() + moe.aux_loss).backward()
# v0.17: the tensor-amax floor (plain and multi-part paths), dequantization of strided message rows with per-row
# tensor scales, persistent GEMMs with an SM margin
for R in (0, 1, 37, 130):
    xb = torch.randn(R, 256, device=d, dtype=torch.bfloat16)
    fl = torch.full((1,), 7.0, device=d)
    q, _ = ops.quantize_dual(xb, col=False, row_sr_seed=1, amax_floor=fl)
    ops.quantize_dual(xb, rht_seed=3, col_pad=128, amax_floor=fl)
    ops.quantize_dual(xb, bias_grad=True, row_sr_seed=1, col_sr_seed=2, rht_seed=3, col_pad=128, amax_floor=fl)
    msg = torch.cat([q.data, q.scales, q.gdec.view(1, 1).expand(R, 1).contiguous().view(torch.uint8)], 1)
    for dt in (torch.bfloat16, torch.float32):
        ops.dequantize(msg[:, :128], msg[:, 128:144], msg[:, 144:148].view(torch.float32)[:, 0], dt)
        ops.dequantize(q.data, q.scales, q.gdec, dt)
from nvfp4 import _library as _lib  # noqa: E402
for dt in (torch.bfloat16, torch.float32, torch.float16):
    for dd in (256, 200):  # 16-byte vectors / scalar path
        rows = torch.randn(130, dd + 8, device=d, dtype=dt)[:, :dd]  # strided rows
        posc = torch.randint(-1, 140, (37, 3), device=d)
        wc = torch.rand(37, 3, device=d)
        gc = _lib.combine_fwd(rows, posc, wc)
        _lib.combine_fwd(rows.contiguous(), posc, None)
        inv = torch.randint(-1, 111, (130,), device=d)
        _lib.combine_bwd_rows(gc, inv, wc, 3)
        _lib.combine_bwd_w(gc, rows.contiguous(), posc)
# v0.18: MXFP8 return messages (round-to-nearest and stochastic rounding; strided wide messages; zero rows)
for R in (0, 1, 37, 130):
    for dt in (torch.bfloat16, torch.float32):
        xm8 = (torch.randn(R, 256 + 64, device=d, dtype=dt) * 100)[:, :256]
        for seed in (None, torch.tensor([5], device=d, dtype=torch.int64)):
            m8 = _lib.mxfp8_encode(xm8, seed)
            _lib.mxfp8_decode(m8, 256, dt)
            wide = torch.zeros(R, 300, device=d, dtype=torch.uint8)
            wide[:, :264] = m8
            _lib.mxfp8_decode(wide[:, :264], 256, torch.bfloat16)
ops.set_sm_margin(40)
qa, _ = ops.quantize_dual(torch.randn(384, 512, device=d), col=False)
ops.gemm(qa, qa, impl="tma")
ops.set_sm_margin(0)
# dynamic tile claiming: dense (and split-K), grouped with runs of unused tiles, ragged K with an empty group
# (claiming engages only with more units than CTAs: a margin leaves 3 CTAs). Not under racecheck: its
# serialized execution of a block makes the threads that wait for a claim spin almost forever (and a
# flag-synchronized ring is outside what it models); memcheck, synccheck and initcheck cover it.
import os  # noqa: E402
ops.set_dynamic_tiles(os.environ.get("NVFP4_SANITIZE_TOOL") != "racecheck")
ops.set_sm_margin(torch.cuda.get_device_properties(0).multi_processor_count - 3)
qd, _ = ops.quantize_dual(torch.randn(1024, 512, device=d), col=False)
ops.gemm(qd, qa, impl="tma")
ops.gemm(qa, qa, torch.float32, impl="tma", splits=2)
tg = torch.tensor([-1, 0, 0, -1, -1, -1, 1, -1], device=d, dtype=torch.int32)
am_, _ = ops.quantize_dual(torch.randn(8 * 128, 512, device=d), col=False)
bm_, _ = ops.quantize_dual(torch.randn(2 * 256, 512, device=d), col=False)
ops.gemm_grouped(am_, bm_, "m", tg, group_rows=256)
sg = torch.tensor([[0, 256, 0], [256, 0, 128], [256, 256, 256]], device=d, dtype=torch.int32)
ak_, _ = ops.quantize_dual(torch.randn(128, 512, device=d), col=False)
bk_, _ = ops.quantize_dual(torch.randn(256, 512, device=d), col=False)
ops.gemm_grouped(ak_, bk_, "k", sg, out=torch.zeros(384, 256, device=d), out_dtype=torch.float32)
ops.set_dynamic_tiles(False)
ops.set_sm_margin(0)
torch.cuda.synchronize()
print("DRIVE_OK")
