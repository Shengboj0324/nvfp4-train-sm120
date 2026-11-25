# Using nvfp4-train-sm120 in a training stack

This guide is for integrating the library into real training code. [DESIGN.md](DESIGN.md) explains how it
works and [RESULTS.md](RESULTS.md) what it measured.

## 1. Requirements

| | |
|---|---|
| GPU | compute capability **12.0** exactly: GeForce RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 Ti / 5060 and their laptop parts. The kernels are built for the arch-specific target `sm_120a`, which runs on no other GPU (not on sm_100 B200, sm_121 GB10, or older GPUs). |
| OS | Linux, WSL2 or native Windows (Visual Studio 2022 with the C++ workload). |
| Toolkit | CUDA ≥ 12.8 (TMA descriptor API by version needs ≥ 12.5). Tested: 13.0 on Linux/WSL2, 12.9 on Windows. Driver tested: 595.79. |
| PyTorch | a CUDA build ≥ 2.7, tested with 2.14.1+cu130 (WSL2) and 2.7.1+cu128 (Windows). The extension must be built against the same PyTorch it runs with. |

`nvfp4.available()` returns whether the extension is built and the current GPU is supported. Every op
also checks its inputs and raises a clear error for CPU tensors or unsupported GPUs.

## 2. Install

```bash
pip install --no-build-isolation .                  # builds csrc/ for sm_120a (2-4 min)
# or a wheel for deployment:
pip wheel --no-build-isolation --no-deps -w dist .  # dist/nvfp4_train_sm120-<version>-cp3XX-*.whl
```

`MAX_JOBS=4` limits build parallelism (the build is CPU and memory heavy). `NVFP4_SKIP_CUDA=1`
installs only the pure-PyTorch reference (CPU CI, numerics studies).

**Native Windows:** `tools\build.cmd` finds CUDA (`CUDA_HOME`, `CUDA_PATH` or the default install
folder) and Visual Studio (`vswhere`), sets up the x64 compiler environment and builds in place under
the resource guard; `tools\check.cmd` also runs the tests. `pip install --no-build-isolation .` works
too from an "x64 Native Tools" prompt. PyTorch's Windows wheels have no Triton and no NCCL, so
`torch.compile` there needs a non-inductor backend (or a Windows build of Triton) and multi-process
training uses gloo.

Verify an installation on the target machine:

```bash
tools/check.sh            # build + 637 tests (~35 s of GPU time); Windows: tools\check.cmd
tools/check.sh release    # + compute-sanitizer (memcheck, racecheck, synccheck, initcheck) + 5-minute soak
```

On a multi-GPU Linux machine (for example a rented cloud instance), `tools/pod_session.sh` does all
of it in one go: installs a CUDA toolkit and PyTorch if needed, builds, runs the tests (including the
two-GPU NCCL tests, which skip on one GPU), sanitizers and soak, the benchmarks, and GPT-124M runs on
one GPU and with DDP over all GPUs, and packs every log and result into one archive.

## 3. Integrating

### 3.1 Linear layers

```python
from nvfp4 import NVFP4Linear, convert_linears

# Replace the transformer-block linears (dims must be multiples of 128), keeping the last blocks
# (and the LM head / embeddings, which are not converted) in BF16, as in NVIDIA's recipe:
converted = convert_linears(model.blocks, skip=("22.", "23."))
```

`NVFP4Linear` is an `nn.Linear` subclass: its `state_dict` is interchangeable with `nn.Linear`, so
checkpoints load either way, and the weights stay in their original dtype (fp32 master weights or
bf16). Inputs of any leading shape and of dtype fp32 / bf16 / fp16 are accepted; the output has the
autocast dtype (or the input dtype without autocast). The bias is added inside the GEMM's store, and
`layer(x, residual=r)` returns `r + layer(x)` with the add fused there too (bit-identical to adding
afterwards; for the attention output projection, see §3.2).

Options:

| Argument | Default | Meaning |
|---|---|---|
| `wgrad_rht` | `True` | 16-point random Hadamard transform on the wgrad operands (NVIDIA recipe). |
| `fuse_wgrad_accumulation` | `False` | Add the weight gradient into an existing fp32 `weight.grad` inside the GEMM, saving a read-modify-write of the gradient per micro-batch under gradient accumulation. Autograd still sees the gradient (the accumulated tensor is handed to it without a copy), so gradient hooks, DDP and FSDP2 work. A tensor hook on the weight receives the accumulated gradient rather than this micro-batch's part. Not applied under `torch.autograd.grad`, to non-fp32 gradients, or inside `torch.compile`, which take the ordinary path. |
| `cache_weight_quantization` | `False` | Keep the NVFP4 copies of the weight and reuse them while it is unchanged: every micro-batch after the first in a gradient-accumulation step skips the weight quantizer (1–8% of a block step), bit-identical results, ~1.1 bytes per weight element of memory. Invalidated by any `torch.optim` optimizer step (also fused ones, which don't bump version counters), by in-place changes through the parameter, `load_state_dict`, `.to()` and assigning `.data`. **Not** noticed: in-place writes through `param.data` outside an optimizer step (`p.data.add_(...)`; PyTorch gives `.data` its own version counter, so nothing can see them); call `nvfp4.invalidate_weight_caches(model)` after those. Off automatically for FSDP1 (non-leaf views), inside `torch.compile`, and permanently for a layer captured in a CUDA graph. Under FSDP2, which gathers weights into new memory every forward, it hits only when the allocator returns the same address within a step (correct either way). |
| `backend` | `"cuda"` | `"emulate"` runs the identical recipe with the pure-PyTorch reference on any device (slow; for verification). |

### 3.2 Transformer blocks with fused normalization, activation and residual adds

```python
from nvfp4 import layernorm_linear, layernorm_mlp, rmsnorm_linear, rmsnorm_swiglu_mlp, gelu_mlp, swiglu_mlp

def attend(q, k, v):                         # your attention; q, k, v: [b, t, d]
    b, t, d = q.shape
    q, k, v = (z.view(b, t, heads, d // heads).transpose(1, 2) for z in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)

# Llama / Mistral / Qwen style, pre-norm (RMSNorm, SwiGLU), fully fused (v0.8):
q, k, v, res = rmsnorm_linear(x, blk.input_norm, blk.qkv, chunks=3, return_residual=True)
x = blk.o_proj(attend(q, k, v), residual=res)                       # x + o_proj(attn)
x = rmsnorm_swiglu_mlp(x, blk.post_norm, blk.gate, blk.up, blk.down, residual=True)
#                                     # x + down(silu(gate(n)) * up(n)), n = post_norm(x)

# GPT-2 / BERT / ViT style, pre-norm (LayerNorm, GELU tanh or exact; biases anywhere):
q, k, v, res = layernorm_linear(x, blk.ln1, blk.qkv, chunks=3, return_residual=True)
x = blk.proj(attend(q, k, v), residual=res)
x = layernorm_mlp(x, blk.ln2, blk.fc1, blk.fc2, approximate="tanh", residual=True)

# post-norm (original BERT): h = LN1(x + attn(x)); out = LN2(h + mlp(h))
out = blk.ln2(gelu_mlp(h, blk.fc1, blk.fc2, approximate="none", residual=True))

# the plain forms still work: qkv = rmsnorm_linear(x, norm, lin); y = x + layernorm_mlp(x, ln, fc1, fc2)
```

The modules stay ordinary `nn.LayerNorm` / `nn.RMSNorm` / `NVFP4Linear` modules (same parameters,
same state dict); only the forward call changes. Requirements: `NVFP4Linear` with `backend="cuda"`,
normalization over the last dimension. What each option fuses:

| Option | Effect |
|---|---|
| `chunks=3` | q, k, v as views of the qkv GEMM output; their gradients (contiguous when they come from PyTorch's flash attention) are quantized in place instead of being concatenated into one tensor. |
| `return_residual=True` | also returns a handle of x; pass it as the residual of the attention output projection (`NVFP4Linear.forward(attn, residual=handle)`), whose epilogue does the add, and whose residual gradient is then added inside this function's norm backward. |
| `residual=True` (MLPs) | returns `x + mlp(x)`: the add in the down projection's epilogue, the residual gradient in the norm backward. |
| norm `None`, `gelu_mlp`, `swiglu_mlp` | no normalization: post-norm blocks, plain MLPs. |
| biases | anywhere: fc1's and the SwiGLU gate / up biases are added inside the activation kernel (gradients from the same read), the others in the GEMM epilogue (gradients from the output-gradient quantizer). |

The RMSNorm module may be `torch.nn.RMSNorm` (with or without weight, default eps) or any module with
`weight` and `variance_epsilon` (Hugging Face's `LlamaRMSNorm`). The normalized input is quantized
once for both the gate and the up projection. Outputs are bit-identical to the unfused composition of
the same NVFP4 ops; gradients differ only in rounding (the fused path quantizes fp32 values directly
where the unfused one rounds them to bf16 first). `experiments/train_gpt.py --fuse-producers` shows a
full GPT model; `bench/bench_llama_block.py` a Llama-style block.

### 3.3 Attention in 8 bits (v0.9)

```python
from nvfp4.attention import attention
o = attention(q, k, v, causal=True)   # instead of F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

q is [B, H, N, D], k and v [B, Hkv, N, D] (H a multiple of Hkv: grouped-query attention without
repeating k / v), bf16 or fp16, head dim 64 or 128, any strides with a unit stride along D (the views
`chunks=3` returns work as they are). Q K^T and dO V^T run on INT8 tensor cores with per-token scales,
the products with probabilities and score gradients on MXFP8; softmax statistics and accumulation stay
FP32 (DESIGN.md §7). The output is a [B, H, N, D] view of [B, N, H, D] memory, so
`o.transpose(1, 2).reshape(B, N, H * D)` is free; gradients come back in the inputs' layouts when those
are dense, else as the same kind of view. Anything it doesn't cover (other head dims, fp32, attention
masks, dropout, a non-positive scale) falls back to `F.scaled_dot_product_attention`
(`nvfp4.attention.supported(q, k, v)` tells which path a call takes).

Two backwards (DESIGN.md §7.1):

| `backward=` | what | fwd / fwd+bwd vs flash attention (D = 128, 4k–8k tokens) | quality (GPT-124M, 4096 tokens) | deterministic |
|---|---|---|---|---|
| `"bf16"` (default) | PyTorch's flash backward of the 8-bit forward (on the dequantized operands; P V in two e4m3 terms) | 1.2–1.6× / 1.07–1.12× | same loss as BF16 attention | under `torch.use_deterministic_algorithms`, like SDPA |
| `"8bit"` | this package's INT8 / MXFP8 backward (one-term P V) | 1.5–2.1× / 1.12–1.18× | +0.5% loss | always (no atomics) |

When to use it: for forward-only work (inference, evaluation) pass `backward="8bit"` for the faster
forward (no backward runs anyway). For training, the default costs no measurable quality at 512 or 4096
tokens on GPT-124M and gains 7–12% per attention call at head dim 128 (4–7% at 64), so its value
grows with the share of attention in the step (long context). Errors are those of 8-bit operands in the
forward (output ~1–3% relative to exact attention, vs ~0.2% for BF16 flash attention); validate on your
model.

### 3.4 Mixture of experts (v0.11; grouped GEMMs since v0.12; sync-free since v0.13; biases since v0.14)

```python
from nvfp4.moe import MoE
moe = MoE(d=1024, d_ff=2816, n_experts=8, top_k=2, activation="swiglu")   # or "gelu_tanh", "gelu"; bias=True
nvfp4.convert_linears(moe, fuse_wgrad_accumulation=True)   # = moe.to_nvfp4(...); the router stays BF16
y = moe(x)                                    # x [..., d]
loss = loss + 0.01 * moe.aux_loss             # Switch load-balancing loss (1.0 when balanced)
```

- **Routing.** Dropless top-k (softmax in fp32, weights renormalized over the k choices).
- **Parameters.** Expert weights are stacked: `w_gate`, `w_up` [E, d_ff, d] and `w_down` [E, d, d_ff]
  (SwiGLU), or `w_fc1` [E, d_ff, d] and `w_fc2` [E, d, d_ff] (GELU). With `bias=True` every projection
  has per-expert biases, `b_gate`, `b_up` [E, d_ff] and `b_down` [E, d] (`b_fc1`, `b_fc2`;
  `moe.experts_biases()`), initialized like nn.Linear's. On the NVFP4 path the GEMM's store adds them, to
  an expert's real rows only, so the padding rows stay zero and never reach a quantization block's scale;
  their gradients come from the column sums the gradient quantizers compute anyway (no extra pass).
  `convert_linears` / `to_nvfp4` take NVFP4Linear's keywords: `wgrad_rht`, `fuse_wgrad_accumulation`,
  `cache_weight_quantization`.
- **NVFP4 path** (DESIGN.md §4.5):
  - The routed tokens are gathered once into a buffer sorted by expert (each expert padded to 128 rows)
    and quantized once.
  - Each projection of all experts is one grouped GEMM (`ops.gemm_grouped`), and the weight gradients of
    all experts are one more.
  - The activation and its backward are fused into the quantizers, as in the dense MLPs.
- **BF16 path** (before conversion, or under `ops.set_high_precision`, e.g. a BF16 tail; v0.14): the same
  buffer and tile table, with grouped BF16 GEMMs written in Triton (`nvfp4/_bf16_grouped.py`), one launch per
  projection and per gradient. fp32 experts (no autocast) use IEEE fp32 products. Without Triton (PyTorch's
  native Windows wheels) the experts run one after another, reading the expert sizes on the host; set
  `moe.bf16_impl = "loop"` to force that, `"grouped"` to require Triton (default `"auto"`).
  (`torch._grouped_mm` would do the same job, but on sm_120 PyTorch 2.14 runs it as a loop that
  synchronizes.)
- **Determinism.** Dispatch and combine are gathers (no scatter-add atomics), so both paths are
  deterministic.
- **No host sync (NVFP4 path since v0.13, BF16 path since v0.14).** The expert sizes stay on the GPU:
  - The token buffer has the worst-case size for the step's tokens (T k + E·127 rows, rounded up to 256),
    and its layout tables are computed on the GPU.
  - The grouped GEMMs skip the unused tail tiles and the experts that got no tokens.
  - A training step with NVFP4 MoE layers can therefore be captured in a CUDA graph. One graph serves any
    routing of the same number of tokens.
  - It also compiles with `torch.compile(fullgraph=True)`.
  - Only the BF16 loop (no Triton, or `bf16_impl="loop"`) reads the sizes, once per call.
  - `moe.last_counts` reads them on demand (that read synchronizes).
- **Compiled vs eager.** By default inductor keeps fp32 intermediates inside its fused kernels where eager
  rounds to bf16. A slightly different bf16 gradient moves an NVFP4 tensor scale, and every rounding after
  it. Compiled MoE training is then statistically the same as eager, not bit-identical. With
  `torch._inductor.config.emulate_precision_casts = True` (and `fallback_random = True` for the seeds),
  the outputs, the bias gradients and the NVFP4 expert weight gradients are bit-identical to eager (tested;
  the input gradient, and on the BF16 path the weight gradients behind the activation's backward, have
  under 1% of their elements a bf16 step or two apart).
- **Scripts.** `experiments/train_gpt.py --moe-experts 8` trains a GPT with MoE MLPs; `bench/bench_moe.py`
  times a layer (RESULTS.md §8).
- **Changed in 0.12:** the per-expert `experts` ModuleList is gone (and the `bias` argument, back in 0.14
  with stacked biases). To load a 0.11 state dict, stack its `experts.{e}.gate.weight` etc. along a new
  dim 0 (`experts.{e}.gate.bias` into `b_gate`). The stacked weights are
  parameters, not `nn.Linear` modules, so a model-level init that matches `isinstance(m, nn.Linear)` (GPT's
  N(0, 0.02) with scaled residual projections, for example) no longer reaches them: initialize
  `moe.experts_weights()` explicitly, as `experiments/train_gpt.py` does. Left at their default (uniform,
  bound 1/√fan_in) the GPT's expert output projections start about 3.6× larger, and NVFP4 training ends
  about 1% worse.

### 3.5 Recipe options (v0.11)

Process-wide switches, read when a quantizer launches (set them before building / capturing the step):

| Switch | Effect | Default |
|---|---|---|
| `ops.set_four_six(True)` (or `with ops.four_six_scales():`) | "Four Over Six" block scales on every round-to-nearest operand: each 16-block keeps the smaller-error of max → 6 and max → 4; tensor scale 6 · 256 / amax | off |
| `ops.set_ms_eden(True)` | MS-EDEN on the wgrad operands: 64-point RHT along the tokens, and the gradient operand quantized with RTN elements and an EDEN-corrected, stochastically rounded block scale (unbiased, about half the error of element SR) | off |
| `ops.set_ms_eden(True, dgrad=True)` | MS-EDEN on the dgrad operands as well (v0.12): dY and the GELU / SwiGLU input gradients rotated along N by a 64-point RHT and quantized with EDEN; the dgrad weight operand is the dequantized fprop weight re-quantized under the same rotation | off |
| `ops.set_weight_blocks("1d")` | 1×16 weight blocks for fprop, the dequantized fprop weight re-quantized along N for dgrad | `"2d"` (16×16) |
| `ops.set_high_precision(True)` | BF16 tail: `NVFP4Linear` and the fused sub-blocks compute exactly what the modules they replace would (BF16 GEMMs, ordinary autograd) until switched off; weights and optimizer state are untouched, so flip it for the last ~10% of steps | off |

The BF16 tail is the one option that closes a large part of the gap (+1.32% → about +0.6% at 10% of the
steps, for ≈4% of the run's time); the quantizer options lower the loss by 0.03–0.13% each at
GPT-124M scale for 2–13% throughput. Recommended: the default recipe plus a BF16 tail; add
`set_ms_eden(True)` and `set_four_six(True)` only after checking them on your model.

Their measured effect on GPT-124M is in RESULTS.md §4.2; `experiments/train_gpt.py` has `--four-six`,
`--ms-eden`, `--ms-eden-dgrad`, `--weight-1d`, `--bf16-tail FRAC`, and `--hp-modules fc2` (keep a linear in BF16 in every
block).

### 3.6 Low-level ops

```python
from nvfp4 import quantize, quantize_dual, gemm, norm_backward
qa = quantize(a)                                   # NVFP4Tensor(data, scales, gdec), along the last dim
qb, qbt = quantize_dual(b, block_2d=True)          # b and b^T from one read
d = gemm(qa, qb)                                   # a @ b^T, bf16 (or fp32 / into `out`, accumulate=True)
y = gemm(qa, qb, torch.bfloat16, bias=bias, residual=r)  # r + (a @ b^T + bias), fused in the store
r_q, c_q, db = quantize_dual([dq, dk, dv], bias_grad=True)  # concatenation read in place + column sums
# grouped GEMMs (MoE): rows of `a` grouped in 128-row tiles (tile_group: int32 group per tile, -1 = unused), B stacked
y = ops.gemm_grouped(a_q, w_stacked_q, "m", tile_group, group_rows=N)
ops.gemm_grouped(a_t, b_t, "k", segs, out_dtype=torch.float32, out=dw, accumulate=True)  # segs: (k0, len, row0)
```

Seeds (`row_sr_seed`, `col_sr_seed`, `rht_seed`) are Python ints or one-element int64 tensors on the
GPU; a tensor is read by the kernel, so the value can come from a GPU random generator without a host
sync, and changes on every CUDA-graph replay. Both give bit-identical results for the same value.

## 4. Behaviour you can rely on

- **Bit-exact quantization.** Every quantizer (including the fused LayerNorm / GELU variants) is
  bit-identical to `nvfp4.reference`, the executable spec; the GEMM matches an fp64 reference to ~1e-7.
- **Determinism.** (`nvfp4.attention(backward="8bit")` is deterministic by construction; the default
  backward is PyTorch's flash backward and follows the deterministic-mode switch below.) With the same seed (`torch.manual_seed`), forward and backward are bit-reproducible,
  including stochastic rounding and split-K. The layers draw their rounding seeds from the CUDA
  generator, on the GPU. PyTorch's own flash / memory-efficient attention backward is *not*
  bit-reproducible by default (it accumulates dQ with atomics). For bitwise reproducible runs, set
  `torch.use_deterministic_algorithms(True)` (and `CUBLAS_WORKSPACE_CONFIG=:4096:8`): with PyTorch 2.14
  flash attention then uses a reproducible backward at a small cost (0.42 vs 0.29 ms for an
  8×12×512×64 causal forward + backward), the NVFP4 layers run unchanged, and a block's gradients
  repeat bit for bit (tested; `tools/soak.py` runs this way). cuDNN attention and FlexAttention were
  reproducible on this GPU even without it.
- **No host synchronization.** A training step through the layers never waits for the GPU (tested with
  `torch.cuda.set_sync_debug_mode("error")`), so the CPU can run ahead of the GPU.
- **Non-finite values are never hidden.** A NaN or ±inf anywhere in a quantized tensor makes its
  per-tensor scale non-finite, so every output that tensor feeds is non-finite. A diverging run shows
  NaN/inf in its loss exactly as in BF16 (the whole GEMM output becomes non-finite, not just one row).
- **Activation checkpointing** (`torch.utils.checkpoint`, both modes) reproduces gradients bit for bit:
  the seeds come from the CUDA generator, whose state checkpointing restores before recomputing.
- **Streams.** All kernels run on the current CUDA stream and allocate through PyTorch's caching
  allocator; concurrent streams are safe (split-K workspaces are per call). As always in PyTorch,
  synchronize streams yourself when one consumes another's output.
- **Data parallel.** DDP (also with `gradient_as_bucket_view` and `no_sync` gradient accumulation),
  FSDP1 (`FullyShardedDataParallel`, flat parameters or `use_orig_params=True`, with `no_sync`) and
  FSDP2 (`fully_shard`, with `set_requires_gradient_sync`) produce gradients bit-identical to a
  single-process computation of the same micro-batches, with or without fused accumulation (tested
  with two processes, on Linux and on Windows; the MoE layer too, v0.13). Under FSDP1 the layers see views of the flat
  parameter, so fused accumulation quietly takes the ordinary path. Kernel attributes are set per device, so one process driving several GPUs works
  too.
- **Hangs become errors.** GEMM pipeline waits are bounded (1 s, then `trap`), so a fault surfaces as
  a CUDA error instead of a display-driver watchdog reset.

## 5. Things to know in production

| Topic | What happens |
|---|---|
| `torch.compile` | No graph breaks: `fullgraph=True` works for models with NVFP4 layers and the fused LayerNorm / MLP blocks (the kernels are registered as `torch.library` custom ops). With `torch._inductor.config.fallback_random = True` the compiled step is bit-identical to eager; otherwise inductor draws its own random numbers (different seeds, same distribution). |
| CUDA graphs | Inference and whole training steps (forward, backward, optimizer) can be captured. Seeds are drawn on the GPU, so each replay uses fresh stochastic-rounding / RHT seeds. Use the usual recipe: warm up on a side stream, `zero_grad(set_to_none=True)` before capture, `capturable=True` optimizers. |
| DDP / FSDP | DDP, FSDP1 and FSDP2 tested (see §4), including fused wgrad accumulation. Pass FSDP2 a mesh built from your process group (`DeviceMesh.from_group`) when it isn't NCCL. PyTorch 2.7's FSDP2 sums gradients over gloo where 2.14 averages them (same for `nn.Linear`). |
| Memory | Saved activations are NVFP4 (~0.56 bytes per element) instead of BF16; peak memory of a block step is 7–14% below BF16. |
| Shapes | `in_features` / `out_features` must be multiples of 128. Token counts are arbitrary (padded internally). Very small layers (< 128 wide) should stay BF16. |
| Fused blocks | GPT / BERT / ViT style (LayerNorm, tanh or exact GELU) and Llama style (RMSNorm, SwiGLU), pre- or post-norm, biases anywhere, residual adds fused. Attention between them is yours: PyTorch's SDPA, or `nvfp4.attention` (§3.3). |
| Attention | `nvfp4.attention` saves the dequantized q, k, v (default backward) or 8-bit copies in two layouts (`backward="8bit"`), as much memory as the BF16 tensors SDPA saves, plus the output; `torch.compile` traces it as custom ops. |
| Version changes | v0.8 changed the stochastic-rounding generator (one hash per element pair) and quantizes fp32 output gradients without first rounding them to bf16: gradients are not bit-identical to v0.7's for the same seed (forward results are). |
| Precision recipe | Keep embeddings, the LM head and the last 1–2 blocks in BF16. Measured quality gap at 124M / 507M parameters: +1.0–1.4% validation loss, constant over training; at 124M over three seeds +0.019 ± 0.002 nats (+1.28%) (RESULTS.md §4–5). Training the last 10% of steps with `ops.set_high_precision(True)` halves it (+0.62%, RESULTS.md §4.2); serve such a model in BF16. Validate on your own model before relying on it. |
| Throughput | Gains grow with width: GPT block fwd+bwd 1.8× BF16 at d=768, 2.7× at d=2048, 3.2× at d=4096 (3.5× for micro-batches after the first with the weight cache); Llama block 2.1–3.3× (3.6×) on an RTX 5090 Laptop GPU. |

## 6. Operating on a laptop / workstation

Long GPU jobs on a laptop at its power limit are where hardware and drivers misbehave. `tools/guard.py`
wraps any command, samples the GPU and host every second, kills the job at 84 °C GPU, 90% GPU memory,
< 1.5 GB free host RAM or sustained CPU overload, waits for the GPU to cool before starting and cools
down after. `tools/run.sh <name> <script> ...` runs a Python script under it.

On the development laptop one Windows bugcheck (0x7E, NVIDIA driver) happened as a 13-minute run
exited, at 76 °C; the machine had similar crashes before this project. Since then training scripts
synchronize and free GPU memory before exiting and the kernels cannot spin indefinitely; keep the
NVIDIA driver up to date.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nvfp4 CUDA extension not built` | Install with `pip install --no-build-isolation .` against the PyTorch you run with. |
| `CUDA_FORCE_PTX_JIT is set ...`, or `no kernel image is available` from any PyTorch op | The environment variable `CUDA_FORCE_PTX_JIT=1` (left behind by some old TensorFlow / CUDA guides) makes the driver ignore compiled GPU code; neither this extension nor PyTorch's wheels ship sm_120 PTX. Remove the variable. |
| `torch.compile` fails with `cannot import name ... from 'torch._inductor...'` | Leftover files from another PyTorch version in `site-packages/torch` (a downgrade over a newer install). Reinstall PyTorch into a clean folder. |
| `nvfp4 kernels run on compute capability (12, 0) ... only` | Not an RTX 50-series GPU. Use `backend="emulate"` or BF16. |
| `NVFP4Linear needs in/out features divisible by 128` | Keep that layer in `nn.Linear`; `convert_linears` skips such layers automatically. |
| `x must be bf16 or fp32` (low-level ops) | Cast first; `NVFP4Linear` casts for you. |
| `CUDA error: unspecified launch failure` from the GEMM | A bounded pipeline wait expired (a bug or a GPU fault). Re-run `tools/sanitize.sh` and report it. |
| Loss becomes NaN | Exactly as in BF16 training: a non-finite value reached a quantized tensor (by design it is not hidden). |
