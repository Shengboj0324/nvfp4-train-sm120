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
tools/check.sh            # build + 528 tests (~30 s of GPU time); Windows: tools\check.cmd
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
autocast dtype (or the input dtype without autocast).

Options:

| Argument | Default | Meaning |
|---|---|---|
| `wgrad_rht` | `True` | 16-point random Hadamard transform on the wgrad operands (NVIDIA recipe). |
| `fuse_wgrad_accumulation` | `False` | Add the weight gradient into an existing fp32 `weight.grad` inside the GEMM, saving a read-modify-write of the gradient per micro-batch under gradient accumulation. Autograd still sees the gradient (the accumulated tensor is handed to it without a copy), so gradient hooks, DDP and FSDP2 work. A tensor hook on the weight receives the accumulated gradient rather than this micro-batch's part. Not applied under `torch.autograd.grad`, to non-fp32 gradients, or inside `torch.compile`, which take the ordinary path. |
| `cache_weight_quantization` | `False` | Keep the NVFP4 copies of the weight and reuse them while it is unchanged: every micro-batch after the first in a gradient-accumulation step skips the weight quantizer (2–6% of a block step), bit-identical results, ~1.1 bytes per weight element of memory. Invalidated by any `torch.optim` optimizer step (also fused ones, which don't bump version counters), by in-place changes through the parameter, `load_state_dict`, `.to()` and assigning `.data`. **Not** noticed: in-place writes through `param.data` outside an optimizer step (`p.data.add_(...)`); call `nvfp4.invalidate_weight_caches(model)` after those. Off automatically for FSDP1 (non-leaf views), inside `torch.compile`, and permanently for a layer captured in a CUDA graph. Gives no speedup under FSDP2, which gathers weights into fresh memory every forward (still correct). |
| `backend` | `"cuda"` | `"emulate"` runs the identical recipe with the pure-PyTorch reference on any device (slow; for verification). |

### 3.2 Pre-norm blocks with fused normalization and activation

```python
from nvfp4 import layernorm_linear, layernorm_mlp, rmsnorm_linear, rmsnorm_swiglu_mlp

# GPT-2 / BERT / ViT style (LayerNorm, GELU; biases allowed)
qkv = layernorm_linear(x, block.ln1, block.qkv)                       # block.qkv(block.ln1(x))
y = x + layernorm_mlp(x, block.ln2, block.fc1, block.fc2)             # fc2(gelu_tanh(fc1(ln2(x))))
y = x + layernorm_mlp(x, block.ln2, block.fc1, block.fc2, approximate="none")  # exact GELU (BERT, ViT)

# Llama / Mistral / Qwen style (RMSNorm, SwiGLU)
qkv = rmsnorm_linear(x, block.input_norm, block.qkv)                  # biases allowed (Qwen2)
y = x + rmsnorm_swiglu_mlp(x, block.post_norm, block.gate, block.up, block.down)
#                                         # down(silu(gate(n)) * up(n)), n = post_norm(x)
```

The modules stay ordinary `nn.LayerNorm` / `nn.RMSNorm` / `NVFP4Linear` modules (same parameters,
same state dict); only the forward call changes. Requirements: `NVFP4Linear` with `backend="cuda"`,
normalization over the last dimension. Biases are supported everywhere except on the SwiGLU gate and
up projections; fc1's bias is added inside the GELU kernel and its gradient comes from the same read.
The RMSNorm module may be `torch.nn.RMSNorm` (with or without weight, default eps) or any module with
`weight` and `variance_epsilon` (Hugging Face's `LlamaRMSNorm`). The normalized input is quantized
once for both the gate and the up projection. `experiments/train_gpt.py --fuse-producers` shows a full
GPT model; `bench/bench_llama_block.py` a Llama-style block.

### 3.3 Low-level ops

```python
from nvfp4 import quantize, quantize_dual, gemm
qa = quantize(a)                                   # NVFP4Tensor(data, scales, gdec), along the last dim
qb, qbt = quantize_dual(b, block_2d=True)          # b and b^T from one read
d = gemm(qa, qb)                                   # a @ b^T, bf16 (or fp32 / into `out`, accumulate=True)
```

Seeds (`row_sr_seed`, `col_sr_seed`, `rht_seed`) are Python ints or one-element int64 tensors on the
GPU; a tensor is read by the kernel, so the value can come from a GPU random generator without a host
sync, and changes on every CUDA-graph replay. Both give bit-identical results for the same value.

## 4. Behaviour you can rely on

- **Bit-exact quantization.** Every quantizer (including the fused LayerNorm / GELU variants) is
  bit-identical to `nvfp4.reference`, the executable spec; the GEMM matches an fp64 reference to ~1e-7.
- **Determinism.** With the same seed (`torch.manual_seed`), forward and backward are bit-reproducible,
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
  with two processes, on Linux and on Windows). Under FSDP1 the layers see views of the flat
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
| Fused blocks | GPT / BERT style (LayerNorm, tanh or exact GELU, biases) and Llama style (RMSNorm, SwiGLU). Attention itself stays PyTorch's BF16 SDPA; its output is quantized by `o_proj` as usual. |
| Precision recipe | Keep embeddings, the LM head and the last 1–2 blocks in BF16. Measured quality gap at 124M / 507M parameters: +1.0–1.4% validation loss, constant over training; at 124M over three seeds +0.019 ± 0.002 nats (+1.28%) (RESULTS.md §4–5). Validate on your own model before relying on it. |
| Throughput | Gains grow with width: block fwd+bwd 1.8× BF16 at d=768, 2.4× at d=2048, 3.1× at d=4096 on an RTX 5090 Laptop GPU. |

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
