# Using nvfp4-train-sm120 in a training stack

This guide is for integrating the library into real training code. [DESIGN.md](DESIGN.md) explains how it
works and [RESULTS.md](RESULTS.md) what it measured.

## 1. Requirements

| | |
|---|---|
| GPU | compute capability **12.0** exactly: GeForce RTX 5090 / 5080 / 5070 Ti / 5070 / 5060 Ti / 5060 and their laptop parts. The kernels are built for the arch-specific target `sm_120a`, which runs on no other GPU (not on sm_100 B200, sm_121 GB10, or older GPUs). |
| OS | Linux or WSL2. Native Windows builds have not been tried. |
| Toolkit | CUDA ≥ 12.8 (TMA descriptor API by version needs ≥ 12.5; tested with 13.0). Driver tested: 595.79. |
| PyTorch | a CUDA build, tested with 2.14.1+cu130 (needs `-std=c++20` for extensions). The extension must be built against the same PyTorch it runs with. |

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

Verify an installation on the target machine:

```bash
tools/check.sh            # build + 305 tests (~10 s of GPU time)
tools/check.sh release    # + compute-sanitizer (memcheck, racecheck, synccheck, initcheck) + 5-minute soak
```

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
| `fuse_wgrad_accumulation` | `False` | Add the weight gradient into an existing fp32 `weight.grad` inside the GEMM. Saves a read-modify-write per micro-batch, but autograd then never sees that gradient: per-parameter hooks don't fire (DDP's gradient bucketing relies on them) and `torch.autograd.grad` returns nothing for the weight. Use only with single-GPU loops that read `.grad`. |
| `backend` | `"cuda"` | `"emulate"` runs the identical recipe with the pure-PyTorch reference on any device (slow; for verification). |

### 3.2 Pre-LN blocks with fused LayerNorm / GELU

```python
from nvfp4 import layernorm_linear, layernorm_mlp

qkv = layernorm_linear(x, block.ln1, block.qkv)          # block.qkv(block.ln1(x))
y = x + layernorm_mlp(x, block.ln2, block.fc1, block.fc2) # fc2(gelu_tanh(fc1(ln2(x))))
```

The modules stay ordinary `nn.LayerNorm` / `NVFP4Linear` modules (same parameters, same state dict);
only the forward call changes. Requirements: bias-free `NVFP4Linear` with `backend="cuda"`, a
LayerNorm over the last dimension, tanh-approximate GELU. `experiments/train_gpt.py --fuse-producers`
shows a full model (`Block.forward` and `fuse_producers`).

### 3.3 Low-level ops

```python
from nvfp4 import quantize, quantize_dual, gemm
qa = quantize(a)                                   # NVFP4Tensor(data, scales, gdec), along the last dim
qb, qbt = quantize_dual(b, block_2d=True)          # b and b^T from one read
d = gemm(qa, qb)                                   # a @ b^T, bf16 (or fp32 / into `out`, accumulate=True)
```

## 4. Behaviour you can rely on

- **Bit-exact quantization.** Every quantizer (including the fused LayerNorm / GELU variants) is
  bit-identical to `nvfp4.reference`, the executable spec; the GEMM matches an fp64 reference to ~1e-7.
- **Determinism.** With the same seed (`torch.manual_seed`), forward and backward are bit-reproducible,
  including stochastic rounding and split-K. Note that PyTorch's own flash / memory-efficient attention
  backward is *not* (it accumulates with atomics); use the math SDPA backend if you need bitwise
  reproducible runs (`tools/soak.py` does).
- **Non-finite values are never hidden.** A NaN or ±inf anywhere in a quantized tensor makes its
  per-tensor scale non-finite, so every output that tensor feeds is non-finite. A diverging run shows
  NaN/inf in its loss exactly as in BF16 (the whole GEMM output becomes non-finite, not just one row).
- **Activation checkpointing** (`torch.utils.checkpoint`, both modes) reproduces gradients bit for bit:
  random seeds come from the CPU RNG, whose state checkpointing restores.
- **Streams.** All kernels run on the current CUDA stream and allocate through PyTorch's caching
  allocator; concurrent streams are safe (split-K workspaces are per call). As always in PyTorch,
  synchronize streams yourself when one consumes another's output.
- **Multi-GPU.** One process per GPU (DDP) works. Kernel attributes are set per device, so one process
  driving several GPUs works too.
- **Hangs become errors.** GEMM pipeline waits are bounded (1 s, then `trap`), so a fault surfaces as
  a CUDA error instead of a display-driver watchdog reset.

## 5. Things to know in production

| Topic | What happens |
|---|---|
| `torch.compile` | Works and matches eager bit for bit, with a graph break at each NVFP4 layer (the layers are custom autograd Functions; the seed draw uses `.item()`). The surrounding ops are still compiled. |
| CUDA graphs | Inference capture/replay works. For training capture, note that the stochastic-rounding / RHT seeds are drawn on the host and would be frozen into the graph (every replay uses the same seeds). |
| DDP | Works (gradients are ordinary `.grad` tensors). Do not combine with `fuse_wgrad_accumulation=True` (see §3.1). FSDP has not been tested. |
| Memory | Saved activations are NVFP4 (~0.56 bytes per element) instead of BF16; peak memory of a block step is 7–14% below BF16. |
| Shapes | `in_features` / `out_features` must be multiples of 128. Token counts are arbitrary (padded internally). Very small layers (< 128 wide) should stay BF16. |
| Precision recipe | Keep embeddings, the LM head and the last 1–2 blocks in BF16. Measured quality gap at 124M / 507M parameters: +1.0–1.1% validation loss, constant over training (RESULTS.md §4–5). Validate on your own model before relying on it. |
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
| `nvfp4 kernels run on compute capability (12, 0) ... only` | Not an RTX 50-series GPU. Use `backend="emulate"` or BF16. |
| `NVFP4Linear needs in/out features divisible by 128` | Keep that layer in `nn.Linear`; `convert_linears` skips such layers automatically. |
| `x must be bf16 or fp32` (low-level ops) | Cast first; `NVFP4Linear` casts for you. |
| `CUDA error: unspecified launch failure` from the GEMM | A bounded pipeline wait expired (a bug or a GPU fault). Re-run `tools/sanitize.sh` and report it. |
| Loss becomes NaN | Exactly as in BF16 training: a non-finite value reached a quantized tensor (by design it is not hidden). |
