import functools
import os
import subprocess
import sys

# torch.compile's inductor otherwise starts one compile worker per CPU; on a laptop that load spike is
# what tools/guard.py kills jobs for. Must be set before torch is imported.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import pytest  # noqa: E402
import torch  # noqa: E402


def _cuda_ok() -> bool:
    try:
        from nvfp4 import ops
        return ops.available()
    except Exception:  # noqa: BLE001
        return False


requires_sm120 = pytest.mark.skipif(not _cuda_ok(), reason="needs the built extension and an sm_120 GPU")

# torch.compile backend for the tests: inductor needs Triton, which PyTorch's Windows wheels don't ship.
# Without it, "aot_eager" still checks the parts that are this package's responsibility: dynamo traces
# the layers without graph breaks (fullgraph=True) and AOTAutograd handles the custom ops' backward.
try:
    import triton  # noqa: F401
    COMPILE_BACKEND = "inductor"
    HAS_TRITON = True
except ImportError:
    COMPILE_BACKEND = "aot_eager"
    HAS_TRITON = False

# the grouped BF16 MoE experts (nvfp4._bf16_grouped) are Triton kernels
requires_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton (not in PyTorch's Windows wheels)")


@functools.lru_cache(maxsize=None)
def gloo_cuda_reductions_ok() -> bool:
    """False if gloo's reductions of CUDA tensors crash this PyTorch build (PyTorch 2.11's Windows wheels die
    with an access violation in all_reduce, even with one rank; all_to_all works). Probed once, in a
    subprocess."""
    if sys.platform != "win32":
        return True
    code = ("import torch, torch.distributed as dist; "
            "s = dist.TCPStore('127.0.0.1', 0, 1, True, use_libuv=False); "
            "dist.init_process_group('gloo', store=s, rank=0, world_size=1); "
            "t = torch.ones(4, device='cuda'); dist.all_reduce(t); assert t.sum().item() == 4")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=300).returncode == 0
    except subprocess.TimeoutExpired:
        return False


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
