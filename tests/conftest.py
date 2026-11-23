import os

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
except ImportError:
    COMPILE_BACKEND = "aot_eager"


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
