import pytest
import torch


def _cuda_ok() -> bool:
    try:
        from nvfp4 import ops
        return ops.available()
    except Exception:  # noqa: BLE001
        return False


requires_sm120 = pytest.mark.skipif(not _cuda_ok(), reason="needs the built extension and an sm_120 GPU")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
