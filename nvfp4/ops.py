"""Thin Python API over the CUDA kernels. Each op has a pure-PyTorch twin in ``nvfp4.reference``."""
from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from nvfp4 import _C  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CPU-only installs
    _C = None


def available() -> bool:
    return _C is not None and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (12, 0)


@dataclass
class NVFP4Tensor:
    """A 2D tensor quantized along its last dim: value = e2m1(data) * e4m3(scales) * gdec."""

    data: torch.Tensor    # uint8 [R, C/2]
    scales: torch.Tensor  # uint8 [R, C/16]
    gdec: torch.Tensor    # fp32 [1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1] * 2

    def as_tuple(self):
        return self.data, self.scales, self.gdec.reshape(())

    def dequantize(self) -> torch.Tensor:
        from nvfp4 import reference
        return reference.dequantize(self.data, self.scales, self.gdec.reshape(()))


def quantize(x: torch.Tensor, *, transpose: bool = False, block_2d: bool = False, stochastic: bool = False,
             sr_seed: int = 0, rht_seed: int | None = None) -> NVFP4Tensor:
    """Quantize ``x`` (or ``x.T`` if ``transpose``) along the last dim. ``rht_seed`` enables the 16-point RHT."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    if not x.is_contiguous():
        x = x.contiguous()
    d, s, g = _C.quantize(x, transpose, block_2d, stochastic, int(sr_seed), rht_seed is not None,
                          int(rht_seed or 0))
    return NVFP4Tensor(d, s, g)


def gemm(a: NVFP4Tensor, b: NVFP4Tensor, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """D[M, N] = A[M, K] @ B[N, K]^T with both operands in NVFP4 (block-scaled tensor-core MMA)."""
    if _C is None:
        raise RuntimeError("nvfp4 CUDA extension not built")
    return _C.gemm(a.data, a.scales, a.gdec, b.data, b.scales, b.gdec, out_dtype)
