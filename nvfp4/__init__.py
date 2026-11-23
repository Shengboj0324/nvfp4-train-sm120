"""nvfp4: NVFP4 (4-bit) training kernels for consumer Blackwell GPUs (sm_120).

Layers:   NVFP4Linear, convert_linears (nvfp4.linear); layernorm_linear, layernorm_mlp (nvfp4.fused)
Ops:      quantize, quantize_dual (+ LayerNorm / GELU fused variants), gemm (nvfp4.ops)
Spec:     nvfp4.reference (pure PyTorch, runs anywhere)
"""
from nvfp4 import reference
from nvfp4.fused import layernorm_linear, layernorm_mlp
from nvfp4.linear import NVFP4Linear, convert_linears
from nvfp4.ops import (NVFP4Tensor, available, gelu_backward_quantize_dual, gelu_quantize_dual, gemm,
                       layernorm_quantize_dual, quantize, quantize_dual)

__version__ = "0.6.0"
__all__ = ["reference", "NVFP4Tensor", "NVFP4Linear", "available", "convert_linears", "gelu_backward_quantize_dual",
           "gelu_quantize_dual", "gemm", "layernorm_linear", "layernorm_mlp", "layernorm_quantize_dual", "quantize",
           "quantize_dual", "__version__"]
