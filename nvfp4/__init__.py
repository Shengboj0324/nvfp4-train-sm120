"""nvfp4: NVFP4 (4-bit) training kernels for consumer Blackwell GPUs (sm_120).

Layers:   NVFP4Linear, convert_linears (nvfp4.linear); layernorm_linear, layernorm_mlp, rmsnorm_linear,
          rmsnorm_swiglu_mlp, gelu_mlp, swiglu_mlp (nvfp4.fused); attention (nvfp4.attention: 8-bit
          flash attention); MoE (nvfp4.moe: mixture of experts with NVFP4 experts)
Ops:      quantize, quantize_dual (+ LayerNorm / RMSNorm / GELU / SwiGLU fused variants), gemm (with a
          bias / residual epilogue), norm_backward (nvfp4.ops)
Spec:     nvfp4.reference (pure PyTorch, runs anywhere)
"""
from nvfp4 import reference
from nvfp4.attention import attention
from nvfp4.fused import gelu_mlp, layernorm_linear, layernorm_mlp, rmsnorm_linear, rmsnorm_swiglu_mlp, swiglu_mlp
from nvfp4.linear import NVFP4Linear, WeightQuantCache, convert_linears, invalidate_weight_caches
from nvfp4.moe import MoE
from nvfp4.ops import (NVFP4Tensor, available, gelu_backward_quantize_dual, gelu_quantize_dual, gemm,
                       layernorm_quantize_dual, norm_backward, quantize, quantize_dual, rmsnorm_quantize_dual,
                       swiglu_backward_quantize_dual, swiglu_quantize_dual)

__version__ = "0.17.0"
__all__ = ["reference", "attention", "MoE", "NVFP4Tensor", "NVFP4Linear", "WeightQuantCache", "available", "convert_linears",
           "invalidate_weight_caches", "gelu_backward_quantize_dual", "gelu_mlp", "norm_backward", "swiglu_mlp",
           "gelu_quantize_dual", "gemm", "layernorm_linear", "layernorm_mlp", "layernorm_quantize_dual", "quantize",
           "quantize_dual", "rmsnorm_linear", "rmsnorm_quantize_dual", "rmsnorm_swiglu_mlp",
           "swiglu_backward_quantize_dual", "swiglu_quantize_dual", "__version__"]
