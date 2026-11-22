"""Build: pip install -e . --no-build-isolation   (needs CUDA >= 12.8 nvcc and a CUDA build of PyTorch).

Set NVFP4_SKIP_CUDA=1 to install only the pure-Python reference (used by CI without a GPU).
"""
import os

from setuptools import find_packages, setup

ext_modules, cmdclass = [], {}
if os.environ.get("NVFP4_SKIP_CUDA") != "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # The block-scaled MMA is an arch-specific feature: it needs the 'a' target (sm_120a).
    arch = os.environ.get("NVFP4_ARCH", "120a")
    ext_modules = [
        CUDAExtension(
            name="nvfp4._C",
            sources=["csrc/bindings.cpp", "csrc/quantize.cu", "csrc/quantize_dual.cu", "csrc/gemm.cu", "csrc/gemm_tma.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                         "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v,-warn-spills"],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="nvfp4-train-sm120",
    version="0.1.0",
    description="NVFP4 training kernels for consumer Blackwell (sm_120)",
    packages=find_packages(include=["nvfp4", "nvfp4.*"]),
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
