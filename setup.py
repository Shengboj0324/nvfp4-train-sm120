"""Build: pip install -e . --no-build-isolation   (needs CUDA >= 12.8 nvcc and a CUDA build of PyTorch).

Set NVFP4_SKIP_CUDA=1 to install only the pure-Python reference (used by CI without a GPU).
"""
import os
import re

from setuptools import find_packages, setup

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = re.search(r'__version__ = "([^"]+)"', open(os.path.join(HERE, "nvfp4", "__init__.py")).read()).group(1)

ext_modules, cmdclass = [], {}
if os.environ.get("NVFP4_SKIP_CUDA") != "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # The block-scaled MMA is an arch-specific feature: it needs the 'a' target (sm_120a).
    arch = os.environ.get("NVFP4_ARCH", "120a")
    if os.name == "nt":  # MSVC host compiler (run from a VS 2022 x64 developer environment: tools/build.cmd)
        cxx = ["/O2", "/std:c++20", "/Zc:__cplusplus", "/permissive-"]
        host = ["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/permissive-"]
    else:
        cxx, host = ["-O3", "-std=c++20"], []
    ext_modules = [
        CUDAExtension(
            name="nvfp4._C",
            sources=["csrc/bindings.cpp", "csrc/quantize.cu", "csrc/quantize_dual.cu", "csrc/gemm.cu", "csrc/gemm_tma.cu",
                     "csrc/norm.cu", "csrc/attention.cu", "csrc/moe_combine.cu"],
            extra_compile_args={
                "cxx": cxx,
                "nvcc": ["-O3", "-std=c++20", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                         "--expt-relaxed-constexpr", "-lineinfo", "-Xptxas=-v,-warn-spills"] + host
                        + os.environ.get("NVFP4_NVCC_FLAGS", "").split(),  # ablations, e.g. -DNVFP4_WARPS_N=2
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="nvfp4-train-sm120",
    version=VERSION,
    description="NVFP4 (4-bit) training kernels for consumer Blackwell GPUs (RTX 50-series, sm_120)",
    long_description=open(os.path.join(HERE, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/Shengboj0324/nvfp4-train-sm120",
    packages=find_packages(include=["nvfp4", "nvfp4.*"]),
    python_requires=">=3.10",
    install_requires=["torch>=2.7"],  # the CUDA build needs the same torch it was compiled against
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Environment :: GPU :: NVIDIA CUDA :: 13",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
