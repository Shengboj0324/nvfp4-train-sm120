#!/usr/bin/env bash
# One-time environment setup for WSL2 Ubuntu 22.04 (run as root or with sudo).
# Installs the CUDA 13.0 toolkit (nvcc with sm_120a support) and a venv with PyTorch cu130.
set -euo pipefail
VENV=${VENV:-$HOME/nvfp4}
if ! command -v /usr/local/cuda-13.0/bin/nvcc >/dev/null; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  # WSL: toolkit only, never the Linux driver (the Windows driver provides libcuda).
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cuda-toolkit-13-0 ninja-build python3.10-venv python3.10-dev
fi
/usr/local/cuda-13.0/bin/nvcc --version | tail -2
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q -U pip wheel setuptools
"$VENV/bin/pip" install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install -q numpy pytest ninja
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'archs', torch.cuda.get_arch_list())"
