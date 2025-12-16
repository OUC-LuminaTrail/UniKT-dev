#!/usr/bin/env bash

# Installer script to automatically create a conda environment and install project dependencies on Linux
# Usage:
#   ./scripts/setup_env.sh [--env-name NAME|-n NAME] [--cpu] [--cuda] [--cuda-ver cu117] [--force] [--yes]
# Default behavior: auto-detect GPU (use CUDA/cu117 if nvidia-smi is available), otherwise install CPU packages

set -euo pipefail

ENV_NAME="ktexp"
PY_VER="3.10"
DEFAULT_CUDA_VER="cu117"
FORCE=0
ASSUME_YES=0
FORCE_CPU=0
FORCE_CUDA=0
CUDA_VER="$DEFAULT_CUDA_VER"

function usage() {
  cat <<EOF
Usage: $0 [--env-name NAME|-n NAME] [--cpu] [--cuda] [--cuda-ver cu117] [--force] [--yes]

Options:
  -n, --env-name NAME   Specify the conda environment name (default: ktexp)
  --cpu                 Force install CPU builds
  --cuda                Force install CUDA builds (default CUDA version: cu117)
  --cuda-ver VER        Specify CUDA version (e.g., cu117)
  --force               Remove and recreate environment if it already exists
  --yes                 Non-interactive; assume yes to prompts
  -h, --help            Show this help message

Examples:
  $0 -n myenv --cpu               # Create a CPU environment named 'myenv'
  $0 --env-name myenv --cuda --cuda-ver cu117  # Create a CUDA/cu117 environment named 'myenv'

EOF
}

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--env-name) ENV_NAME="$2"; shift 2 ;;
    --cpu) FORCE_CPU=1; shift ;;
    --cuda) FORCE_CUDA=1; shift ;;
    --cuda-ver) CUDA_VER="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Validate environment name (must be non-empty and contain no spaces)
if [ -z "$ENV_NAME" ] || [[ "$ENV_NAME" =~ [[:space:]] ]]; then
  echo "Error: Invalid environment name '$ENV_NAME'. Name must be non-empty and contain no spaces."
  exit 1
fi

# Check that conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found. Please install Anaconda/Miniconda and ensure conda is in PATH."
  exit 1
fi

# Source conda 脚本以便在非交互 shell 中使用 conda activate
CONDA_BASE=$(conda info --base 2>/dev/null || true)
if [ -z "$CONDA_BASE" ]; then
  echo "Could not determine conda base path; please check your conda installation."
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# If the environment already exists, prompt or handle according to --force
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [ "$FORCE" -eq 1 ]; then
    echo "Environment $ENV_NAME already exists; --force specified: removing and recreating."
    conda remove -n "$ENV_NAME" --all -y
  else
    if [ "$ASSUME_YES" -eq 1 ]; then
      echo "Environment $ENV_NAME already exists; skipping creation (--yes specified)."
    else
      read -rp "Environment $ENV_NAME already exists. Recreate? [y/N]: " yn
      yn=${yn:-N}
      if [[ "$yn" =~ ^[Yy]$ ]]; then
        conda remove -n "$ENV_NAME" --all -y
      else
        echo "Using existing environment $ENV_NAME."
        # Activate and continue installing dependencies
      fi
    fi
  fi
fi

# Create the environment (if it doesn't exist)
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda environment: $ENV_NAME (python=$PY_VER)"
  conda create -n "$ENV_NAME" python="$PY_VER" -y
fi

# Activate environment
conda activate "$ENV_NAME"

# Check if 'uv' is installed
USE_UV=0
if command -v uv >/dev/null 2>&1; then
  echo "Detected 'uv' command; 'uv pip install' will be used to accelerate package installation."
  USE_UV=1
fi

pip_install() {
  if [ "$USE_UV" -eq 1 ]; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

# Upgrade pip, setuptools, wheel
pip_install --upgrade pip setuptools wheel

# Decide whether to install CPU or CUDA versions
INSTALL_TARGET="cpu"
if [ "$FORCE_CPU" -eq 1 ]; then
  INSTALL_TARGET="cpu"
elif [ "$FORCE_CUDA" -eq 1 ]; then
  INSTALL_TARGET="cuda"
else
  # Auto-detect presence of NVIDIA GPU (nvidia-smi available)
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA GPU detected (nvidia-smi available); will install CUDA version (default: $CUDA_VER)."
    INSTALL_TARGET="cuda"
  else
    echo "No NVIDIA GPU detected; installing CPU version."
    INSTALL_TARGET="cpu"
  fi
fi

# Installation steps
if [ "$INSTALL_TARGET" = "cpu" ]; then
  echo "Installing CPU dependencies (torch cpu + torch_geometric, etc.)..."
  pip_install "torch==1.13.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
  pip_install "torch_geometric" "pyg-lib" -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
else
  echo "Installing CUDA($CUDA_VER) dependencies (torch + torch_geometric, etc.)..."
  pip_install "torch==1.13.1+$CUDA_VER" --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
  pip_install "torch_geometric" "pyg-lib" -f https://data.pyg.org/whl/torch-1.13.1+$CUDA_VER.html || {
    # Fallback install attempt (indexes or link formats may differ)
    pip_install "torch_geometric" "pyg-lib" -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
  }
fi

# Common dependencies
echo "Installing common Python packages: dhg, optuna, pandas, pyarrow, swanlab, python-dotenv..."
pip_install dhg optuna pandas pyarrow swanlab python-dotenv

# Print version info for verification
echo "\nInstallation completed — verification info:"
python - <<PY
import sys
import importlib
pkgs = ["torch", "torch_geometric", "dhg", "optuna", "pandas", "pyarrow", "swanlab"]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', str(m))
        print(f"{p}: {v}")
    except Exception as e:
        print(f"{p}: import failed ({e})")

import torch
print('torch.cuda.is_available():', torch.cuda.is_available())
PY

echo "\n✅ Environment is ready. To use it:"
echo "  conda activate $ENV_NAME"
echo "To rerun this script, ensure it is executable: chmod +x $0"

exit 0
