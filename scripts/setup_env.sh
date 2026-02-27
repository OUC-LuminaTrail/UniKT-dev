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

usage() {
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
      # Present options to the user. Default action is to exit.
      while true; do
        echo "Environment '$ENV_NAME' already exists. Choose an action:"
        echo "  1) Force recreate the environment (delete and recreate)"
        echo "  2) Continue and install into the existing environment"
        echo "  3) Enter a new environment name"
        echo "  4) Exit (default)"
        read -rp "Select 1/2/3/4 [4]: " choice
        choice=${choice:-4}
        case "$choice" in
          1)
            echo "Removing environment '$ENV_NAME'..."
            conda remove -n "$ENV_NAME" --all -y
            # After removal, the script will create the env below
            break
            ;;
          2)
            echo "Using existing environment '$ENV_NAME'."
            USE_EXISTING=1
            break
            ;;
          3)
            read -rp "Enter new environment name: " newname
            newname=${newname:-}
            if [ -z "$newname" ] || [[ "$newname" =~ [[:space:]] ]]; then
              echo "Invalid name. Name must be non-empty and contain no spaces."
              continue
            fi
            ENV_NAME="$newname"
            # If the new name exists, loop again to present options for the new name
            if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
              echo "Environment '$ENV_NAME' already exists; repeating options for the new name."
              continue
            else
              # New name does not exist; proceed to create it below
              break
            fi
            ;;
          4)
            echo "Exiting without changes."
            exit 0
            ;;
          *)
            echo "Invalid selection; please enter 1, 2, 3, or 4."
            ;;
        esac
      done
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

# PyPI-only packages (torch, torch-geometric, pyg-lib, dhg, swanlab)
if [ "$INSTALL_TARGET" = "cpu" ]; then
  echo "Installing PyPI packages (CPU version): torch, torch_geometric, pyg_lib"
  pip_install "torch==1.13.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu
  pip_install "torch_geometric>=2.7.0,<3" "pyg-lib>=0.4.0,<0.5" -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
else
  echo "Installing PyPI packages (CUDA version): torch, torch_geometric, pyg_lib"
  pip_install "torch==1.13.1+$CUDA_VER" --extra-index-url https://download.pytorch.org/whl/$CUDA_VER
  pip_install "torch_geometric>=2.7.0,<3" "pyg-lib>=0.4.0,<0.5" -f https://data.pyg.org/whl/torch-1.13.1+$CUDA_VER.html || {
    # Fallback install attempt (indexes or link formats may differ)
    pip_install "torch_geometric>=2.7.0,<3" "pyg-lib>=0.4.0,<0.5" -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
  }
fi

# Conda-forge dependencies
echo "Installing dependencies from conda-forge: optuna, scikit-learn, pandas, pyarrow, python-dotenv, ruff, pytest, polars"
conda install -c conda-forge -y "optuna>=4.6.0,<5" "scikit-learn>=1.7.2,<2" "pandas>=2.3.3,<3" "pyarrow>=12.0.1,<13" "python-dotenv>=1.2.1,<2" "ruff>=0.15,<0.16" "pytest>=9.0.2,<10" "polars>=1.38.1,<2"

# PyPI-only dependencies (dhg, swanlab)
echo "Installing remaining PyPI packages: dhg, swanlab"
pip_install "dhg==0.9.*" "swanlab<0.8"

# Print version info for verification
echo "Installation completed — verification info:"
python - <<PY
import sys
import importlib
pkgs = ["torch", "torch_geometric", "dhg", "optuna", "pandas", "pyarrow", "swanlab", "polars", "sklearn", "ruff", "pytest"]
for p in pkgs:
    try:
        if p == "sklearn":
            m = importlib.import_module("sklearn")
        else:
            m = importlib.import_module(p)
        v = getattr(m, '__version__', str(m))
        print(f"{p}: {v}")
    except Exception as e:
        print(f"{p}: import failed ({e})")

import torch
print('torch.cuda.is_available():', torch.cuda.is_available())
PY

echo "✅ Environment is ready. To use it:"
echo "   conda activate $ENV_NAME"

exit 0