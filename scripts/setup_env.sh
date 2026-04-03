#!/usr/bin/env bash

# Installer script to automatically create a conda environment and install project dependencies on Linux
# Usage:
#   ./scripts/setup_env.sh [--env-name NAME|-n NAME] [--feature gpu|cpu|dhg-gpu|dhg-cpu] [--force] [--yes]
# Default behavior: auto-detect GPU (use gpu feature if nvidia-smi is available), otherwise cpu

set -euo pipefail

# ==============================================================================
# Version Configuration
# ==============================================================================
# Common versions (shared across all features)
TORCH_VER="2.10.0"
PYG_LIB_VER="==0.6.0"
TORCH_SCATTER_VER="==2.1.2"
TORCH_GEOMETRIC_VER="==2.7.0"
POLARS_VER=">=1.39.3,<2"
PANDAS_VER="==3.0.2"
SKLEARN_VER="==1.8.0"
OPTUNA_VER="==4.8.0"
PYARROW_VER=">=23.0.0,<24"
PYTHON_DOTENV_VER=">=1.2.1,<2"
RUFF_VER=">=0.15,<0.16"
PYTEST_VER=">=9.0.2,<10"
SEABORN_VER=">=0.13.2,<0.14"
MATPLOTLIB_VER=">=3.10.8,<4"
SWANLAB_VER=">=0.7.13,<0.8"
DHG_VER=""
PY_VER="3.12"
CUDA_VER="cu128"

# DHG-specific version overrides (applied when feature is dhg-gpu or dhg-cpu)
DHG_TORCH_VER="1.13.1"
DHG_PYG_LIB_VER=">=0.4.0,<0.5"
DHG_TORCH_SCATTER_VER="==2.1.1"
DHG_TORCH_GEOMETRIC_VER=">=2.7.0,<3"
DHG_POLARS_VER=">=1.38.1,<2"
DHG_PANDAS_VER=">=2.3.3,<3"
DHG_SKLEARN_VER=">=1.7.2,<2"
DHG_OPTUNA_VER=">=4.6.0,<5"
DHG_PYARROW_VER=">=12.0.1,<13"
DHG_PY_VER="3.10"
DHG_CUDA_VER_GPU="cu117"
DHG_CUDA_VER_CPU="cpu"
DHG_DHGA_VER="==0.9.*"

# ==============================================================================
# Runtime Variables
# ==============================================================================
ENV_NAME="kt-exp"
FEATURE=""
FORCE=0
ASSUME_YES=0
FORCE_CPU=0
FORCE_GPU=0

usage() {
  cat <<EOF
Usage: $0 [--env-name NAME|-n NAME] [--feature gpu|cpu|dhg-gpu|dhg-cpu] [--force] [--yes]

Options:
  -n, --env-name NAME   Specify the conda environment name (default: kt-exp)
  --feature FEATURE     Specify feature to install: gpu, cpu, dhg-gpu, dhg-cpu
  --cpu                 Force install cpu feature
  --gpu                 Force install gpu feature
  --force               Remove and recreate environment if it already exists
  --yes                 Non-interactive; assume yes to prompts
  -h, --help            Show this help message

Examples:
  $0 -n myenv --feature cpu               # Create a CPU environment named 'myenv'
  $0 --env-name myenv --feature dhg-gpu   # Create a CUDA/dhg-gpu environment named 'myenv'

EOF
}

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--env-name) ENV_NAME="$2"; shift 2 ;;
    --feature) FEATURE="$2"; shift 2 ;;
    --cpu) FORCE_CPU=1; shift ;;
    --gpu) FORCE_GPU=1; shift ;;
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
USE_EXISTING=0
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  if [ "$FORCE" -eq 1 ]; then
    echo "Environment $ENV_NAME already exists; --force specified: removing and recreating."
    conda remove -n "$ENV_NAME" --all -y
  else
    if [ "$ASSUME_YES" -eq 1 ]; then
      echo "Environment $ENV_NAME already exists; using existing environment (--yes specified)."
      USE_EXISTING=1
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

# Determine feature if not specified
if [ -z "$FEATURE" ]; then
  if [ "$FORCE_GPU" -eq 1 ]; then
    FEATURE="gpu"
  elif [ "$FORCE_CPU" -eq 1 ]; then
    FEATURE="cpu"
  else
    # Auto-detect presence of NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "NVIDIA GPU detected (nvidia-smi available); using 'gpu' feature."
      FEATURE="gpu"
    else
      echo "No NVIDIA GPU detected; using 'cpu' feature."
      FEATURE="cpu"
    fi
  fi
fi

# Validate feature
case "$FEATURE" in
  gpu|cpu|dhg-gpu|dhg-cpu) ;;
  *) echo "Error: Invalid feature '$FEATURE'. Must be one of: gpu, cpu, dhg-gpu, dhg-cpu"; exit 1 ;;
esac

# Set configuration based on feature (override defaults for DHG environments)
case "$FEATURE" in
  cpu)
    CUDA_VER="cpu"
    ;;
  dhg-gpu)
    PY_VER="$DHG_PY_VER"
    CUDA_VER="$DHG_CUDA_VER_GPU"
    TORCH_VER="$DHG_TORCH_VER"
    PYG_LIB_VER="$DHG_PYG_LIB_VER"
    TORCH_SCATTER_VER="$DHG_TORCH_SCATTER_VER"
    TORCH_GEOMETRIC_VER="$DHG_TORCH_GEOMETRIC_VER"
    POLARS_VER="$DHG_POLARS_VER"
    PANDAS_VER="$DHG_PANDAS_VER"
    SKLEARN_VER="$DHG_SKLEARN_VER"
    OPTUNA_VER="$DHG_OPTUNA_VER"
    PYARROW_VER="$DHG_PYARROW_VER"
    DHG_VER="$DHG_DHGA_VER"
    ;;
  dhg-cpu)
    PY_VER="$DHG_PY_VER"
    CUDA_VER="$DHG_CUDA_VER_CPU"
    TORCH_VER="$DHG_TORCH_VER"
    PYG_LIB_VER="$DHG_PYG_LIB_VER"
    TORCH_SCATTER_VER="$DHG_TORCH_SCATTER_VER"
    TORCH_GEOMETRIC_VER="$DHG_TORCH_GEOMETRIC_VER"
    POLARS_VER="$DHG_POLARS_VER"
    PANDAS_VER="$DHG_PANDAS_VER"
    SKLEARN_VER="$DHG_SKLEARN_VER"
    OPTUNA_VER="$DHG_OPTUNA_VER"
    PYARROW_VER="$DHG_PYARROW_VER"
    DHG_VER="$DHG_DHGA_VER"
    ;;
esac

# Create the environment (if it doesn't exist)
if [ "$USE_EXISTING" -eq 0 ]; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME (python=$PY_VER)"
    conda create -n "$ENV_NAME" python="$PY_VER" -y
  fi
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

# PyTorch index configuration
if [ "$CUDA_VER" = "cpu" ]; then
  TORCH_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/cpu"
  PYG_FIND_LINKS="-f https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html"
else
  TORCH_INDEX_URL="--extra-index-url https://download.pytorch.org/whl/${CUDA_VER}"
  PYG_FIND_LINKS="-f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_VER}.html"
fi

# Step 1: Install core dependencies first (pyg-lib, torch-scatter) before torch-geometric
# This ensures dependency resolution can find compatible versions
echo "Installing core PyPI packages (feature: $FEATURE)"
echo "  torch==${TORCH_VER}+${CUDA_VER}"
echo "  pyg_lib${PYG_LIB_VER}"
echo "  torch-scatter${TORCH_SCATTER_VER}"

pip_install "torch==${TORCH_VER}+${CUDA_VER}" ${TORCH_INDEX_URL}
pip_install "pyg_lib${PYG_LIB_VER}" ${PYG_FIND_LINKS}
pip_install "torch-scatter${TORCH_SCATTER_VER}" ${PYG_FIND_LINKS}

# Step 2: Install torch-geometric (depends on pyg-lib and torch-scatter)
echo "Installing torch-geometric${TORCH_GEOMETRIC_VER}"
pip_install "torch-geometric${TORCH_GEOMETRIC_VER}" ${PYG_FIND_LINKS}

# Step 3: Install conda-forge dependencies
echo "Installing dependencies from conda-forge"
conda install -c conda-forge -y \
  "optuna${OPTUNA_VER}" \
  "scikit-learn${SKLEARN_VER}" \
  "pandas${PANDAS_VER}" \
  "pyarrow${PYARROW_VER}" \
  "python-dotenv${PYTHON_DOTENV_VER}" \
  "ruff${RUFF_VER}" \
  "pytest${PYTEST_VER}" \
  "polars${POLARS_VER}" \
  "seaborn${SEABORN_VER}" \
  "matplotlib${MATPLOTLIB_VER}"

# Step 4: Install remaining PyPI packages (dhg if applicable, swanlab)
if [ -n "$DHG_VER" ]; then
  echo "Installing dhg${DHG_VER}"
  pip_install "dhg${DHG_VER}"
fi

echo "Installing swanlab${SWANLAB_VER}"
pip_install "swanlab${SWANLAB_VER}"

# Print version info for verification
echo "Installation completed — verification info:"
python - <<PY
import sys
import importlib
pkgs = ["torch", "torch_geometric", "torch_scatter", "pyg_lib", "optuna", "pandas", "pyarrow", "swanlab", "polars", "sklearn", "ruff", "pytest", "seaborn", "matplotlib"]
if "$DHG_VER":
    pkgs.append("dhg")
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
