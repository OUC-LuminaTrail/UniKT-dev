#!/bin/bash

# Resolve repo root so the script runs from any directory. train.py and
# `import model` both rely on being launched from here.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || { echo "Error: failed to cd to repository root: $REPO_ROOT"; exit 1; }

# Same interpreter that train.py needs; override with PYTHON="..." if needed.
PYTHON="${PYTHON:-python}"

# List every registered trainer by reusing the framework's static AST scan.
# Discovery parses source only (no model imports), so it runs in any pixi env
# and stays in sync with the codebase as models are added or removed.
# Word-split on purpose so PYTHON may carry args, e.g. PYTHON="pixi run python".
discover_models() {
    $PYTHON - <<'PY'
import model
from utils.core.registry import TRAINERS

print(" ".join(sorted(TRAINERS.keys())))
PY
}

print_usage() {
    cat <<EOF
Usage: $0 <model_name|ALL> <folds_list> [args...]
       $0 -l | --list

Run K-fold training across the framework's registered models.

  model_name   A single registered trainer (see --list), or ALL for every model.
  folds_list   Whitespace- or comma-separated fold indices, e.g. "0 1 2 3 4".
  args...      Extra flags forwarded to train.py.

Options:
  -l, --list   Print all available models and exit.

Examples:
  $0 GIKT "0 1 2" -d assistments09 --model.epochs 100
  $0 ALL "0 1 2 3 4" -d assistments09 --model.epochs 150
  $0 --list
EOF
}

# Run the requested folds for one model. Returns its exit code.
run_kfold_for_model() {
    local model=$1
    local folds=$2

    echo "=================================================="
    echo "Starting Training for Model: $model"
    echo "Folds: $folds"
    echo "Arguments: $ARGS"
    echo "=================================================="

    for fold in $folds; do
        echo ""
        echo "----------------------------------------"
        echo "Running Fold $fold for $model"
        echo "----------------------------------------"

        $PYTHON train.py -m "$model" --data.fold "$fold" $ARGS
        local rc=$?
        if [ $rc -ne 0 ]; then
            echo "Error: Fold $fold failed for $model (exit code $rc)."
            return $rc
        fi
    done

    echo ""
    echo "=================================================="
    echo "All specified folds ($folds) completed successfully for $model."
    echo "=================================================="
}

# --list: print every model and exit.
if [ "$1" == "-l" ] || [ "$1" == "--list" ]; then
    MODELS="$(discover_models)" || {
        echo "Error: failed to discover models via '$PYTHON'. Activate your pixi env or set PYTHON=..." >&2
        exit 1
    }
    [ -n "$MODELS" ] || { echo "Error: no models discovered." >&2; exit 1; }
    echo "Available models ($(echo "$MODELS" | wc -w) discovered via static registry scan):"
    echo "$MODELS" | tr ' ' '\n'
    exit 0
fi

# Normal usage: <model_name|ALL> <folds_list> [args...].
if [ "$#" -lt 2 ]; then
    print_usage
    exit 1
fi

MODEL_NAME=$1
FOLDS_ARG=$2
shift 2
ARGS="$@"

# Accept either "0 1 2" or "0,1,2".
FOLDS_LIST=${FOLDS_ARG//,/ }

# Resolve the model set: ALL -> every registered trainer; otherwise one model.
if [ "$MODEL_NAME" == "ALL" ]; then
    echo "Running ALL models..."
    MODELS="$(discover_models)" || {
        echo "Error: failed to discover models via '$PYTHON'. Activate your pixi env or set PYTHON=..." >&2
        exit 1
    }
    [ -n "$MODELS" ] || { echo "Error: no models discovered." >&2; exit 1; }
else
    MODELS="$MODEL_NAME"
fi

for model in $MODELS; do
    run_kfold_for_model "$model" "$FOLDS_LIST"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "Stopping execution due to failure in model $model." >&2
        exit $rc
    fi
done
