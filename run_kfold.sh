#!/bin/bash

# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name|ALL> <k_folds> [args...]"
    echo "Supported models: GIKT, SQGKT"
    echo "Example: $0 GIKT 5 --dataset assistments09 --epochs 100"
    echo "Example: $0 ALL 5 --dataset assistments09 --epochs 100"
    exit 1
fi

MODEL_NAME=$1
K_FOLDS=$2
shift 2
ARGS="$@"

# 获取模型对应的脚本
get_model_script() {
    local model=$1
    if [ "$model" == "GIKT" ]; then
        echo "train_gikt.py"
    elif [ "$model" == "SQGKT" ]; then
        echo "train_sqgkt.py"
    else
        echo ""
    fi
}

# 运行单个模型的 K 折交叉验证
run_kfold_for_model() {
    local model=$1
    local script=$(get_model_script "$model")
    
    if [ -z "$script" ]; then
        echo "Error: Unknown model '$model'"
        return 1
    fi

    if [ ! -f "$script" ]; then
        echo "Error: Script '$script' not found for model '$model'"
        return 1
    fi

    echo "=================================================="
    echo "Starting K-Fold Training for Model: $model"
    echo "Script: $script"
    echo "Folds: $K_FOLDS"
    echo "Arguments: $ARGS"
    echo "=================================================="

    for ((i=0; i<K_FOLDS; i++)); do
        echo ""
        echo "----------------------------------------"
        echo "Running Fold $i / $((K_FOLDS-1)) for $model"
        echo "----------------------------------------"
        
        # 运行 python 脚本，传入 --fold 参数和其他参数
        python "$script" --fold "$i" $ARGS
        
        # 检查退出代码
        if [ $? -ne 0 ]; then
            echo "Error: Fold $i failed for $model with exit code $?."
            return 1
        fi
    done
    
    echo ""
    echo "=================================================="
    echo "All $K_FOLDS folds completed successfully for $model."
    echo "=================================================="
}

# 主逻辑
if [ "$MODEL_NAME" == "ALL" ]; then
    echo "Running ALL models..."
    # 定义所有支持的模型列表
    MODELS=("GIKT" "SQGKT")
    
    for model in "${MODELS[@]}"; do
        run_kfold_for_model "$model"
        if [ $? -ne 0 ]; then
            echo "Stopping execution due to failure in model $model."
            exit 1
        fi
    done
else
    run_kfold_for_model "$MODEL_NAME"
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi
