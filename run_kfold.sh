#!/bin/bash

# 检查参数数量
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model_name|ALL> <folds_list> [args...]"
    echo "Supported models: GIKT, SQGKT"
    echo "Example: $0 GIKT \"0 1 2\" --dataset assistments09 --epochs 100"
    echo "Example: $0 ALL \"0 1 2\" --dataset assistments09 --epochs 100"
    exit 1
fi

MODEL_NAME=$1
FOLDS_ARG=$2
shift 2
ARGS="$@"

# 处理折数列表 (支持空格或逗号分隔)
FOLDS_LIST=${FOLDS_ARG//,/ }

# 获取模型对应的脚本
get_model_script() {
    local model=$1
    if [ "$model" == "GIKT" ]; then
        echo "train_gikt.py"
    elif [ "$model" == "SQGKT" ]; then
        echo "train_sqgkt.py"
    elif [ "$model" == "HGIKT" ]; then
        echo "train_hgikt.py"
    else
        echo ""
    fi
}

# 运行单个模型的指定折
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
    echo "Starting Training for Model: $model"
    echo "Script: $script"
    echo "Folds: $FOLDS_LIST"
    echo "Arguments: $ARGS"
    echo "=================================================="

    for fold in $FOLDS_LIST; do
        echo ""
        echo "----------------------------------------"
        echo "Running Fold $fold for $model"
        echo "----------------------------------------"
        
        # 运行 python 脚本，传入 --fold 参数和其他参数
        python "$script" --fold "$fold" $ARGS
        
        # 检查退出代码
        if [ $? -ne 0 ]; then
            echo "Error: Fold $fold failed for $model with exit code $?."
            return 1
        fi
    done
    
    echo ""
    echo "=================================================="
    echo "All specified folds ($FOLDS_LIST) completed successfully for $model."
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
