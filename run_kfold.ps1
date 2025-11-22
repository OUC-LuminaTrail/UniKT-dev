<#
.SYNOPSIS
    Runs K-Fold cross-validation for specified models.
.EXAMPLE
    .\run_kfold.ps1 GIKT 5 --dataset assistments09 --epochs 100
    .\run_kfold.ps1 ALL 5 --dataset assistments09 --epochs 100
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelName,

    [Parameter(Mandatory=$true, Position=1)]
    [int]$KFolds,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)

# 设置错误偏好，遇到错误不立即停止脚本，由逻辑控制
$ErrorActionPreference = "Continue"

# 获取模型对应的脚本
function Get-ModelScript {
    param([string]$Model)
    switch ($Model) {
        "GIKT" { return "train_gikt.py" }
        "SQGKT" { return "train_sqgkt.py" }
        Default { return "" }
    }
}

# 运行单个模型的 K 折交叉验证
function Run-KFoldForModel {
    param([string]$Model)

    $Script = Get-ModelScript -Model $Model

    if ([string]::IsNullOrEmpty($Script)) {
        Write-Host "Error: Unknown model '$Model'" -ForegroundColor Red
        return $false
    }

    if (-not (Test-Path $Script)) {
        Write-Host "Error: Script '$Script' not found for model '$Model'" -ForegroundColor Red
        return $false
    }

    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Starting K-Fold Training for Model: $Model" -ForegroundColor Cyan
    Write-Host "Script: $Script"
    Write-Host "Folds: $KFolds"
    Write-Host "Arguments: $ExtraArgs"
    Write-Host "==================================================" -ForegroundColor Cyan

    for ($i = 0; $i -lt $KFolds; $i++) {
        Write-Host ""
        Write-Host "----------------------------------------" -ForegroundColor Green
        Write-Host "Running Fold $i / $($KFolds - 1) for $Model" -ForegroundColor Green
        Write-Host "----------------------------------------" -ForegroundColor Green

        # 构造参数列表
        $PythonArgs = @($Script, "--fold", "$i") + $ExtraArgs
        
        # 运行 python 脚本
        & python $PythonArgs

        # 检查退出代码
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Fold $i failed for $Model with exit code $LASTEXITCODE." -ForegroundColor Red
            return $false
        }
    }

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "All $KFolds folds completed successfully for $Model." -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    return $true
}

# 主逻辑
if ($ModelName -eq "ALL") {
    Write-Host "Running ALL models..." -ForegroundColor Yellow
    # 定义所有支持的模型列表
    $Models = @("GIKT", "SQGKT")
    
    foreach ($Model in $Models) {
        $Success = Run-KFoldForModel -Model $Model
        if (-not $Success) {
            Write-Host "Stopping execution due to failure in model $Model." -ForegroundColor Red
            exit 1
        }
    }
}
else {
    $Success = Run-KFoldForModel -Model $ModelName
    if (-not $Success) {
        exit 1
    }
}
