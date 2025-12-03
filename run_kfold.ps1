<#
.SYNOPSIS
    Runs K-Fold cross-validation for specified models.
.EXAMPLE
    .\run_kfold.ps1 GIKT "0 1 2" --dataset assistments09 --epochs 100
    .\run_kfold.ps1 ALL "0 1 2" --dataset assistments09 --epochs 100
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelName,

    [Parameter(Mandatory=$true, Position=1)]
    [string]$FoldsStr,

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
        "HGIKT" { return "train_hgikt.py" }
        Default { return "" }
    }
}

# 运行单个模型的指定折
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

    # 解析折数列表
    $Folds = $FoldsStr -split "[, ]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "Starting Training for Model: $Model" -ForegroundColor Cyan
    Write-Host "Script: $Script"
    Write-Host "Folds: $Folds"
    Write-Host "Arguments: $ExtraArgs"
    Write-Host "==================================================" -ForegroundColor Cyan

    foreach ($fold in $Folds) {
        Write-Host ""
        Write-Host "----------------------------------------" -ForegroundColor Green
        Write-Host "Running Fold $fold for $Model" -ForegroundColor Green
        Write-Host "----------------------------------------" -ForegroundColor Green

        # 构造参数列表
        $PythonArgs = @($Script, "--fold", "$fold") + $ExtraArgs
        
        # 运行 python 脚本
        & python $PythonArgs

        # 检查退出代码
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Fold $fold failed for $Model with exit code $LASTEXITCODE." -ForegroundColor Red
            return $false
        }
    }

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "All specified folds ($Folds) completed successfully for $Model." -ForegroundColor Cyan
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
