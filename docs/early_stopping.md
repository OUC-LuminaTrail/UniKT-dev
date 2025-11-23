# 早停（Early Stopping）

本仓库已在通用训练器 `utility/net_trainer.py` 中集成早停机制。默认关闭，可选按验证集指标触发中止训练，并可自动恢复最佳权重。

## 快速使用

```bash
# GIKT，基于 AUC 进行早停，耐心 10 轮训练，启用恢复最佳权重
python train_gikt.py -d assistments09 \
    --es_patience 10 --es_monitor auc --es_mode max --es_min_delta 0.0

# SQGKT，基于 Loss 进行早停，耐心 8 轮训练
python train_sqgkt.py -d assistments12 \
    --es_patience 8 --es_monitor loss --es_mode min --es_min_delta 1e-4
```

## 监控指标说明
- `auc`：默认，AUC 越大越好（`mode='max'`）。
- `acc`：准确率，越大越好（`mode='max'`）。
- `rmse`：均方根误差，越小越好（`mode='min'`）。
- `loss`：验证损失，越小越好（`mode='min'`）。

## 日志与检查点
- TensorBoard 会记录 `ES/BadEpochs`（连续未提升轮数）、`ES/Best`（最佳监控值）。
- 训练过程中会按监控指标保存最佳模型到 `runs/<time>/best_model.pth`。
