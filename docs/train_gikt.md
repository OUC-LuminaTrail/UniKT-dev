# GIKT 训练步骤

## 前置准备

- 完成数据预处理：见 `docs/data_preprocess.md`，或直接运行根目录脚本：

```bash
python data_process.py -d assistments09
python data_process.py -d assistments12
python data_process.py -d assistments17
python data_process.py -d ednet_kt1
```

## 基础训练命令

```bash
# 在 ASSISTments2009 数据集上训练
python train_gikt.py -d assistments09

# 在其他数据集上训练
python train_gikt.py -d assistments12
python train_gikt.py -d assistments17
python train_gikt.py -d ednet_kt1
```

## 启用早停（可选）

- 早停机制详见 `docs/early_stopping.md`。

```bash
# AUC 监控，耐心 10，启用恢复最佳权重
python train_gikt.py -d assistments09 \
  --es_patience 10 --es_monitor auc --es_mode max --es_min_delta 0.0
```

## 完整训练参数示例

```bash
python train_gikt.py \
  -d assistments09 \
  --hidden_dim 100 \
  --embedding_dim 100 \
  --lstm_layers 2 \
  --dropout 0.4 \
  --n_hop 3 \
  --history_neighbour 5 \
  --att_bound 0.2 \
  --epochs 150 \
  --batch_size 128 \
  --lr 0.001 \
  --weight_decay 1e-4 \
  --fold 0 \
  --seed 42 \
  \
  # Early Stopping（可选）
  --es_patience 10 \
  --es_monitor auc \
  --es_mode max \
  --es_min_delta 0.0
```

### 参数说明

- 模型参数：
  - `--hidden_dim`：隐藏层维度（默认：100）
  - `--embedding_dim`：嵌入层维度（默认：100）
  - `--lstm_layers`：LSTM 层数（默认：2）
  - `--dropout`：Dropout 概率（默认：0.4）
  - `--n_hop`：GNN 跳数（默认：3）
  - `--history_neighbour`：考虑的邻居数量（默认：5）
  - `--att_bound`：注意力边界值（默认：0.2）
- 数据参数：
  - `-d, --dataset`：数据集名称（必需）
  - `--data_base_path`：数据目录（默认：`./data`）
  - `--fold`：K 折索引（默认：0）
- 训练参数：
  - `--epochs`：训练轮数（默认：150）
  - `--batch_size`：批大小（默认：128）
  - `--lr`：学习率（默认：0.001）
  - `--lr_decay`：学习率衰减因子（可选）
  - `--weight_decay`：权重衰减（L2 正则）（默认：1e-4）
- 早停参数（可选）：
  - `--es_patience`：耐心轮数（未设置则不启用）
  - `--es_monitor`：监控指标（`auc|acc|rmse|loss`，默认：`auc`）
  - `--es_mode`：指标方向（`max|min`，`rmse/loss` 建议 `min`）
  - `--es_min_delta`：最小改变量阈值（默认：`0.0`）
- 其他：
  - `--seed`：随机种子（默认：42）
  - `--device`：设备（`cuda` 或 `cpu`，默认自动检测）

## K 折交叉验证

若需完整的 K 折评估，可依次训练所有折：

```bash
# 训练第 0 到第 4 折（假设使用 5 折）
for fold in {0..4}; do
  python train_gikt.py -d assistments09 --fold $fold
done
```

## 训练可视化

训练日志默认写入 `runs/`，可使用 TensorBoard 可视化；可参考项目 README 的“查看训练结果”部分。
