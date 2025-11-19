# SQGKT 训练步骤

## 前置准备

- 完成数据预处理：见 `docs/data_preprocess.md`。

## 基础训练命令

```bash
python train_sqgkt.py -d assistments12
```

> 提示：更多可用数据集或参数可通过 `-h` 查看：
>
> ```bash
> python train_sqgkt.py -h
> ```

## 启用早停

- 早停机制详见 `docs/early_stopping.md`。

```bash
# 以 Loss 为监控指标，耐心 8，启用恢复最佳权重
python train_sqgkt.py -d assistments12 \
  --es_patience 8 --es_monitor loss --es_mode min --es_min_delta 1e-4 --es_restore_best
```

## 备注

- 建议指定 `--seed` 以便复现实验结果。
- 设备可通过 `--device` 指定为 `cuda` 或 `cpu`（默认自动检测）。
