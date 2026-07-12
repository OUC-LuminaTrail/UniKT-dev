# Train CLI

(trainpy)=

## train.py

```bash
python train.py -m <model> -d <dataset> [options]
```

| 参数 | 描述 |
| --- | --- |
| ``-m, --model`` | 模型名称（如 GIKT、AKT、DKT） |
| ``-d, --dataset`` | 数据集名称（如 assistments09） |
| ``--fold`` | K 折索引（默认：0） |
| ``--seed`` | 随机种子（默认：42） |
| ``--device`` | 设备（cuda/cpu，自动检测） |
| ``--es_patience`` | 早停耐心轮数（默认：10） |
| ``--max_grad_norm`` | 梯度裁剪阈值 |
| ``--skip_test`` | 跳过测试评估 |
