# KT-GNN 实验模型（PyG 框架）

## 项目依赖

- Python：3.13
- torch：2.8.0
- torch_geometric：2.7.0
- pyg-lib：0.5.0
- scikit-learn：1.7.2
- pandas：2.2.3
- pyarrow：22.0.0
- tensorboard：2.20.0

```bash
# CPU only
conda install scikit-learn jupyterlab pandas pyarrow numpy tensorboard python=3.13 -c conda-forge -y
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cpu.html

# GPU
conda install scikit-learn jupyterlab pandas pyarrow numpy tensorboard python=3.13 -c conda-forge -y
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install torch_geometric pyg-lib -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```

## 项目结构
- data/：数据目录
  - assistments09/
  - assistments12/
  - assistments15/
  - assistments17/
  - EdNet/
- model/
  - GIKT/：GIKT 模型实现
    - GIKT_dataloader.py
    - GIKT_model.py
    - GIKT_train.py
- utility/
  - net_trainer.py：模型训练类
  - data_process/：数据处理模块，用于在不同模型之间建立统一的数据模型
    - assist09.py：ASSISTments2009 数据集预处理
    - assist12.py：ASSISTments2012 数据集预处理
    - assist17.py：ASSISTments2017 数据集预处理
    - ednet_kt1.py：EdNet KT1 数据预处理
    - data_utility.py：常用数据处理函数
