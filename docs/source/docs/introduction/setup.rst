环境配置
========

本文档介绍 UniKT 的环境配置与安装方式，包含多套不同依赖环境（PyTorch、Mamba-SSM、DHG 超图）的开箱即用方案。

环境要求
--------

======== ==========================================================
需求     说明
======== ==========================================================
操作系统 Linux（已在 Ubuntu 24.04 上测试通过，不支持 Ubuntu 22.04）
Python   3.10+（DHG 环境需 3.10）
CUDA     11.7+（GPU 环境）
包管理   pixi（推荐）或 conda
======== ==========================================================

安装方式
--------

方式一：pixi（推荐）
~~~~~~~~~~~~~~~~~~~~

pixi 提供包版本锁定，确保不同设备间依赖一致性，最大程度保证模型数值可复现。预置了多套环境配置，开箱即用：

.. code:: bash

   # 默认 GPU 环境（CUDA 12.8, Python 3.12, PyTorch 2.10）
   pixi shell

   # CPU 环境
   pixi shell -e cpu

   # Mamba 环境（含 mamba-ssm + causal-conv1d）
   pixi shell -e mamba

   # DHG 环境（超图卷积，CUDA 11.7, Python 3.10, PyTorch 1.13）
   pixi shell -e dhg-gpu

方式二：自动 Conda 配置
~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   该脚本未经充分测试，依赖版本可能落后于 pixi 配置文件。推荐使用 pixi。


.. code:: bash

   ./scripts/setup_env.sh # 自动检测 GPU
   ./scripts/setup_env.sh --cpu # 强制 CPU
   ./scripts/setup_env.sh -n myenv # 指定环境名
   ./scripts/setup_env.sh --yes # 非交互模式

方式三：手动 Conda 配置
~~~~~~~~~~~~~~~~~~~~~~~

.. _gpu-环境cuda-128:

GPU 环境（CUDA 12.8）
^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   conda create -n ktexp python=3.12
   conda activate ktexp

   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

   pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cu128.html

   conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
   pip install swanlab

CPU 环境
^^^^^^^^

.. code:: bash

   conda create -n ktexp python=3.12
   conda activate ktexp

   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

   pip install pyg_lib==0.6.0 torch-scatter==2.1.2 torch-geometric==2.7.0 \
    -f https://data.pyg.org/whl/torch-2.10.0+cpu.html

   conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
   pip install swanlab

DHG 环境（用于 HDHKT 等超图模型）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   conda create -n ktexp-dhg python=3.10
   conda activate ktexp-dhg

   pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117

   pip install "pyg_lib>=0.4.0,<0.5" torch-scatter==2.1.1 "torch-geometric>=2.7.0,<3" \
    -f https://data.pyg.org/whl/torch-1.13.1+cu117.html

   pip install "dhg==0.9.*"
   conda install -c conda-forge optuna scikit-learn pandas pyarrow python-dotenv ruff pytest polars seaborn matplotlib -y
   pip install swanlab

环境变量
--------

在项目根目录创建 ``.env`` 文件：

.. code:: bash

   cp .env.example .env

+-----------------------+-----------------------------------------------------+
| 变量                  | 说明                                                |
+=======================+=====================================================+
| ``LARK_WEBHOOK_URL``  | 飞书机器人通知 webhook（可选）                      |
+-----------------------+-----------------------------------------------------+
| ``SWANLAB_WORKSPACE`` | SwanLab 工作空间名称                                |
+-----------------------+-----------------------------------------------------+
| ``SWANLAB_MODE``      | ``cloud``\ （上传到云端）或 ``local``\ （本地存储） |
+-----------------------+-----------------------------------------------------+
| ``LOG_LEVEL``         | 日志级别（默认 INFO）                               |
+-----------------------+-----------------------------------------------------+

验证安装
--------

.. code:: bash

   # 切换到环境
   pixi shell

   # 验证核心依赖
   python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available}')"
   python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')"
   python -c "import optuna; print(f'Optuna {optuna.__version__}')"

   # 运行快速测试
   python -c "from model import TRAINERS; print(f'已发现 {len(TRAINERS)} 个模型')"
