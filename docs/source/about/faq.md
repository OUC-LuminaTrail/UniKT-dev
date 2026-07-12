# 常见问题

## 安装与环境

### 为什么推荐 pixi 而不是 conda？

pixi 提供精确的依赖版本锁定（``pixi.lock``），确保在不同设备上安装的包版本完全一致，这对模型数值可复现性至关重要。此外，pixi 预置了多套环境（PyTorch / Mamba / DHG），无需手动配置。

(支持-windows--macos-吗)=

### 支持 Windows / macOS 吗？

目前不支持。框架依赖的部分包（如某些 PyG 扩展）仅提供 Linux 预编译版本。

(ubuntu-2204-无法运行怎么办)=

### Ubuntu 22.04 无法运行怎么办？

框架依赖较新的 glibc，Ubuntu 22.04 的 glibc 版本不满足要求。请使用 Ubuntu 24.04 或更新的系统。

### 安装后 CUDA 不可用？

检查 PyTorch 是否正确识别 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available)"
```

如果返回 ``False``，请确认安装的是 CUDA 版本的 PyTorch（非 CPU 版），且 CUDA 驱动版本与 PyTorch 要求的 CUDA 版本兼容。

## 数据处理

### 下载数据集失败？

部分数据集源可能在网络受限环境下无法访问。可以尝试：

```bash
# 增加重试次数和线程
python data_process.py download -d assistments09 --max_retries 10 --num_threads 1

# 如果已知数据 URL 变更，可以覆盖下载地址
python data_process.py download -d assistments09 --data_url <新地址>
```

### 处理 EdNet-KT1 时内存不足？

EdNet-KT1 包含超过 1.3 亿条交互记录，处理时可能需要较大内存。建议：

- 使用 ``--sample_ratio`` 参数先采样
- 确保机器有至少 32GB 内存

## 训练

### 训练速度很慢？

- 确认使用的是 GPU 环境（``pixi shell``，非 ``pixi shell -e cpu``）
- 减小 batch_size 可能导致训练变慢，尝试增大到 GPU 显存允许的最大值
- 部分模型（如 DyGKT）支持 ``--max_grad_norm`` 参数优化训练稳定性

### 如何选择合适的模型？

不同模型在不同数据集和任务粒度（KC 级 / 问题级）上表现不同。请参考 [Model Zoo](../model-zoo.md) 排行榜查看各模型在标准 benchmark 上的性能。

### 训练中断后如何恢复？

训练默认会在每个 epoch 后保存检查点。如果中断，可以检查 ``runs/`` 目录下的最新运行。目前框架不直接支持断点续训，需要手动从检查点恢复。

## 模型开发

### 如何添加新模型？

请参阅 [添加新模型](../user-guide/new-model.md) 指南。核心步骤：创建模型目录 → 实现 trainer → 注册装饰器。无需修改框架核心代码。

### 为什么我的模型没有被自动发现？

确保 trainer 文件中使用了 ``@register_trainer("ModelName")`` 装饰器。框架通过静态 AST 扫描 ``model/`` 目录下的 ``.py`` 文件来发现注册。

(多环境兼容gpu--mamba--dhg如何处理)=

### 多环境兼容（GPU / Mamba / DHG）如何处理？

底层采用懒加载——只有当你实际调用 ``TRAINERS.get("ModelName")`` 时才会导入对应模块。这意味着你的模型文件可以导入仅在特定环境可用的包（如 ``mamba-ssm``），只要用户只在正确的环境下调用就不会报错。
