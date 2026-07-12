# Model Zoo

UniKT 内置 35 个知识追踪模型，覆盖序列模型、图神经网络模型、注意力机制模型等多个方向。本页列出了所有已支持的模型及其基准测试结果。

## 评估协议

UniKT 采用标准化评估协议，确保所有模型在同一条件下公平对比：

| 项目 | 说明 |
| --- | --- |
| 交叉验证 | 5 折交叉验证 (5-fold cross-validation)，按用户分层划分 |
| 评估粒度 | 支持 KC 级和问题级两种评估粒度 |
| 主要指标 | AUC 和 ACC，报告 5 折均值 ± 标准差 |
| 数据划分 | 按 fold 参数指定当前折，训练/验证/测试集按 80%/10%/10% 自动划分 |
| 早停 | patience=10，监控 AUC，best 模型恢复 |


:::{note}
标注“UniKT 复现”的为本框架自行复现的模型代码，与原论文实现可能存在差异。
:::

## 排行榜

以下交互式表格展示所有模型在标准数据集上的基准结果。可按数据集标签筛选，按列排序，鼠标悬停 AUC/ACC 查看每折详细数据。

```{raw} html
<div class="arena-search">
  <svg class="arena-search-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
  <input id="arena-filter" type="text" placeholder="筛选模型...">
</div>
<div id="arena-tabs"></div>
<div id="arena-table"></div>
<div id="arena-count"></div>
```

```{raw} html
<script src="_static/arena.js"></script>
```

## 模型总览

| 模型 | 论文 | 代码 |
| --- | --- | --- |
| **DKT** | [Deep Knowledge Tracing (NeurIPS 2015)](https://papers.nips.cc/paper/5654-deep-knowledge-tracing) | [GitHub](https://github.com/chrispiech/DeepKnowledgeTracing) |
| **DKVMN** | [Dynamic Key-Value Memory Networks for Knowledge Tracing (WWW 2017)](https://dl.acm.org/doi/10.1145/3038912.3052580) | [GitHub](https://github.com/jennyzhang0215/DKVMN) |
| **SAKT** | [A Self-Attentive Model for Knowledge Tracing (EDM 2019)](https://arxiv.org/abs/1907.06837) | [GitHub](https://github.com/shalini1194/SAKT) |
| **AKT** | [Context-Aware Attentive Knowledge Tracing (KDD 2020)](https://dl.acm.org/doi/10.1145/3394486.3403282) | [GitHub](https://github.com/arghosh/AKT) |
| **GKT** | [Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network (WI 2019)](https://dl.acm.org/doi/10.1145/3350546.3352513) | [GitHub](https://github.com/jhljx/GKT) |
| **KQN** | [Knowledge Query Network for Knowledge Tracing: How Knowledge Interacts with Skills (LAK 2019)](https://dl.acm.org/doi/10.1145/3303772.3303786) | [GitHub](https://github.com/JSLBen/Knowledge-Query-Network-for-Knowledge-Tracing) |
| **GIKT** | [A Graph-based Interaction Model for Knowledge Tracing (ECML-PKDD 2020)](https://arxiv.org/abs/2009.05991) | [GitHub](https://github.com/ApexEDM/GIKT) |
| **DeepIRT** | [Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable Using Item Response Theory (EDM 2019)](https://arxiv.org/abs/1904.11738) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **SimpleKT** | [A Simple But Tough-to-Beat Baseline for Knowledge Tracing (ICLR 2023)](https://openreview.net/forum?id=9HiGqC9C-KA) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **DTransformer** | [Tracing Knowledge Instead of Patterns: Stable Knowledge Tracing with Diagnostic Transformer (WWW 2023)](https://dl.acm.org/doi/10.1145/3543507.3583255) | [GitHub](https://github.com/yxonic/DTransformer) |
| **HawkesKT** | [Temporal Cross-Effects in Knowledge Tracing (WSDM 2021)](https://dl.acm.org/doi/10.1145/3437963.3441802) | [GitHub](https://github.com/THUwangcy/HawkesKT) |
| **SGKT** | [Session Graph-based Knowledge Tracing (ESA 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0957417422009770) | [GitHub](https://github.com/CCNUZFW/SGKT) |
| **StableKT** | [Enhancing Length Generalization for Attention Based Knowledge Tracing Models with Linear Biases (IJCAI 2024)](https://doi.org/10.24963/ijcai.2024/654) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **MTKT** | [Learning Multi-granularity Temporal Characteristics for Attention Based Knowledge Tracing (Neurocomputing 2025)](https://doi.org/10.1016/j.neucom.2025.131338) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **SQGKT** | [Student-Question Interaction Graph-based Knowledge Tracing (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425027915) | [GitHub](https://github.com/Yingying933/SQGKT) |
| **LBKT** | [Learning Behavior-oriented Knowledge Tracing (KDD 2023)](https://dl.acm.org/doi/10.1145/3580305.3599407) | [GitHub](https://github.com/bigdata-ustc/EduKTM) |
| **DyGKT** | [Dynamic Graph Learning for Knowledge Tracing (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671773) | [GitHub](https://github.com/PengLinzhi/DyGKT) |
| **GRKT** | [Leveraging Pedagogical Theories to Understand Student Learning Process with Graph-based Reasonable Knowledge Tracing (KDD 2024)](https://dl.acm.org/doi/10.1145/3637528.3671853) | [GitHub](https://github.com/JJCui96/GRKT) |
| **extraKT** | [Extending Context Window of Attention Based Knowledge Tracing Models via Length Extrapolation (ECAI 2024)](https://doi.org/10.3233/FAIA240651) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **Mamba4KT** | [Mamba4KT: An Efficient and Effective Mamba-based Knowledge Tracing Model (arXiv 2024)](https://arxiv.org/abs/2405.16542) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **TCKT** | [Learning Consistent Representations with Temporal and Causal Enhancement for Knowledge Tracing (ESA 2024)](https://doi.org/10.1016/j.eswa.2023.123128) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **FAKT** | [A Frequency-Aware Mixture of Heterogeneous Experts Framework for Knowledge Tracing (WWW 2026)](https://dl.acm.org/doi/10.1145/3774904.3792272) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **MCKT** | [Multi-level Contrastive Learning for Knowledge Tracing (ACM TKDD 2025)](https://dl.acm.org/doi/10.1145/3759920) | — |
| **MCSKT** | [An Efficient Knowledge Tracing Model via Mamba Contextual Encoding and Dynamic Sparse Attention Mechanism (EAAI 2026)](https://doi.org/10.1016/j.engappai.2026.114312) | — |
| **MIKT** | [Interpretable Knowledge Tracing with Multiscale State Representation (WWW 2024)](https://dl.acm.org/doi/10.1145/3589334.3645373) | [GitHub](https://github.com/lilstrawberry/MIKT) |
| **QIKT** | [Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/view/26661) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **ReKT** | [Revisiting Knowledge Tracing: A Simple and Powerful Model (ACM MM 2024)](https://openreview.net/forum?id=GYomxff6HZ) | [GitHub](https://github.com/lilstrawberry/ReKT) |
| **RobustKT** | [Enhancing Knowledge Tracing through Decoupling Cognitive Pattern from Error-Prone Data (WWW 2025)](https://dl.acm.org/doi/10.1145/3696410.3714486) | [pyKT](https://github.com/pykt-team/pykt-toolkit) |
| **DAGKT** | [DAGKT: Difficulty and Attempts Boosted Graph-Based Knowledge Tracing (DASFAA 2023)](https://doi.org/10.1007/978-3-031-30108-7_22) | [GitHub](https://github.com/RuiLuo-7/DAGKT) |
| **HDHKT** | — | UniKT 原创 |
| **BDGKT** | [BDGKT: Bidirectional Dynamic Graph Knowledge Tracing (KBS 2026)](https://doi.org/10.1016/j.knosys.2026.115532) | [GitHub](https://github.com/Oia-10/BDGKT) |
| **ClusterKT** | [Cluster-driven Knowledge Tracing: Joint Learning-Forgetting Effects Modeling via State Dependency (ESA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425022389) | [GitHub](https://github.com/Lzhenghua/ClusterKT) |
| **ABKT** | [Ability Boosted Knowledge Tracing (Information Sciences 2022)](https://www.sciencedirect.com/science/article/pii/S0020025522001876) | [GitHub](https://github.com/ccnu-mathits/ABKT) |
| **IEKT** | [Tracing Knowledge State with Individual Cognition and Acquisition Estimation (SIGIR 2021)](https://dl.acm.org/doi/10.1145/3404835.3462827) | [GitHub](https://github.com/ApexEDM/iekt) |
| **UKT** | [Uncertainty-aware Knowledge Tracing (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/35007) | [GitHub](https://github.com/UncertaintyForKnowledgeTracing/UKT) |


## 复现

```bash
# 单折运行
python train.py -m DKT -d assistments09 --fold 0

# 五折交叉验证
./script_run_kfold.sh "0 1 2 3 4" DKT -d assistments09
```

## 贡献新模型

请参考[自定义模型](docs/advanced-guide/customize-model.md)。基本步骤：

1. 在 ``model/<YourModel>/`` 下创建 ``*_trainer.py``、``*_data.py``、``*_model.py``
2. 注册模型：``@register_trainer("YourModel")``、``@register_model_params("YourModel")``
3. 在至少 3 个数据集上完成 5 折训练并报告结果
4. 提交 PR 并附上指标汇总表
