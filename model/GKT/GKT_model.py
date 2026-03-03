"""GKT (Graph-based Knowledge Tracing) 模型实现

基于论文: "Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Networks"
参考实现: https://github.com/jhljx/GKT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """两层全连接 ReLU 网络带批归一化

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        dropout: Dropout 概率
        bias: 是否使用偏置
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs: torch.Tensor) -> torch.Tensor:
        """批归一化，处理 batch_size=1 或 3D 输入的情况"""
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 或 0 会导致 BatchNorm 错误，直接返回输入
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:
            return self.norm(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """擦除与添加门模块

    注意：这个擦除添加门与 DKVMN 中的略有不同。
    更多信息请参考论文 "Dynamic Key-Value Memory Networks for Knowledge Tracing"
    论文链接: https://arxiv.org/abs/1611.08108

    Args:
        feature_dim: 特征维度
        num_c: 概念数量
        bias: 是否使用偏置
    """

    def __init__(self, feature_dim: int, num_c: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征矩阵，形状为 [batch_size, num_c, feature_dim]

        Returns:
            处理后的特征矩阵，形状为 [batch_size, num_c, feature_dim]
        """
        erase_gate = torch.sigmoid(self.erase(x))  # [batch_size, num_c, feature_dim]
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))  # [batch_size, num_c, feature_dim]
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res


class GKT(nn.Module):
    """基于图的知识追踪模型

    使用图神经网络建模学生知识掌握程度

    Args:
        num_c: 概念（技能）数量
        hidden_dim: 隐藏层维度
        emb_size: 嵌入维度
        graph_type: 图类型，"dense" 或 "transition"
        graph: 预计算的图邻接矩阵，形状为 [num_c, num_c]
        dropout: Dropout 概率
        bias: 是否使用偏置
    """

    def __init__(
        self,
        num_c: int,
        hidden_dim: int,
        emb_size: int,
        graph_type: str = "dense",
        graph: torch.Tensor = None,
        dropout: float = 0.5,
        bias: bool = True,
    ):
        super().__init__()
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2  # 响应类型数量 (0/1)
        self.graph_type = graph_type

        # 图矩阵作为参数，但不参与梯度更新
        if graph is None:
            graph = self._build_default_graph(num_c, graph_type)
        self.graph = nn.Parameter(graph, requires_grad=False)

        # One-hot 特征矩阵
        self.register_buffer("one_hot_feat", torch.eye(self.res_len * self.num_c))

        # One-hot 问题矩阵（包含填充行）
        one_hot_q = torch.eye(self.num_c)
        zero_padding = torch.zeros(1, self.num_c)
        self.register_buffer("one_hot_q", torch.cat((one_hot_q, zero_padding), dim=0))

        # 概念和交互嵌入
        self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
        self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        # 自身特征变换函数
        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(
            mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias
        )

        # 邻居特征变换函数列表
        self.f_neighbor_list = nn.ModuleList()
        # f_in: 入边特征变换
        self.f_neighbor_list.append(
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        )
        # f_out: 出边特征变换
        self.f_neighbor_list.append(
            MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        )

        # 擦除添加门
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        # GRU 单元
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # 预测层
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    def _build_default_graph(self, num_c: int, graph_type: str) -> torch.Tensor:
        """构建默认图

        Args:
            num_c: 概念数量
            graph_type: 图类型

        Returns:
            图邻接矩阵
        """
        import numpy as np

        if graph_type == "dense":
            graph = 1.0 / (num_c - 1) * np.ones((num_c, num_c))
            np.fill_diagonal(graph, 0)
        else:
            # 默认使用单位矩阵作为后备
            graph = np.eye(num_c)
        return torch.from_numpy(graph).float()

    def _aggregate(
        self, xt: torch.Tensor, qt: torch.Tensor, ht: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """聚合步骤

        将输入特征与概念嵌入聚合

        Args:
            xt: 当前时间步的交互特征索引，形状为 [batch_size]
            qt: 当前时间步的问题索引，形状为 [batch_size]
            ht: 当前时间步所有概念的隐藏表示，形状为 [batch_size, num_c, hidden_dim]
            batch_size: 批次大小

        Returns:
            聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
        """
        device = xt.device
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1

        # 获取交互嵌入
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=device)
        x_embedding = self.interaction_emb(x_idx_mat)  # [res_len * num_c, emb_size]

        # 获取当前交互的特征嵌入
        masked_feat = F.embedding(
            xt[qt_mask], self.one_hot_feat
        )  # [mask_num, res_len * num_c]
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, emb_size]
        mask_num = res_embedding.shape[0]

        # 获取概念嵌入
        concept_idx_mat = (
            self.num_c * torch.ones((batch_size, self.num_c), device=device).long()
        )
        # 使用 expand 确保形状匹配，避免 CUDA 上的广播问题
        concept_idx_mat[qt_mask, :] = (
            torch.arange(self.num_c, device=device)
            .unsqueeze(0)
            .expand(qt_mask.sum().item(), -1)
        )
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, num_c, emb_size]

        # 将当前交互的嵌入放入对应位置
        if mask_num > 0:
            index_tuple = (torch.arange(mask_num, device=device), qt[qt_mask].long())
            concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(
                index_tuple, res_embedding
            )

        # 拼接隐藏状态和概念嵌入
        tmp_ht = torch.cat(
            (ht, concept_embedding), dim=-1
        )  # [batch_size, num_c, hidden_dim + emb_size]
        return tmp_ht

    def _agg_neighbors(self, tmp_ht: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """邻居聚合步骤

        使用图神经网络聚合邻居信息

        Args:
            tmp_ht: 聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            聚合邻居后的特征，形状为 [batch_size, num_c, hidden_dim]
        """
        device = qt.device
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1

        if not qt_mask.any():
            return tmp_ht[:, :, : self.hidden_dim]

        masked_qt = qt[qt_mask]  # [mask_num, ]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, num_c, hidden_dim + emb_size]
        mask_num = masked_tmp_ht.shape[0]

        # 获取自身特征
        self_index_tuple = (torch.arange(mask_num, device=device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + emb_size]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]

        # 扩展自身特征用于邻居聚合
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.num_c, 1)
        # [mask_num, num_c, hidden_dim + emb_size]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)
        # [mask_num, num_c, 2 * (hidden_dim + emb_size)]

        # 获取邻接矩阵
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, num_c, 1]
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        # [mask_num, num_c, 1]

        # 计算邻居特征
        neigh_features = adj * self.f_neighbor_list[0](
            neigh_ht
        ) + reverse_adj * self.f_neighbor_list[1](neigh_ht)
        # [mask_num, num_c, hidden_dim]

        # 更新特征
        m_next = tmp_ht[:, :, : self.hidden_dim].clone()
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)

        return m_next

    def _update(
        self, tmp_ht: torch.Tensor, ht: torch.Tensor, qt: torch.Tensor
    ) -> torch.Tensor:
        """更新步骤

        使用擦除添加门和 GRU 更新隐藏状态

        Args:
            tmp_ht: 聚合后的特征，形状为 [batch_size, num_c, hidden_dim + emb_size]
            ht: 当前隐藏状态，形状为 [batch_size, num_c, hidden_dim]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            更新后的隐藏状态，形状为 [batch_size, num_c, hidden_dim]
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.sum().item()

        # GNN 聚合
        m_next = self._agg_neighbors(tmp_ht, qt)  # [batch_size, num_c, hidden_dim]

        if mask_num > 0:
            # 擦除添加门
            m_next_qt = m_next[qt_mask].clone()
            m_next_qt = self.erase_add_gate(m_next_qt)  # [mask_num, num_c, hidden_dim]
            m_next[qt_mask] = m_next_qt

            # GRU 更新
            res = self.gru(
                m_next[qt_mask].reshape(-1, self.hidden_dim),
                ht[qt_mask].reshape(-1, self.hidden_dim),
            )
            # [mask_num * num_c, hidden_dim]
            m_next[qt_mask] = res.reshape(-1, self.num_c, self.hidden_dim)

        return m_next

    def _predict_step(self, h_next: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        """预测步骤

        预测所有概念的正确概率

        Args:
            h_next: 下一时间步的隐藏状态，形状为 [batch_size, num_c, hidden_dim]
            qt: 当前时间步的问题索引，形状为 [batch_size]

        Returns:
            预测概率，形状为 [batch_size, num_c]
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, num_c]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, num_c]
        return y

    def _get_next_pred(self, yt: torch.Tensor, q_next: torch.Tensor) -> torch.Tensor:
        """获取下一时间步的预测

        Args:
            yt: 所有概念的预测概率，形状为 [batch_size, num_c]
            q_next: 下一时间步的问题索引，形状为 [batch_size]

        Returns:
            下一问题的预测概率，形状为 [batch_size]
        """
        device = yt.device
        next_qt = torch.where(
            q_next != -1, q_next, self.num_c * torch.ones_like(q_next, device=device)
        )
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, num_c]
        # yt 和 one_hot_qt 的点积
        pred = (yt * one_hot_qt).sum(dim=1)  # [batch_size]
        return pred

    def forward(
        self, sequence: torch.Tensor, response: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """前向传播

        预测语义：y[:, t] 预测 response[:, t+1]
        即在时间步 t 处理完 (sequence[t], response[t]) 后，预测下一个位置 response[t+1]

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]
            mask: 有效位置掩码，形状为 [batch_size, sequence_length]

        Returns:
            预测结果，形状为 [batch_size, sequence_length - 1]
            y[:, t] 预测的是 response[:, t+1]
        """
        device = sequence.device
        batch_size, seq_len = sequence.shape

        # 初始化隐藏状态
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device)

        # 生成交互特征: q * 2 + r
        features = sequence * 2 + response
        questions = sequence

        pred_list = []
        for i in range(seq_len - 1):  # 只需要处理前 seq_len-1 个位置
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1

            # 聚合步骤
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)
            # [batch_size, num_c, hidden_dim + emb_size]

            # 更新步骤
            h_next = self._update(tmp_ht, ht, qt)  # [batch_size, num_c, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # 更新隐藏状态

            # 预测步骤
            yt = self._predict_step(h_next, qt)  # [batch_size, num_c]

            # 获取下一时间步的预测（对 response[:, i+1] 的预测）
            pred = self._get_next_pred(yt, questions[:, i + 1])
            pred_list.append(pred)

        # 堆叠预测结果
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]

        return pred_res
