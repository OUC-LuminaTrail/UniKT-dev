"""AKT (Attentive Knowledge Tracing) 模型实现

原始论文: Ghosh et al., "Context-Aware Attentive Knowledge Tracing", KDD 2020
"""

import math
from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
    注意力计算函数

    Args:
        q: Query张量 [BS, n_heads, seq_len, d_k]
        k: Key张量 [BS, n_heads, seq_len, d_k]
        v: Value张量 [BS, n_heads, seq_len, d_k]
        d_k: 每个头的维度
        mask: 掩码矩阵
        dropout: Dropout层
        zero_pad: 是否对第一行进行零填充
        gamma: 位置衰减参数
        pdiff: 题目难度差异（暂不使用）

    Returns:
        output: 注意力输出 [BS, seq_len, d_model]
    """
    device = q.device
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = (
            torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(device)
        )
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.0
        )
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1.0 * m(gamma).unsqueeze(0)

    if pdiff is None:
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5
        )
    else:
        diff = pdiff.unsqueeze(1).expand(
            pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2]
        )
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5), max=1e5
        )
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        emb_type: str = "qid",
    ):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type

        if emb_type.startswith("qid"):
            self.d_k = d_feature
            self.h = n_heads
            self.kq_same = kq_same

            self.v_linear = nn.Linear(d_model, d_model, bias=bias)
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
            if kq_same is False:
                self.q_linear = nn.Linear(d_model, d_model, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.proj_bias = bias
            self.out_proj = nn.Linear(d_model, d_model, bias=bias)
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            xavier_uniform_(self.gammas)
            self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):
        bs = q.size(0)

        if self.emb_type.startswith("qid"):
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            if self.kq_same is False:
                q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            else:
                q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            gammas = self.gammas
            if self.emb_type.find("pdiff") == -1:
                pdiff = None
            scores = attention(
                q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas, pdiff
            )

            concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)
        return output


class TransformerLayer(nn.Module):
    """Transformer层，包含多头注意力和前馈网络"""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        emb_type: str,
    ):
        super().__init__()
        kq_same = kq_same == 1

        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        device = query.device
        seqlen, _ = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff
            )
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff
            )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)

        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """AKT架构，包含QA编码器和知识检索器"""

    def __init__(
        self,
        n_question: int,
        n_blocks: int,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        model_type: str,
        emb_type: str,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {"akt"}:
            self.blocks_1 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        emb_type=emb_type,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.blocks_2 = nn.ModuleList(
                [
                    TransformerLayer(
                        d_model=d_model,
                        d_feature=d_model // n_heads,
                        d_ff=d_ff,
                        dropout=dropout,
                        n_heads=n_heads,
                        kq_same=kq_same,
                        emb_type=emb_type,
                    )
                    for _ in range(n_blocks * 2)
                ]
            )

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data=None):
        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder: 对0~t-1时刻前的qa信息进行编码
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)

        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # Peek current question
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                    pdiff=pid_embed_data,
                )
                flag_first = False
            else:  # Don't peek current response
                x = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    apply_pos=True,
                    pdiff=pid_embed_data,
                )
                flag_first = True
        return x


class AKT(nn.Module):
    """AKT (Attentive Knowledge Tracing) 模型

    基于上下文感知的知识追踪模型，使用双流Transformer架构：
    - blocks_1: QA编码器，编码历史交互信息
    - blocks_2: 知识检索器，从历史中检索相关知识状态

    Args:
        num_c: 概念（技能）数量
        d_model: 模型隐藏维度
        n_blocks: Transformer块数量
        dropout: Dropout概率
        d_ff: 前馈网络维度
        kq_same: Key和Query是否使用相同的线性变换
        final_fc_dim: 最终全连接层维度
        num_attn_heads: 注意力头数量
        separate_qa: 是否使用独立的QA嵌入
        l2: L2正则化系数
        emb_type: 嵌入类型
    """

    def __init__(
        self,
        num_c: int,
        d_model: int = 256,
        n_blocks: int = 2,
        dropout: float = 0.2,
        d_ff: int = 256,
        kq_same: int = 1,
        final_fc_dim: int = 512,
        num_attn_heads: int = 8,
        separate_qa: bool = False,
        l2: float = 1e-5,
        emb_type: str = "qid",
    ):
        super().__init__()
        self.model_name = "akt"
        self.num_c = num_c
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model

        # 概念嵌入层
        self.q_embed = nn.Embedding(self.num_c, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        # Architecture: 双流Transformer
        self.model = Architecture(
            n_question=self.num_c,
            n_blocks=n_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=d_model,
            d_feature=d_model / num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            model_type=self.model_type,
            emb_type=self.emb_type,
        )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

    def base_emb(self, q_data, target):
        """计算基础嵌入"""
        q_embed_data = self.q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """前向传播

        Args:
            sequence: 概念ID序列，形状为 [batch_size, sequence_length]
            response: 响应序列，形状为 [batch_size, sequence_length]
            mask: 有效位置掩码，形状为 [batch_size, sequence_length]

        Returns:
            预测结果，形状为 [batch_size, sequence_length]
            在时刻 t 的输出预测的是 t+1 的标签
        """
        # 获取嵌入
        q_embed_data, qa_embed_data = self.base_emb(sequence, response)

        # 通过双流Transformer架构
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data=None)

        # 拼接输出和问题嵌入
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        # Sigmoid激活
        preds = torch.sigmoid(output)

        # AKT输出语义说明：
        # AKT的第二个流（blocks_2）第一层可以"peek current question"
        # 因此 preds[:, t] 使用 sequence[0:t+1] 和 response[0:t+1] 预测 response[t]

        return preds
