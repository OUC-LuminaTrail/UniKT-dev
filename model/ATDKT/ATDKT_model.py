"""ATDKT (Auxiliary Task enhanced DKT) 模型实现

原始论文: Liu et al., "Enhancing Deep Knowledge Tracing with Auxiliary Tasks", WWW 2023
"""

import torch
from torch import nn
from torch.nn import functional as F


class ATDKT(nn.Module):
    """ATDKT 模型

    Args:
        num_q: 题目数量
        num_c: 概念（技能）数量
        emb_size: 嵌入维度（同时为 LSTM 隐藏维度）
        seq_len: 最大序列长度（因果掩码尺寸）
        dropout: Dropout 概率
        use_qt: 是否启用 QT 辅助任务
        use_ik: 是否启用 IK 辅助任务
        qt_encoder: QT 关系网络类型，"transformer" 或 "lstm"
        qt_num_layers: QT 关系网络层数
        num_attn_heads: QT Transformer 注意力头数
        qt_with_interaction: QT 编码器输入是否并入交互嵌入
            （False 对应原始实现的 delxemb 变体：输入仅题目+概念嵌入）
        ik_start: IK 损失的起始位置（前若干步历史正确率估计不稳）
    """

    def __init__(
        self,
        num_q: int,
        num_c: int,
        emb_size: int,
        seq_len: int,
        dropout: float = 0.1,
        use_qt: bool = True,
        use_ik: bool = True,
        qt_encoder: str = "transformer",
        qt_num_layers: int = 1,
        num_attn_heads: int = 4,
        qt_with_interaction: bool = False,
        ik_start: int = 50,
    ):
        super().__init__()
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.use_qt = use_qt
        self.use_ik = use_ik
        self.qt_with_interaction = qt_with_interaction
        self.ik_start = ik_start

        # interaction embedding: c + num_c * r
        self.interaction_emb = nn.Embedding(num_c * 2, emb_size)
        self.lstm_layer = nn.LSTM(emb_size, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        # mastery distribution head: hidden → hidden//2 → num_c
        self.out_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_c),
        )

        if use_qt:
            self.question_emb = nn.Embedding(num_q, emb_size)
            self.concept_emb = nn.Embedding(num_c, emb_size)
            if qt_encoder == "transformer":
                # causal mask, position t attends to positions <= t
                att_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1
                )
                self.register_buffer("att_mask", att_mask, persistent=False)
                # NOTE: pykt's encoder layer omits the dropout argument (pinned
                # at torch's 0.1 default); we tie it to the model dropout for a
                # coherent search space — identical at the default dropout=0.1.
                encoder_layer = nn.TransformerEncoderLayer(
                    emb_size, nhead=num_attn_heads, dropout=dropout, batch_first=True
                )
                self.qt_net = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=qt_num_layers,
                    norm=nn.LayerNorm(emb_size),
                )
            else:
                self.qt_net = nn.LSTM(emb_size, emb_size, batch_first=True)
            self.qclasifier = nn.Sequential(
                nn.Linear(emb_size, emb_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(emb_size // 2, num_c),
            )

        if use_ik:
            self.hisclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, 1),
            )

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor,
        question: torch.Tensor,
        history_corr: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """前向传播

        Args:
            sequence: 概念ID序列 [B, S]
            response: 响应序列 [B, S]
            mask: 有效位掩码 [B, S]
            question: 题目ID序列 [B, S]
            history_corr: 截止当前的全局历史正确率 [B, S]，
                IK 任务目标，仅训练态使用

        Returns:
            dict: preds 为 same-position 概率 [B, S]（位置 t 预测
            response[t]，位置 0 为占位 0）；qt_loss / ik_loss 仅在
            训练态且对应任务启用时非 None
        """
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        sup_mask = None
        if self.training and (self.use_qt or self.use_ik):
            sup_mask = torch.cat([mask[:, :-1], torch.zeros_like(mask[:, :1])], dim=1)

        xemb = self.interaction_emb(sequence + self.num_c * response)

        qt_loss = None
        if self.use_qt:
            qemb = self.question_emb(question)
            cemb = self.concept_emb(sequence)
            catemb = qemb + cemb
            if self.qt_with_interaction:
                catemb = catemb + xemb
            if isinstance(self.qt_net, nn.TransformerEncoder):
                seq = catemb.shape[1]
                qh = self.qt_net(catemb, mask=self.att_mask[:seq, :seq])
            else:
                qh, _ = self.qt_net(catemb)
            if self.training and sup_mask.any():
                qt_loss = F.cross_entropy(
                    self.qclasifier(qh[sup_mask]), sequence[sup_mask]
                )
            xemb = xemb + qh + cemb
            if self.qt_with_interaction:
                xemb = xemb + qemb

        h, _ = self.lstm_layer(xemb)

        ik_loss = None
        if self.use_ik and self.training and history_corr is not None:
            # IK range: positions in [ik_start, S-1)
            flag = mask[:, self.ik_start : -1]
            if flag.any():
                rpreds = torch.sigmoid(self.hisclasifier(h)).squeeze(-1)
                ik_loss = F.mse_loss(
                    rpreds[:, self.ik_start : -1][flag],
                    history_corr[:, self.ik_start : -1][flag],
                )

        h = self.dropout_layer(h)
        y = torch.sigmoid(self.out_layer(h))  # [B, S, num_c]

        target = sequence[:, 1:].long().unsqueeze(-1)
        y_next = y[:, :-1].gather(-1, target).squeeze(-1)
        y = torch.cat([torch.zeros_like(y_next[:, :1]), y_next], dim=1)

        return {"preds": y, "qt_loss": qt_loss, "ik_loss": ik_loss}
