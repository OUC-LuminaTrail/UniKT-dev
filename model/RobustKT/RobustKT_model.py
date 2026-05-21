"""RobustKT model implementation.

This ports the official pyKT RobustKT implementation into the local training
framework. The smoothing module follows pyKT's causal convolution smoothing and
residual decomposition; despite the original variable name, it does not perform
an FFT.
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class CausalTemporalConv(nn.Module):
    """One-dimensional temporal convolution that never looks ahead."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        crop = self.conv.padding[0]
        if crop == 0:
            return out
        return out[:, :, :-crop]


class Smooth(nn.Module):
    """Causal smoothing plus residual fusion used by RobustKT."""

    def __init__(self, dropout: float, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.causal_conv = CausalTemporalConv(hidden_size, hidden_size, kernel_size)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        conv_input = input_tensor.permute(0, 2, 1)
        trend = self.causal_conv(conv_input).permute(0, 2, 1)
        random = input_tensor - trend
        smoothed = trend + (self.sqrt_beta**2) * random
        hidden_states = self.out_dropout(smoothed)
        return self.layer_norm(hidden_states + input_tensor)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: int,
    mask: torch.Tensor,
    dropout: nn.Dropout,
    zero_pad: bool,
    gamma: torch.Tensor | None = None,
    pdiff: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    batch_size, heads, seq_len = scores.size(0), scores.size(1), scores.size(2)
    device = q.device

    x1 = torch.arange(seq_len, device=device).expand(seq_len, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        masked_scores = scores.masked_fill(mask == 0, -1e32)
        masked_scores = F.softmax(masked_scores, dim=-1)
        masked_scores = masked_scores * mask.float().to(device)
        distcum_scores = torch.cumsum(masked_scores, dim=-1)
        disttotal_scores = torch.sum(masked_scores, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].float()
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect,
            min=0.0,
        )
        dist_scores = dist_scores.sqrt().detach()

    if gamma is None:
        gamma = torch.zeros(heads, 1, 1, device=device)
    gamma = -1.0 * nn.Softplus()(gamma).unsqueeze(0)
    if pdiff is None:
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma).exp(), min=1e-5),
            max=1e5,
        )
    else:
        diff = pdiff.unsqueeze(1).expand(
            pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2]
        )
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(
            torch.clamp((dist_scores * gamma * diff).exp(), min=1e-5),
            max=1e5,
        )
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(batch_size, heads, 1, seq_len, device=device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RobustKT decay distance effect."""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        n_heads: int,
        dropout: float,
        kq_same: bool,
        bias: bool = True,
        emb_type: str = "qid",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.0)
            constant_(self.v_linear.bias, 0.0)
            if not self.kq_same:
                constant_(self.q_linear.bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        zero_pad: bool,
        pdiff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)

        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        if not self.kq_same:
            q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        if "pdiff" not in self.emb_type:
            pdiff = None
        scores = attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            self.gammas,
            pdiff,
        )
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(concat)


class TransformerLayer(nn.Module):
    """Transformer layer used by RobustKT's two-stream architecture."""

    def __init__(
        self,
        d_model: int,
        d_feature: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        emb_type: str,
    ) -> None:
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(
            d_model,
            d_feature,
            n_heads,
            dropout,
            kq_same=kq_same == 1,
            emb_type=emb_type,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        mask: int,
        query: torch.Tensor,
        key: torch.Tensor,
        values: torch.Tensor,
        apply_pos: bool = True,
        pdiff: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seq_len, seq_len)), k=mask).astype(
            "uint8"
        )
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        query2 = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=mask == 0,
            pdiff=pdiff,
        )

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class Architecture(nn.Module):
    """RobustKT two-stream architecture with causal smoothing."""

    def __init__(
        self,
        n_blocks: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float,
        kq_same: int,
        emb_type: str,
        kernel_size: int,
    ) -> None:
        super().__init__()
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
        self.smooth = Smooth(dropout, d_model, kernel_size)

    def forward(
        self,
        q_embed_data: torch.Tensor,
        qa_embed_data: torch.Tensor,
        pid_embed_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = self.smooth(qa_embed_data)
        x = self.smooth(q_embed_data)

        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)

        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    apply_pos=False,
                    pdiff=pid_embed_data,
                )
                flag_first = False
            else:
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


class RobustKT(nn.Module):
    """RobustKT knowledge tracing model."""

    def __init__(self, args: Any, data_metadata: dict[str, Any]) -> None:
        super().__init__()
        self.model_name = "robustkt"
        self.num_skills = data_metadata["num_skills"]
        self.num_questions = data_metadata.get("num_questions", 0)
        self.n_pid = self.num_questions
        self.dropout = args.dropout
        self.kq_same = args.kq_same
        self.l2 = args.l2
        self.separate_qa = bool(args.separate_qa)
        self.emb_type = "qid"
        embed_l = args.d_model

        if self.num_questions > 0:
            self.difficult_param = nn.Embedding(self.num_questions + 1, 1)
            self.q_embed_diff = nn.Embedding(self.num_skills + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, embed_l)

        self.q_embed = nn.Embedding(self.num_skills, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.num_skills + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)

        self.model = Architecture(
            n_blocks=args.n_blocks,
            n_heads=args.num_attn_heads,
            dropout=args.dropout,
            d_model=args.d_model,
            d_ff=args.d_ff,
            kq_same=args.kq_same,
            emb_type=self.emb_type,
            kernel_size=args.kernel_size,
        )
        self.out = nn.Sequential(
            nn.Linear(args.d_model + embed_l, args.final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(args.final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        self.reset()

    def reset(self) -> None:
        if self.num_questions > 0:
            for parameter in self.parameters():
                if parameter.size(0) == self.num_questions + 1:
                    torch.nn.init.constant_(parameter, 0.0)

    def build_pid_data(
        self, question: torch.Tensor | None, valid_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """Build Rasch pid data with 0 reserved for padding."""
        if self.num_questions <= 0 or question is None:
            return None
        pid_data = question.clone() + 1
        return pid_data.masked_fill(~valid_mask.bool(), 0)

    def base_emb(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed_data = self.q_embed(sequence)
        if self.separate_qa:
            qa_data = sequence + self.num_skills * response
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_embed_data = self.qa_embed(response) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor | None = None,
        question: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_embed_data, qa_embed_data = self.base_emb(sequence, response)
        c_reg_loss = torch.tensor(0.0, device=sequence.device)
        pid_embed_data = None
        valid_mask = torch.ones_like(sequence, dtype=torch.bool) if mask is None else mask
        pid_data = self.build_pid_data(question, valid_mask)

        if self.num_questions > 0 and pid_data is not None:
            q_embed_diff_data = self.q_embed_diff(sequence)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

            qa_embed_diff_data = self.qa_embed_diff(response)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )
            c_reg_loss = (pid_embed_data**2).sum() * self.l2

        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        return torch.sigmoid(output), c_reg_loss
