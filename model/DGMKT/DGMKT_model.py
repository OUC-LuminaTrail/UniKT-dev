"""DGMKT: Leveraging Student Profiles and the Mamba Framework to Enhance Knowledge Tracing

- H 分支：学生-概念超图（HGNN）学生画像
- D 分支：习题有向转移图（GCN）+ 位置注意力池化画像
- 双 Mamba1 骨干 + sigmoid 门控融合 + 三预测头（h / d / ensemble）
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from torch_geometric.nn import GCNConv


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    """GPT-2 style init: zero Linear bias, N(0, 0.02) embeddings, and scale
    residual projections by 1/sqrt(n_residuals * n_layer)."""
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    """Mamba1 骨干：n_layer 个 Block（无 MLP）+ 末层 RMSNorm。"""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
    ) -> None:
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        def _create_block(layer_idx: int) -> Block:
            mixer_cls = partial(Mamba, layer_idx=layer_idx)
            norm_cls = partial(RMSNorm, eps=1e-5)
            return Block(
                d_model,
                mixer_cls,
                nn.Identity,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
            )

        self.layers = nn.ModuleList([_create_block(i) for i in range(n_layer)])
        self.norm_f = RMSNorm(d_model, eps=1e-5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states


class HGNNConv(nn.Module):
    """单层超图卷积：先线性变换，再沿超图邻接矩阵 G 传播。"""

    def __init__(self, in_ft: int, out_ft: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        self.bias = nn.Parameter(torch.Tensor(out_ft))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        x = x.matmul(self.weight)
        x = x + self.bias
        return torch.sparse.mm(G, x)


class HGNN(nn.Module):
    def __init__(self, in_ch: int, n_hid: int):
        super().__init__()
        self.hgc1 = HGNNConv(in_ch, n_hid)

    def forward(self, x: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return F.relu(self.hgc1(x, G))


class DGMKT(nn.Module):
    """DGMKT 主模型。

    Args:
        num_c_raw: 未做 16 倍数填充的技能数。
        num_users: 超图节点数（原始学生数）。
        max_seq_len: 序列长度（位置注意力参数 pos 按此构造）。
        G: HGNN 归一化超图邻接矩阵（稀疏 COO，随 build_components 构建）。
        d_model: 隐藏维度。
        n_layer: 每路 Mamba 骨干的层数。
        pad_num_c_multiple: num_c 向上取整的倍数。
    """

    def __init__(
        self,
        num_c_raw: int,
        num_users: int,
        max_seq_len: int,
        G: torch.Tensor,
        d_model: int = 512,
        n_layer: int = 4,
        pad_num_c_multiple: int = 16,
    ) -> None:
        super().__init__()
        num_c = num_c_raw
        if num_c % pad_num_c_multiple != 0:
            num_c += pad_num_c_multiple - (num_c % pad_num_c_multiple)
        self.num_c_raw = num_c_raw  # pad skill id (falls on a real class row)
        self.num_c = num_c
        self.num_users = num_users
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.skill_embedding = nn.Embedding(num_c + 1, d_model)
        self.answer_embedding = nn.Embedding(3, d_model)  # values {0, 1, 2=pad}
        self.change_dim_D = nn.Linear(d_model * 3, d_model)
        self.change_dim_H = nn.Linear(d_model * 3, d_model)

        # D branch: skill-transition GCN over the shared skill embedding
        self.gcn_conv1 = GCNConv(d_model, 8)
        self.gcn_conv2 = GCNConv(8, d_model)
        # H branch: frozen random student features propagated through the hypergraph
        self.net = HGNN(d_model, d_model)
        # Persistent so checkpoints reproduce the HGNN inputs on reload (RNG differs at rebuild)
        self.register_buffer("stu", torch.empty(num_users, d_model).normal_())
        # Deterministically rebuilt from data; excluded from state_dict
        self.register_buffer("G", G, persistent=False)
        # One-hot table: class indicator rows plus one all-zero row
        self.register_buffer(
            "one_hot_table",
            torch.cat([torch.eye(num_c), torch.zeros(1, num_c)], dim=0),
        )
        self.pos = nn.Parameter(torch.rand(max_seq_len, max_seq_len, 1))

        self.backbone_H = MixerModel(d_model=d_model, n_layer=n_layer)
        self.backbone_D = MixerModel(d_model=d_model, n_layer=n_layer)

        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)
        self.fc_d = nn.Linear(d_model, num_c)
        self.fc_h = nn.Linear(d_model, num_c)
        self.fc_ensemble = nn.Linear(2 * d_model, num_c)

        self.apply(partial(_init_weights, n_layer=n_layer))

    def _get_next_pred(self, res: torch.Tensor, skill: torch.Tensor) -> torch.Tensor:
        """Gather the logit of the next skill at each position."""
        next_skill = skill[:, 1:]
        one_hot_skill = F.embedding(next_skill, self.one_hot_table)
        return (res * one_hot_skill).sum(dim=-1)

    def forward(
        self, student: torch.Tensor, skill: torch.Tensor, answer: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            student: 学生（原始 id）[B, L]，0-based，pad 位复用属主 id。
            skill: 技能 id [B, L]，pad 位置为未填充的 num_skills。
            answer: 作答 [B, L]，取值 {0, 1, 2=pad}。

        Returns:
            (logit_h, logit_d, ensemble_logit)，各 [B, L-1] raw logits，
            out[t] 为对 skill[t+1] 的预测。
        """
        batch_size, seq_len = skill.shape

        # H branch: student profile lookup
        stu_embedding = self.net(self.stu, self.G)  # [num_users, d]
        stu_h = stu_embedding[student]

        mask = torch.ne(answer, 2).unsqueeze(-1).float()

        # D branch: batched per-sample GCN over skill transition graphs.
        # Disconnected components (offset by b*N) keep propagation per-sample;
        # weight.repeat lets gradients flow back into the shared skill embedding.
        n_nodes = self.num_c + 1
        offsets = torch.arange(batch_size, device=skill.device) * n_nodes
        src = (skill[:, :-1] + offsets.unsqueeze(1)).reshape(-1)
        dst = (skill[:, 1:] + offsets.unsqueeze(1)).reshape(-1)
        edge_index = torch.stack([src, dst])
        x = self.skill_embedding.weight.repeat(batch_size, 1)
        h = F.relu(self.gcn_conv1(x, edge_index))
        h = self.gcn_conv2(h, edge_index)
        all_stu_h = (
            h.view(batch_size, n_nodes, -1).gather(
                1, skill.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            * mask
        )

        # Pad positions join the softmax denominator with logit 0 (numerators
        # are already zeroed by the mask above)
        effective_lengths = mask.sum(dim=1).squeeze(-1).long()
        expand_pos = F.softmax(self.pos[effective_lengths - 1] * mask, dim=1)
        all_stu_h = (
            torch.sum(all_stu_h * expand_pos, dim=1)
            .unsqueeze(1)
            .expand(-1, seq_len, -1)
        )

        # Interaction embedding: concatenation order depends on correctness
        skill_emb = self.skill_embedding(skill)
        answer_emb = self.answer_embedding(answer)
        skill_answer = torch.cat((skill_emb, answer_emb), dim=2)
        answer_skill = torch.cat((answer_emb, skill_emb), dim=2)
        is_correct = answer.unsqueeze(2).expand_as(skill_answer) == 1
        x = torch.where(is_correct, skill_answer, answer_skill)

        x_DG = self.change_dim_D(torch.cat((all_stu_h, x), dim=-1))
        x_HG = self.change_dim_H(torch.cat((stu_h, x), dim=-1))

        h_DG = self.backbone_D(x_DG)
        h_HG = self.backbone_H(x_HG)
        logit_h = self.fc_h(h_HG)
        logit_d = self.fc_d(h_DG)
        theta = torch.sigmoid(self.w1(h_HG) + self.w2(h_DG))
        ensemble_logit = self.fc_ensemble(
            torch.cat([theta * h_HG, (1 - theta) * h_DG], dim=-1)
        )

        return (
            self._get_next_pred(logit_h[:, :-1], skill),
            self._get_next_pred(logit_d[:, :-1], skill),
            self._get_next_pred(ensemble_logit[:, :-1], skill),
        )
