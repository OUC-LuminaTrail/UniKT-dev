"""KeenKT 模型实现"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


def nig_distance_matmul(mean1, cov1, mean2, cov2):
    """Pairwise distance between NIG distributions (mean + sqrt-variance)."""
    mean1_sq = torch.sum(mean1**2, dim=-1, keepdim=True)
    mean2_sq = torch.sum(mean2**2, dim=-1, keepdim=True)
    mean_diff = (
        mean1_sq
        + mean2_sq.transpose(-2, -1)
        - 2 * torch.matmul(mean1, mean2.transpose(-2, -1))
    )
    cov1_sq = torch.sum(cov1**2, dim=-1, keepdim=True)
    cov2_sq = torch.sum(cov2**2, dim=-1, keepdim=True)
    cov_diff = (
        cov1_sq
        + cov2_sq.transpose(-2, -1)
        - 2
        * torch.matmul(
            torch.sqrt(torch.clamp(cov1, min=1e-24)),
            torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-2, -1),
        )
    )
    return mean_diff + cov_diff


def d2s_1overx(distance):
    return 1 / (1 + distance)


def uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout):
    """Attention scored by (negative) NIG distribution distance."""
    scores = -nig_distance_matmul(q_mean, q_cov, k_mean, k_cov) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores)
    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores, v_cov)
    return output_mean, output_cov


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gating: [B, T, H] -> [B, T, H]."""

    def __init__(self, hidden_dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // reduction, hidden_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = x.mean(dim=1)
        s = self.fc2(self.relu(self.fc1(s))).unsqueeze(1)
        s = self.sigmoid(s)
        return x * s


class CosinePositionalEmbedding(nn.Module):
    """Sin/cos positional embedding, selected by input sequence length."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, : x.size(1), :]


class DiffusionModule(nn.Module):
    """Residual MLP denoiser for the diffusion auxiliary loss."""

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return x + self.net(x)


class NIGNCELoss(nn.Module):
    """InfoNCE loss based on pairwise NIG distribution distance."""

    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.activation = nn.ELU()

    def forward(self, mean1, cov1, mean2, cov2):
        cov1 = self.activation(cov1) + 1
        cov2 = self.activation(cov2) + 1
        sim11 = d2s_1overx(nig_distance_matmul(mean1, cov1, mean1, cov1)) / (
            self.temperature
        )
        sim22 = d2s_1overx(nig_distance_matmul(mean2, cov2, mean2, cov2)) / (
            self.temperature
        )
        sim12 = -d2s_1overx(nig_distance_matmul(mean1, cov1, mean2, cov2)) / (
            self.temperature
        )
        d = sim12.shape[-1]
        # exclude self-matches from the negative pool
        sim11[..., range(d), range(d)] = float("-inf")
        sim22[..., range(d), range(d)] = float("-inf")
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        return self.criterion(logits, labels)


class MultiHeadAttention(nn.Module):
    """Dual-stream (mean/cov) multi-head attention; query and key share
    the same projection."""

    def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)
        xavier_uniform_(self.out_mean_proj.weight)
        xavier_uniform_(self.out_cov_proj.weight)
        if self.proj_bias:
            constant_(self.v_mean_linear.bias, 0.0)
            constant_(self.v_cov_linear.bias, 0.0)
            constant_(self.k_mean_linear.bias, 0.0)
            constant_(self.k_cov_linear.bias, 0.0)
            constant_(self.out_mean_proj.bias, 0.0)
            constant_(self.out_cov_proj.bias, 0.0)

    def forward(
        self, query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, mask
    ):
        batch_size = query_mean.size(0)

        value_mean = (
            self.v_mean_linear(values_mean)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        value_cov = (
            self.v_cov_linear(values_cov)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        # query and key share the same projection
        key_mean = (
            self.k_mean_linear(key_mean)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        key_cov = (
            self.k_cov_linear(key_cov)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        query_mean = (
            self.k_mean_linear(query_mean)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )
        query_cov = (
            self.k_cov_linear(query_cov)
            .view(batch_size, -1, self.h, self.d_k)
            .transpose(1, 2)
        )

        scores_mean, scores_cov = uattention(
            query_mean,
            query_cov,
            key_mean,
            key_cov,
            value_mean,
            value_cov,
            self.d_k,
            mask,
            self.dropout,
        )

        concat_mean = (
            scores_mean.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        concat_cov = (
            scores_cov.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output_mean = self.out_mean_proj(concat_mean)
        output_cov = self.out_cov_proj(concat_cov)
        return output_mean, output_cov


class TransformerLayer(nn.Module):
    """Dual-stream transformer block.

    The cov stream is passed through ELU()+1 after each residual add and
    LayerNorm to stay positive (it encodes distribution variance).
    """

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, seq_len):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()
        # strictly lower triangular: position t attends only to j < t
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool)
        self.register_buffer(
            "src_mask",
            ones.tril(diagonal=-1).view(1, 1, seq_len, seq_len),
            persistent=False,
        )

    def forward(self, query_mean, query_cov, values_mean, values_cov):
        query2_mean, query2_cov = self.masked_attn_head(
            query_mean,
            query_cov,
            query_mean,
            query_cov,
            values_mean,
            values_cov,
            mask=self.src_mask,
        )
        query_mean = query_mean + self.dropout1(query2_mean)
        query_cov = query_cov + self.dropout1(query2_cov)
        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)

        query2_mean = self.mean_linear2(
            self.dropout(self.activation(self.mean_linear1(query_mean)))
        )
        query2_cov = self.cov_linear2(
            self.dropout(self.activation(self.cov_linear1(query_cov)))
        )
        query_mean = query_mean + self.dropout2(query2_mean)
        query_cov = query_cov + self.dropout2(query2_cov)
        query_mean = self.layer_norm2(query_mean)
        query_cov = self.layer_norm2(self.activation2(query_cov) + 1)
        return query_mean, query_cov


class Architecture(nn.Module):
    """Stack of dual-stream transformer blocks over NIG embeddings."""

    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, seq_len):
        super().__init__()
        self.d_model = d_model
        self.position_mean_embeddings = CosinePositionalEmbedding(
            d_model, max_len=seq_len
        )
        self.position_cov_embeddings = CosinePositionalEmbedding(
            d_model, max_len=seq_len
        )
        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    seq_len=seq_len,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self, q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data
    ):
        q_mean_embed_data = q_mean_embed_data + self.position_mean_embeddings(
            q_mean_embed_data
        )
        q_cov_embed_data = q_cov_embed_data + self.position_cov_embeddings(
            q_cov_embed_data
        )
        qa_mean_embed_data = qa_mean_embed_data + self.position_mean_embeddings(
            qa_mean_embed_data
        )
        qa_cov_embed_data = qa_cov_embed_data + self.position_cov_embeddings(
            qa_cov_embed_data
        )

        # keep covariance parameters positive
        q_cov_embed_data = F.elu(q_cov_embed_data) + 1
        qa_cov_embed_data = F.elu(qa_cov_embed_data) + 1

        y_mean = qa_mean_embed_data
        y_cov = qa_cov_embed_data
        x_mean = q_mean_embed_data
        x_cov = q_cov_embed_data

        for block in self.blocks_2:
            x_mean, x_cov = block(x_mean, x_cov, y_mean, y_cov)
        return x_mean, x_cov


class KeenKT(nn.Module):
    """KeenKT: NIG uncertainty embeddings + dual-stream distance attention.

    Prediction semantics (same-position): preds[:, t] estimates the response
    at position t given the concept at t and the interaction history before t.
    """

    def __init__(
        self,
        num_skills,
        n_pid,
        d_model=256,
        d_ff=512,
        n_blocks=4,
        n_heads=8,
        dropout=0.2,
        final_fc_dim=256,
        final_fc_dim2=256,
        se_ratio=16,
        seq_len=200,
        emb_type="stoc_qid",
        use_cl=True,
        use_diffusion=True,
        noise_level=0.3,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.n_pid = n_pid
        self.emb_type = emb_type
        self.use_cl = use_cl
        self.use_diffusion = use_diffusion
        self.noise_level = noise_level

        if n_pid > 0:
            self.q_embed_diff = nn.Embedding(num_skills + 1, d_model)
            self.difficult_param = nn.Embedding(n_pid + 1, d_model)

        # NIG parameter embeddings (mu, alpha, beta, delta) per concept
        self.mu_q_embed = nn.Embedding(num_skills, d_model)
        self.alpha_q_embed = nn.Embedding(num_skills, d_model)
        self.beta_q_embed = nn.Embedding(num_skills, d_model)
        self.delta_q_embed = nn.Embedding(num_skills, d_model)
        self.mu_qa_embed = nn.Embedding(2, d_model)
        self.alpha_qa_embed = nn.Embedding(2, d_model)
        self.beta_qa_embed = nn.Embedding(2, d_model)
        self.delta_qa_embed = nn.Embedding(2, d_model)

        self.model = Architecture(
            n_blocks=n_blocks,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            seq_len=seq_len,
        )
        self.se_gate = SEBlock(d_model, reduction=se_ratio)
        self.out = nn.Sequential(
            nn.Linear(d_model * 4, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_fc_dim2, 1),
        )
        if use_diffusion:
            self.diffusion_module = DiffusionModule(d_model)
        if use_cl:
            self.wloss = NIGNCELoss(1)
        self.reset()

    def reset(self):
        # difficulty embeddings start at zero contribution
        if self.n_pid > 0:
            torch.nn.init.constant_(self.difficult_param.weight, 0.0)

    def base_emb(self, q_data, target):
        """Look up NIG parameter embeddings and convert them to actual
        mean / sqrt-variance representations."""
        q_mean_embed = self.mu_q_embed(q_data)
        q_alpha_embed = self.alpha_q_embed(q_data)
        q_beta_embed = self.beta_q_embed(q_data)
        q_delta_embed = self.delta_q_embed(q_data)
        qa_mean_embed = self.mu_qa_embed(target) + q_mean_embed
        qa_alpha_embed = self.alpha_qa_embed(target) + q_alpha_embed
        qa_beta_embed = self.beta_qa_embed(target) + q_beta_embed
        qa_delta_embed = self.delta_qa_embed(target) + q_delta_embed

        q_alpha_pos = F.softplus(q_alpha_embed) + 1e-8
        q_beta_con = torch.tanh(q_beta_embed) * q_alpha_pos * 0.999
        q_delta_pos = F.elu(q_delta_embed) + 1
        q_gamma = torch.sqrt(torch.clamp(q_alpha_pos**2 - q_beta_con**2, min=1e-8))
        q_mean_actual = q_mean_embed + (
            q_delta_pos * q_beta_con / torch.clamp(q_gamma, min=1e-8)
        )
        q_sqrt_var = (
            torch.sqrt(q_delta_pos)
            * q_alpha_pos
            / torch.clamp(q_gamma, min=1e-8) ** 1.5
        )

        qa_alpha_pos = F.softplus(qa_alpha_embed) + 1e-8
        qa_beta_con = torch.tanh(qa_beta_embed) * qa_alpha_pos * 0.999
        qa_delta_pos = F.elu(qa_delta_embed) + 1
        qa_gamma = torch.sqrt(torch.clamp(qa_alpha_pos**2 - qa_beta_con**2, min=1e-8))
        qa_mean_actual = qa_mean_embed + (
            qa_delta_pos * qa_beta_con / torch.clamp(qa_gamma, min=1e-8)
        )
        qa_sqrt_var = (
            torch.sqrt(qa_delta_pos)
            * qa_alpha_pos
            / torch.clamp(qa_gamma, min=1e-8) ** 1.5
        )
        return q_mean_actual, q_sqrt_var, qa_mean_actual, qa_sqrt_var

    def _encode_raw(self, q_data, target, pid_data):
        """Shared pipeline: NIG embeddings -> Rasch modulation -> attention."""
        q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed = self.base_emb(
            q_data, target
        )
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_mean_embed = q_mean_embed + pid_embed_data * q_embed_diff_data
            q_cov_embed = q_cov_embed + pid_embed_data * q_embed_diff_data
        mean_out, cov_out = self.model(
            q_mean_embed, q_cov_embed, qa_mean_embed, qa_cov_embed
        )
        return mean_out, cov_out, q_mean_embed, q_cov_embed

    def _concat_head(self, mean_out, cov_out, q_mean_emb, q_cov_emb):
        if self.emb_type == "stoc_qid":
            return torch.cat([mean_out, cov_out, q_mean_emb, q_cov_emb], dim=-1)
        # qid ablation: single (mean) stream, cov embeddings only
        return torch.cat([mean_out, mean_out, q_cov_emb, q_cov_emb], dim=-1)

    def forward(self, q_data, target, pid_data, mask=None, target_aug=None):
        """Unified forward.

        Returns dict with preds [B, S] (sigmoid probabilities) and scalar
        cl_loss / diffusion_loss, which are zeros outside training mode.
        """
        mean_raw, cov_raw, q_mean_emb, q_cov_emb = self._encode_raw(
            q_data, target, pid_data
        )
        mean_out = self.se_gate(mean_raw)
        cov_out = self.se_gate(cov_raw)
        zero = mean_out.new_zeros(())

        diffusion_loss = zero
        if self.training and self.use_diffusion:
            latent_noisy = mean_out + torch.randn_like(mean_out) * self.noise_level
            diffusion_loss = F.mse_loss(self.diffusion_module(latent_noisy), mean_out)

        cl_loss = zero
        if self.training and self.use_cl:
            aug_target = target if target_aug is None else target_aug
            # view 2 stays ungated: upstream's gated copy is overwritten by
            # an ungated re-run before pooling
            mean2_raw, cov2_raw, _, _ = self._encode_raw(q_data, aug_target, pid_data)
            pair_mask = mask[:, :-1] & mask[:, 1:]
            cl_mask = torch.cat(
                [
                    torch.ones(mask.size(0), 1, dtype=mask.dtype, device=mask.device),
                    pair_mask,
                ],
                dim=1,
            ).unsqueeze(-1)
            pm, pc = mean_out * cl_mask, cov_out * cl_mask
            pm2, pc2 = mean2_raw * cl_mask, cov2_raw * cl_mask
            if self.emb_type == "stoc_qid":
                cl_loss = self.wloss(pm.mean(1), pc.mean(1), pm2.mean(1), pc2.mean(1))
            else:
                cl_loss = self.wloss(pm.mean(1), pm.mean(1), pm2.mean(1), pm2.mean(1))

        concat_q = self._concat_head(mean_out, cov_out, q_mean_emb, q_cov_emb)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))
        return {"preds": preds, "cl_loss": cl_loss, "diffusion_loss": diffusion_loss}
