import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GRKT"]


def _positive_activate(mode, x):
    if mode == "sigmoid":
        return torch.sigmoid(x)
    if mode == "softplus":
        return F.softplus(x)
    if mode == "relu":
        return torch.relu(x)
    if mode == "softmax":
        return x.softmax(0)
    return x


def _sparse_mm(sparse_mat, dense):
    """CSR-sparse @ dense, handles both 2D [NK,KH] and 3D [B,NK,KH] inputs."""
    if dense.dim() == 2:
        return torch.sparse.mm(sparse_mat, dense)
    B, NK, KH = dense.shape
    flat = dense.permute(1, 0, 2).reshape(NK, B * KH)
    out = torch.sparse.mm(sparse_mat, flat)
    return out.reshape(NK, B, KH).permute(1, 0, 2)


class PositiveLinear(nn.Module):
    def __init__(self, d_in, d_out, mode):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out))
        self.mode = mode

    def forward(self, x):
        return x.matmul(_positive_activate(self.mode, self.weight))


class GRKT(nn.Module):
    """Graph-based Recursive Knowledge Tracing model."""

    def __init__(
        self,
        metadata,
        rel_map,
        pre_map,
        *,
        d_hidden: int,
        k_hidden: int,
        k_hop: int,
        tau: float,
        alpha: float,
        pos_mode: str,
        thresh: float,
    ):
        super().__init__()

        num_questions = metadata["num_questions"]
        num_skills = metadata["num_skills"]

        self.k_hop = k_hop
        self.tau = tau
        self.alpha = alpha
        self.DH = d_hidden
        self.KH = k_hidden
        self.num_skills = num_skills
        self.NK = num_skills + 1

        # Embeddings
        self.init_hidden = nn.Parameter(torch.randn(self.NK, k_hidden))
        self.know_master_proj = PositiveLinear(k_hidden, 1, pos_mode)
        self.know_embedding = nn.Embedding(
            num_skills + 1, d_hidden, padding_idx=num_skills
        )
        self.prob_embedding = nn.Embedding(num_questions, d_hidden)

        self.req_matrix = nn.Linear(d_hidden, d_hidden, bias=False)
        self.pos_mode = pos_mode

        # Aggregation matrices
        self.agg_rel_matrix = nn.ModuleList(
            [PositiveLinear(k_hidden, k_hidden, pos_mode) for _ in range(k_hop)]
        )
        self.agg_pre_matrix = nn.ModuleList(
            [PositiveLinear(k_hidden, k_hidden, pos_mode) for _ in range(k_hop)]
        )
        self.agg_sub_matrix = nn.ModuleList(
            [PositiveLinear(k_hidden, k_hidden, pos_mode) for _ in range(k_hop)]
        )

        # Problem difficulty
        self.prob_diff_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

        # Knowledge gain
        self.gain_ffn = nn.Sequential(
            nn.Linear(2 * d_hidden + k_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, k_hidden),
        )
        self.gain_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.gain_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.gain_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.gain_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.gain_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.gain_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )

        # Knowledge loss
        self.loss_ffn = nn.Sequential(
            nn.Linear(2 * d_hidden + k_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, k_hidden),
        )
        self.loss_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.loss_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.loss_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.loss_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.loss_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.loss_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )

        # Graph maps (thresholded boolean)
        NK = self.NK
        if thresh == 0:
            full_rel = torch.ones(num_skills, num_skills, dtype=torch.bool)
            full_pre = torch.ones(num_skills, num_skills, dtype=torch.bool)
        else:
            full_rel = (rel_map > thresh).astype(np.float32)
            full_pre = (pre_map > thresh).astype(np.float32)
        padded_rel = torch.zeros(NK, NK, dtype=torch.bool)
        padded_pre = torch.zeros(NK, NK, dtype=torch.bool)
        padded_rel[:num_skills, :num_skills] = torch.BoolTensor(full_rel)
        padded_pre[:num_skills, :num_skills] = torch.BoolTensor(full_pre)
        self.register_buffer("rel_map", padded_rel, persistent=False)
        self.register_buffer("pre_map", padded_pre, persistent=False)
        self.register_buffer(
            "sub_map", self.pre_map.transpose(-1, -2), persistent=False
        )

        # Inter-step learning
        self.decision_mlp = nn.Sequential(
            nn.Linear(4 * d_hidden + k_hidden, 2 * d_hidden),
            nn.ReLU(),
            nn.Linear(2 * d_hidden, 2),
        )
        self.learn_mlp = nn.Sequential(
            nn.Linear(4 * d_hidden + k_hidden, 2 * d_hidden),
            nn.ReLU(),
            nn.Linear(2 * d_hidden, k_hidden),
        )
        self.learn_matrix_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_matrix_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_matrix_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )

        # Learn kernel
        self.learn_kernel_matrix_rel = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.learn_kernel_matrix_pre = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.learn_kernel_matrix_sub = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.learn_kernel_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_kernel_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.learn_kernel_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )

        # Forget kernel
        self.forget_kernel_matrix_rel = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.forget_kernel_matrix_pre = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.forget_kernel_matrix_sub = nn.ModuleList(
            [nn.Linear(d_hidden, k_hidden, bias=False)]
            + [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop - 1)]
        )
        self.forget_kernel_output_rel = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.forget_kernel_output_pre = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )
        self.forget_kernel_output_sub = nn.ModuleList(
            [nn.Linear(k_hidden, k_hidden, bias=False) for _ in range(k_hop)]
        )

    def _graph_conv_merged_gl(
        self,
        total_gain,
        total_loss,
        alpha_1,
        beta_rel_sp,
        beta_pre_sp,
        beta_sub_sp,
    ):
        """Merged graph convolution for gain + loss (OPT 1 + sparse)."""
        B2 = total_gain.size(0)
        alpha_2 = alpha_1.repeat(2, 1, 1)
        for k in range(self.k_hop):
            stacked = torch.cat(
                [
                    self.gain_matrix_rel[k](total_gain),
                    self.loss_matrix_rel[k](total_loss),
                ],
                dim=0,
            )
            out = (_sparse_mm(beta_rel_sp, stacked) * alpha_2).relu()
            tg_rel = self.gain_output_rel[k](out[:B2])
            tl_rel = self.loss_output_rel[k](out[B2:])

            stacked = torch.cat(
                [
                    self.gain_matrix_pre[k](total_gain),
                    self.loss_matrix_pre[k](total_loss),
                ],
                dim=0,
            )
            out = (_sparse_mm(beta_pre_sp, stacked) * alpha_2).relu()
            tg_pre = self.gain_output_pre[k](out[:B2])
            tl_pre = self.loss_output_pre[k](out[B2:])

            stacked = torch.cat(
                [
                    self.gain_matrix_sub[k](total_gain),
                    self.loss_matrix_sub[k](total_loss),
                ],
                dim=0,
            )
            out = (_sparse_mm(beta_sub_sp, stacked) * alpha_2).relu()
            tg_sub = self.gain_output_sub[k](out[:B2])
            tl_sub = self.loss_output_sub[k](out[B2:])

            total_gain = total_gain + tg_rel + tg_pre + tg_sub
            total_loss = total_loss + tl_rel + tl_pre + tl_sub

        return total_gain.relu(), total_loss.relu()

    def forward(self, questions, knows, corrs, times):
        """
        Args:
            questions: [B, S] question IDs
            knows: [B, S, K] skill IDs per question (padded with 0)
            corrs: [B, S] correctness (0/1, will be cast to bool internally)
            times: [B, S] timestamps for temporal dynamics. When the dataset
                   provides timestamps, these are absolute seconds (millisecond
                   timestamps / 1000); otherwise they are 1-indexed sequence
                   positions (seq_pos + 1). Internally delta_time = new - old
                   handles both cases.

        Returns:
            scores: [B, S] predicted probabilities (sigmoid output)
        """
        B, S, K = knows.size()
        DH, KH, NK = self.DH, self.KH, self.NK

        h = self.init_hidden.repeat(B, 1, 1)
        h_initial = h

        total_know_embedding = self.know_embedding.weight
        prob_embedding = self.prob_embedding(questions)
        know_embedding = self.know_embedding(knows)

        alpha_matrix = (
            self.req_matrix(prob_embedding)
            .matmul(total_know_embedding.transpose(-1, -2))
            .sigmoid()
        )

        beta_matrix = (
            self.req_matrix(total_know_embedding)
            .matmul(total_know_embedding.transpose(-1, -2))
            .sigmoid()
        )

        rel_map = self.rel_map.float()
        pre_map = self.pre_map.float()
        sub_map = self.sub_map.float()

        beta_rel_tilde = beta_matrix * rel_map / rel_map.sum(-1, True).clamp(1)
        beta_pre_tilde = beta_matrix * pre_map / pre_map.sum(-1, True).clamp(1)
        beta_sub_tilde = beta_matrix * sub_map / sub_map.sum(-1, True).clamp(1)

        beta_rel_sp = beta_rel_tilde.to_sparse_csr()
        beta_pre_sp = beta_pre_tilde.to_sparse_csr()
        beta_sub_sp = beta_sub_tilde.to_sparse_csr()

        scores = []

        # Precompute learn and forget kernels (global, not per-step)
        lk_tilde = total_know_embedding
        for k in range(self.k_hop):
            lk_tilde_4_rel = self.learn_kernel_output_rel[k](
                _sparse_mm(
                    beta_rel_sp, self.learn_kernel_matrix_rel[k](lk_tilde)
                ).relu()
            )
            lk_tilde_4_pre = self.learn_kernel_output_pre[k](
                _sparse_mm(
                    beta_pre_sp, self.learn_kernel_matrix_pre[k](lk_tilde)
                ).relu()
            )
            lk_tilde_4_sub = self.learn_kernel_output_sub[k](
                _sparse_mm(
                    beta_sub_sp, self.learn_kernel_matrix_sub[k](lk_tilde)
                ).relu()
            )
            lk_tilde = lk_tilde_4_rel if k == 0 else lk_tilde + lk_tilde_4_rel
            lk_tilde = lk_tilde + lk_tilde_4_pre + lk_tilde_4_sub
        learn_kernel_para = F.softplus(lk_tilde) * self.alpha

        fk_tilde = total_know_embedding
        for k in range(self.k_hop):
            fk_tilde_4_rel = self.forget_kernel_output_rel[k](
                _sparse_mm(
                    beta_rel_sp, self.forget_kernel_matrix_rel[k](fk_tilde)
                ).relu()
            )
            fk_tilde_4_pre = self.forget_kernel_output_pre[k](
                _sparse_mm(
                    beta_pre_sp, self.forget_kernel_matrix_pre[k](fk_tilde)
                ).relu()
            )
            fk_tilde_4_sub = self.forget_kernel_output_sub[k](
                _sparse_mm(
                    beta_sub_sp, self.forget_kernel_matrix_sub[k](fk_tilde)
                ).relu()
            )
            fk_tilde = fk_tilde_4_rel if k == 0 else fk_tilde + fk_tilde_4_rel
            fk_tilde = fk_tilde + fk_tilde_4_pre + fk_tilde_4_sub
        forget_kernel_para = F.softplus(fk_tilde) * self.alpha

        learn_count = torch.zeros(B, NK, dtype=torch.long, device=h.device)

        corrs = corrs.bool()

        # OPT 2: pre-allocated scratch buffers — only safe in eval (no autograd)
        _use_buf = not torch.is_grad_enabled()
        if _use_buf:
            buf_gain = torch.empty(B, NK, KH, device=h.device)
            buf_loss = torch.empty(B, NK, KH, device=h.device)
            buf_learn = torch.empty(B, NK, KH, device=h.device)

        for i in range(S):
            h = h.clamp(min=-10, max=10)

            time = times[:, i]
            know = knows[:, i]
            corr = corrs[:, i]

            alpha = alpha_matrix[:, i]
            alpha_1 = alpha.unsqueeze(-1)

            know_safe = know.clamp(0, self.NK - 1)
            know_mask = know < self.num_skills
            know_cnt = know_mask.sum(-1, True).clamp(1)
            prob_emb = prob_embedding[:, i]
            know_emb = know_embedding[:, i].sum(-2)
            know_emb = know_emb / know_cnt
            know_prob_emb = torch.cat([know_emb, prob_emb], -1)

            # Knowledge aggregation
            h_tilde = h
            for k in range(self.k_hop):
                h_tilde_2_rel = self.agg_rel_matrix[k](h_tilde) * alpha_1
                h_tilde_2_pre = self.agg_pre_matrix[k](h_tilde) * alpha_1
                h_tilde_2_sub = self.agg_sub_matrix[k](h_tilde) * alpha_1

                h_tilde_3_rel = _sparse_mm(beta_rel_sp, h_tilde_2_rel)
                h_tilde_3_pre = _sparse_mm(beta_pre_sp, h_tilde_2_pre)
                h_tilde_3_sub = _sparse_mm(beta_sub_sp, h_tilde_2_sub)

                h_tilde = h_tilde + h_tilde_3_rel + h_tilde_3_pre + h_tilde_3_sub

            # Prediction
            master = self.know_master_proj(h_tilde).squeeze(-1)
            master = master.gather(-1, know_safe)
            master = master.masked_fill(~know_mask, 0)
            master = master.sum(-1) / know_cnt.squeeze(-1).clamp(1)
            diff = self.prob_diff_mlp(know_prob_emb).squeeze(-1)
            score = (master - diff).sigmoid()
            scores.append(score)

            # Knowledge gain & loss — shared preparation
            know_index = know_safe[:, :, None].expand(B, K, KH)
            target_h = h.gather(-2, know_index)
            know_prob_emb_1 = know_prob_emb.unsqueeze(-2).expand(B, K, 2 * DH)

            ffn_input = torch.cat([know_prob_emb_1, target_h], -1)
            gain = self.gain_ffn(ffn_input) * know_mask.unsqueeze(-1).float()
            loss_val = self.loss_ffn(ffn_input) * know_mask.unsqueeze(-1).float()

            if _use_buf:
                total_gain = buf_gain.zero_().scatter_(-2, know_index, gain)
                total_loss = buf_loss.zero_().scatter_(-2, know_index, loss_val)
            else:
                total_gain = torch.zeros_like(h).scatter(-2, know_index, gain)
                total_loss = torch.zeros_like(h).scatter(-2, know_index, loss_val)

            # OPT 1: merged graph convolution for gain + loss
            total_gain, total_loss = self._graph_conv_merged_gl(
                total_gain,
                total_loss,
                alpha_1,
                beta_rel_sp,
                beta_pre_sp,
                beta_sub_sp,
            )

            # Update knowledge state
            corr_1 = corr[:, None, None]
            h = h + torch.where(corr_1, total_gain, -total_loss)
            learn_count = learn_count + ((corr_1 * total_gain) > 0).any(-1).long()

            # Inter-step learning
            if i != S - 1:
                new_know = knows[:, i + 1]
                new_time = times[:, i + 1]

                new_know_safe = new_know.clamp(0, self.NK - 1)
                new_know_index = new_know_safe[:, :, None].expand(B, K, KH)
                new_target_h = h.gather(-2, new_know_index)
                total_target_h = torch.cat([target_h, new_target_h], -2)
                total_know_index = torch.cat([know_index, new_know_index], -2)

                new_prob_emb = prob_embedding[:, i + 1]
                new_know_mask = new_know < self.num_skills
                new_know_cnt = new_know_mask.sum(-1, True).clamp(1)
                new_know_emb = know_embedding[:, i + 1].sum(-2)
                new_know_emb = new_know_emb / new_know_cnt

                new_know_prob_emb = torch.cat([new_know_emb, new_prob_emb], -1)
                total_know_prob_emb = torch.cat([know_prob_emb, new_know_prob_emb], -1)
                total_know_prob_emb_1 = total_know_prob_emb.unsqueeze(-2).expand(
                    B, 2 * K, 4 * DH
                )

                learn_input = torch.cat([total_know_prob_emb_1, total_target_h], -1)
                decision = self.decision_mlp(learn_input)
                if self.training:
                    decision_gumbel_mask = F.gumbel_softmax(
                        decision, tau=self.tau, hard=True, dim=-1
                    )
                else:
                    # Deterministic hard argmax (no Gumbel noise) at inference.
                    decision_gumbel_mask = torch.zeros_like(decision).scatter_(
                        -1, decision.argmax(dim=-1, keepdim=True), 1.0
                    )
                decision_gumbel_mask_1 = decision_gumbel_mask[:, :, :1]

                learn = self.learn_mlp(learn_input)
                learn_3 = learn * decision_gumbel_mask_1
                total_know_mask = torch.cat([know_mask, new_know_mask], -1)
                learn_3 = learn_3 * total_know_mask.unsqueeze(-1).float()

                if _use_buf:
                    total_learn = buf_learn.zero_().scatter_(
                        -2, total_know_index, learn_3
                    )
                else:
                    total_learn = torch.zeros_like(h).scatter(
                        -2, total_know_index, learn_3
                    )

                for k in range(self.k_hop):
                    tl4_rel = self.learn_output_rel[k](
                        _sparse_mm(
                            beta_rel_sp, self.learn_matrix_rel[k](total_learn)
                        ).relu()
                    )
                    tl4_pre = self.learn_output_pre[k](
                        _sparse_mm(
                            beta_pre_sp, self.learn_matrix_pre[k](total_learn)
                        ).relu()
                    )
                    tl4_sub = self.learn_output_sub[k](
                        _sparse_mm(
                            beta_sub_sp, self.learn_matrix_sub[k](total_learn)
                        ).relu()
                    )
                    total_learn = total_learn + tl4_rel + tl4_pre + tl4_sub
                total_learn = total_learn.relu()

                history_gain = (h - h_initial).clamp(0)

                delta_time = (new_time - time).clamp(0).float()[:, None, None]

                # OPT 3: broadcasting — no expand needed
                learn_exp = (
                    -(learn_count[:, :, None].float() + 1)
                    * delta_time
                    * learn_kernel_para
                ).exp()
                forget_exp = (
                    -(learn_count[:, :, None].float() + 1)
                    * delta_time
                    * forget_kernel_para
                ).exp()

                h = h + (1 - learn_exp) * total_learn
                gain_after_forget = (
                    history_gain * forget_exp * (total_learn == 0).all(-1, True)
                )
                h = h - (history_gain - gain_after_forget) * (total_learn == 0).all(
                    -1, True
                )

                learn_count = learn_count + (total_learn > 0).any(-1).long()

        return torch.stack(scores, -1)
