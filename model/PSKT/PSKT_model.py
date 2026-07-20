"""PSKT model implementation."""

import torch
import torch.nn as nn
import torch.nn.init as init


class PSKT(nn.Module):
    """PSKT model.

    Args:
        num_questions: Number of questions (ids are 1-based, 0 = padding).
        num_skills: Number of skills (ids are 1-based, 0 = padding).
        embed_dim: Embedding dimension.
        max_concepts: Maximum number of skills per question.
        max_time_interval: Maximum time interval in minutes.
    """

    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        embed_dim: int = 256,
        max_concepts: int = 4,
        max_time_interval: int = 43200,
    ):
        super().__init__()
        self.num_questions = num_questions
        self.num_skills = num_skills
        self.embed_dim = embed_dim
        self.max_concepts = max_concepts
        self.max_time_interval = max_time_interval
        self.kc_dim = num_skills + 1

        self.q_embedding = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        self.c_embedding = nn.Embedding(self.kc_dim, embed_dim, padding_idx=0)
        self.r_embedding = nn.Embedding(3, embed_dim, padding_idx=2)
        self.TD_embedding = nn.Embedding(max_time_interval + 1, embed_dim)

        self.trans_QDiff = nn.Linear(embed_dim, 1)
        self.trans_QAlpha = nn.Linear(embed_dim, 1)

        self.knowledge_init = nn.Parameter(
            init.xavier_uniform_(torch.rand(1, self.kc_dim, dtype=torch.float32))
        )

        self.trans_skq = nn.Linear(embed_dim + self.kc_dim, self.kc_dim)
        self.trans_ekq = nn.Linear(embed_dim + self.kc_dim, self.kc_dim)

        self.ks_gate1 = nn.Linear(2 * self.kc_dim, self.kc_dim)
        self.ks_gate2 = nn.Linear(3 * self.kc_dim, self.kc_dim)

        self.ka_gate2 = nn.Linear(2 * self.kc_dim, self.kc_dim)

        self.ki_gate1 = nn.Linear(3 * self.kc_dim, self.kc_dim)
        self.ki_gate2 = nn.Linear(2 * self.kc_dim + embed_dim, self.kc_dim)

        self.fo_gate = nn.Linear(embed_dim * 2 + self.kc_dim, self.kc_dim)

        self.sigmoid = nn.Sigmoid()

        self.trans_knowledge_all = nn.Sequential(
            nn.Linear(embed_dim + self.kc_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, self.kc_dim),
        )

        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([1]))

        self.trans_G = nn.Linear(embed_dim + self.kc_dim, 1)
        self.trans_S = nn.Linear(embed_dim + self.kc_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                init.xavier_uniform_(p)
        for module in self.modules():
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _avg_skill_embedding(self, skills):
        idx = skills.clamp(min=0).long()
        embed_all = self.c_embedding(idx)
        mask = (skills != -1).float().unsqueeze(-1)
        count = mask.sum(dim=2).clamp(min=1.0)
        return (embed_all * mask).sum(dim=2) / count

    def _aggregate_predictions(self, pred_per_skill, skills):
        idx = skills.clamp(min=0).long()
        gathered = pred_per_skill.gather(-1, idx)
        mask = (skills != -1).float()
        count = mask.sum(dim=-1).clamp(min=1.0)
        return (gathered * mask).sum(dim=-1) / count

    def forward(self, question, skills, response, timestamp, mask=None):
        """Return same-position probabilities [B, S] where position t predicts response[t]."""
        Q_embed = self.q_embedding(question)
        K_n = self._avg_skill_embedding(skills)
        R_embed = self.r_embedding(response)

        bs, length = question.size(0), question.size(1)

        Q_Diff = self.sigmoid(self.trans_QDiff(Q_embed))
        Q_alpha = self.sigmoid(self.trans_QAlpha(Q_embed))

        QC_encoder = K_n + Q_embed * (self.a * Q_Diff + self.b * Q_alpha)

        TD = (timestamp[:, 1:] - timestamp[:, :-1]) // 60
        TD = torch.clamp(TD, min=0, max=self.max_time_interval)
        TD_embed = self.TD_embedding(TD.long())

        h_kc_pre = self.knowledge_init.repeat(bs, 1)
        loop_len = int(mask.sum(dim=1).max()) if mask is not None else length
        correct_pred_all = question.new_zeros(
            bs, length, self.kc_dim, dtype=torch.float32
        )

        for i in range(loop_len):
            qc = QC_encoder[:, i]
            qa = R_embed[:, i]
            diff = Q_Diff[:, i]
            alpha = Q_alpha[:, i]

            skq = self.trans_skq(torch.cat([qc, h_kc_pre], dim=-1))
            ekq = self.trans_ekq(torch.cat([qc, h_kc_pre], dim=-1))
            resk = ekq - skq

            ks_title = torch.tanh(self.ks_gate1(torch.cat([skq, h_kc_pre], dim=-1)))
            ks = self.sigmoid(self.ks_gate2(torch.cat([ekq, skq, resk], dim=-1)))
            ks = ks * ks_title

            ka_title = torch.tanh(resk)
            ka = self.sigmoid(self.ka_gate2(torch.cat([ekq, ks], dim=-1)))
            ka = ka * ka_title

            next_info = torch.cat([ka, qc], dim=-1)
            knowledge_for_next_concept = self.sigmoid(
                self.trans_knowledge_all(next_info)
            )

            exp1 = torch.exp(-1.702 * 4 * alpha * (knowledge_for_next_concept - diff))
            correct_pred = 1 / (1 + exp1)

            G = self.sigmoid(self.trans_G(next_info)) * 0.5
            S = self.sigmoid(self.trans_S(next_info)) * 0.5
            correct_pred = G * (1 - correct_pred) + (1 - S) * correct_pred

            correct_pred_all[:, i] = correct_pred

            if i != length - 1:
                ki_title = (
                    torch.tanh(self.ki_gate1(torch.cat([resk, skq, ekq], dim=-1))) + 1
                ) / 2
                ki = self.sigmoid(self.ki_gate2(torch.cat([ks, ka, qa], dim=-1)))
                ki = ki * ki_title
                h_kc_pre = h_kc_pre + ki

                qt = QC_encoder[:, i + 1]
                ts = TD_embed[:, i]
                foin = self.sigmoid(self.fo_gate(torch.cat([qt, ts, h_kc_pre], dim=-1)))
                h_kc_pre = h_kc_pre * foin

        correct_pred = self._aggregate_predictions(correct_pred_all, skills)
        return correct_pred
