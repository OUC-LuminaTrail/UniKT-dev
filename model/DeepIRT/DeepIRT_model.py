"""DeepIRT model implementation.

Paper: Deep-IRT: Make Deep Learning Based Knowledge Tracing Explainable
Using Item Response Theory (Yeung, 2019)
"""

import torch
from torch import nn
from torch.nn.init import kaiming_normal_


class DeepIRT(nn.Module):
    """DKVMN memory network with IRT-style prediction.

    The forward path follows PyKT's ``deep_irt.py``: read/write memory with
    DKVMN, infer student ability from ``f`` and item difficulty from ``k``,
    then predict with ``sigmoid(irt_scale * ability - difficulty)``.
    """

    def __init__(
        self,
        num_c: int,
        dim_s: int,
        size_m: int,
        dropout: float = 0.2,
        emb_type: str = "qid",
        irt_scale: float = 3.0,
    ):
        super().__init__()
        if emb_type != "qid":
            raise ValueError("DeepIRT currently supports only emb_type='qid'.")

        self.model_name = "deep_irt"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        self.irt_scale = irt_scale

        self.k_emb_layer = nn.Embedding(self.num_c, self.dim_s)
        self.v_emb_layer = nn.Embedding(self.num_c * 2, self.dim_s)

        self.Mk = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = nn.Parameter(torch.Tensor(self.size_m, self.dim_s))
        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.f_layer = nn.Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = nn.Dropout(dropout)
        # Kept to mirror PyKT DeepIRT; the IRT head below replaces this layer.
        self.p_layer = nn.Linear(self.dim_s, 1)

        self.diff_layer = nn.Sequential(nn.Linear(self.dim_s, 1), nn.Tanh())
        self.ability_layer = nn.Sequential(nn.Linear(self.dim_s, 1), nn.Tanh())

        self.e_layer = nn.Linear(self.dim_s, self.dim_s)
        self.a_layer = nn.Linear(self.dim_s, self.dim_s)

    def forward(
        self,
        sequence: torch.Tensor,
        response: torch.Tensor,
        mask: torch.Tensor | None = None,
        qtest: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run DeepIRT over a skill/response sequence.

        Args:
            sequence: Skill/KC IDs, shape ``[batch_size, seq_len]``.
            response: Binary correctness labels, shape ``[batch_size, seq_len]``.
            mask: Unused compatibility argument for this repository's trainers.
            qtest: When true, return ``(p, f, k)`` like PyKT for analysis.

        Returns:
            Probability tensor ``p`` with shape ``[batch_size, seq_len]``.
        """
        del mask
        batch_size = sequence.shape[0]

        x = sequence + self.num_c * response
        k = self.k_emb_layer(sequence)
        v = self.v_emb_layer(x)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)
        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2), strict=True
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (
                wt.unsqueeze(-1) * at.unsqueeze(1)
            )
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        read_content = (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)
        f = torch.tanh(self.f_layer(torch.cat([read_content, k], dim=-1)))

        stu_ability = self.ability_layer(self.dropout_layer(f))
        que_diff = self.diff_layer(self.dropout_layer(k))
        p = torch.sigmoid(self.irt_scale * stu_ability - que_diff).squeeze(-1)

        if qtest:
            return p, f, k
        return p
