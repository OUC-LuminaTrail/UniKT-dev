"""Hypergraph structure and HGNN convolution on plain torch sparse tensors.

``Hypergraph`` stores incidence in PyG's ``hyperedge_index`` layout
(row 0: vertex ids, row 1: hyperedge ids — the format consumed by
``torch_geometric.nn.HypergraphConv``). ``HGNNConv`` implements the HGNN
convolution (AAAI 2019) on top of it.
"""

import torch
import torch.nn as nn


class Hypergraph:
    r"""加权超图。

    关联结构以 PyG 惯例的 ``hyperedge_index`` ([2, num_incidences]，第 0 行
    为顶点 id、第 1 行为超边 id) 与 ``hyperedge_weight`` ([num_e]) 表示，
    可直接作为 ``torch_geometric.nn.HypergraphConv`` 的输入。

    超边规范化：每条超边顶点升序排序；顶点集合相同的超边按迭代均值
    ``(old + new) / 2`` 合并权重，并保留首次出现顺序。

    Args:
        num_v: 顶点数
        e_list: 超边列表，每条超边为其顶点 id 列表
        e_weight: 超边权重列表；``None`` 时全为 1，标量时广播到所有超边
        device: 张量所在设备

    Example:
        >>> hg = Hypergraph(num_v=4, e_list=[[0, 1, 2], [2, 3]], e_weight=[0.8, 0.5])
        >>> hg.num_v, hg.num_e
        (4, 2)
        >>> smoothed = hg.smoothing_with_HGNN(torch.randn(4, 8))
    """

    def __init__(
        self,
        num_v: int,
        e_list: list[list[int]] | None = None,
        e_weight: list[float] | float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_v = int(num_v)
        self._device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        if e_list is None:
            e_list = []
        if e_weight is None:
            weights: list[float] = [1.0] * len(e_list)
        elif isinstance(e_weight, (int, float)):
            weights = [float(e_weight)] * len(e_list)
        else:
            weights = [float(w) for w in e_weight]
        if len(e_list) != len(weights):
            raise ValueError(
                f"Number of hyperedges ({len(e_list)}) and weights ({len(weights)}) must match"
            )

        # Canonicalize edges: sorted vertex tuples keyed in insertion order, so
        # duplicate vertex sets merge (running mean) and first-seen order
        # becomes the hyperedge id order.
        merged: dict[tuple[int, ...], float] = {}
        for vertices, w in zip(e_list, weights):
            key = tuple(sorted(int(v) for v in vertices))
            merged[key] = (merged[key] + w) / 2.0 if key in merged else w
        self._edge_groups: list[tuple[tuple[int, ...], float]] = list(merged.items())

        v_ids: list[int] = []
        e_ids: list[int] = []
        for e_idx, (vertices, _) in enumerate(self._edge_groups):
            v_ids.extend(vertices)
            e_ids.extend([e_idx] * len(vertices))
        self.hyperedge_index = torch.tensor(
            [v_ids, e_ids], dtype=torch.long, device=self._device
        )
        self.hyperedge_weight = torch.tensor(
            [w for _, w in self._edge_groups],
            dtype=torch.float,
            device=self._device,
        )

        self._cache: dict[str, torch.Tensor] = {}

    def __repr__(self) -> str:
        return f"Hypergraph(num_v={self.num_v}, num_e={self.num_e})"

    @property
    def num_e(self) -> int:
        """Number of (canonicalized) hyperedges."""
        return len(self._edge_groups)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device | str) -> "Hypergraph":
        """Move the incidence tensors and cached matrices to ``device``."""
        self._device = torch.device(device)
        self.hyperedge_index = self.hyperedge_index.to(self._device)
        self.hyperedge_weight = self.hyperedge_weight.to(self._device)
        for key, val in self._cache.items():
            if isinstance(val, torch.Tensor):
                self._cache[key] = val.to(self._device)
        return self

    @property
    def num_incidences(self) -> int:
        """Number of (vertex, hyperedge) incidence pairs."""
        return int(self.hyperedge_index.size(1))

    @property
    def H(self) -> torch.Tensor:
        """Sparse incidence matrix ``[num_v, num_e]`` (unweighted, coalesced)."""
        if "H" not in self._cache:
            self._cache["H"] = torch.sparse_coo_tensor(
                self.hyperedge_index,
                torch.ones(self.num_incidences, device=self._device),
                torch.Size([self.num_v, self.num_e]),
                device=self._device,
            ).coalesce()
        return self._cache["H"]

    @property
    def H_T(self) -> torch.Tensor:
        """Transposed sparse incidence matrix ``[num_e, num_v]``."""
        if "H_T" not in self._cache:
            self._cache["H_T"] = self.H.t()
        return self._cache["H_T"]

    @property
    def D_v(self) -> torch.Tensor:
        """Weighted vertex degrees ``[num_v]``: ``D_v[v] = sum of w_e over edges containing v``."""
        if "D_v" not in self._cache:
            H = self.H
            val = self.hyperedge_weight[H.indices()[1]] * H.values()
            H_weighted = torch.sparse_coo_tensor(
                H.indices(), val, size=H.shape, device=self._device
            ).coalesce()
            self._cache["D_v"] = torch.sparse.sum(H_weighted, dim=1).to_dense().view(-1)
        return self._cache["D_v"]

    @property
    def D_e(self) -> torch.Tensor:
        """Hyperedge sizes ``[num_e]`` (number of vertices per hyperedge)."""
        if "D_e" not in self._cache:
            self._cache["D_e"] = torch.sparse.sum(self.H_T, dim=1).to_dense().view(-1)
        return self._cache["D_e"]

    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""HGNN 归一化拉普拉斯 ``D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}``（稀疏）。

        零度顶点/空边的逆幂次按 0 处理。
        """
        if "L_HGNN" not in self._cache:
            dv_neg_1_2 = self._diag(self._inv_pow(self.D_v, -0.5))
            w_e = self._diag(self.hyperedge_weight)
            de_neg_1 = self._diag(self._inv_pow(self.D_e, -1.0))
            self._cache["L_HGNN"] = (
                dv_neg_1_2.mm(self.H)
                .mm(w_e)
                .mm(de_neg_1)
                .mm(self.H_T)
                .mm(dv_neg_1_2)
                .coalesce()
            )
        return self._cache["L_HGNN"]

    def smoothing_with_HGNN(self, X: torch.Tensor) -> torch.Tensor:
        r"""以 HGNN 拉普拉斯平滑特征：``X' = L_HGNN X``。

        Args:
            X: 特征矩阵 ``[num_v, C]``；设备不一致时自动迁移到超图设备
        """
        if self._device != X.device:
            X = X.to(self._device)
        return self.L_HGNN.mm(X)

    @staticmethod
    def _diag(values: torch.Tensor) -> torch.Tensor:
        """Wrap a dense vector into a sparse diagonal matrix."""
        n = values.numel()
        idx = torch.arange(n, device=values.device).view(1, -1).repeat(2, 1)
        return torch.sparse_coo_tensor(
            idx, values, torch.Size([n, n]), device=values.device
        ).coalesce()

    @staticmethod
    def _inv_pow(degrees: torch.Tensor, exponent: float) -> torch.Tensor:
        """Elementwise power with zero degrees mapped back to zero."""
        val = degrees**exponent
        val[torch.isinf(val)] = 0
        return val


class HGNNConv(nn.Module):
    r"""HGNN 卷积层（`Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_, AAAI 2019）。

    矩阵形式：

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H}
        \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        \mathbf{X} \mathbf{\Theta} \right)

    Args:
        in_channels: 输入通道数 :math:`C_{in}`
        out_channels: 输出通道数 :math:`C_{out}`
        bias: 是否学习偏置
        use_bn: 是否使用 BatchNorm
        drop_rate: Dropout 概率
        is_last: 为 ``True`` 时跳过激活与 dropout（由调用方自行处理）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ) -> None:
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""前向传播。

        Args:
            X: 输入顶点特征矩阵 ``[N, C_in]``
            hg: 包含 :math:`N` 个顶点的超图

        Returns:
            输出顶点特征矩阵 ``[N, C_out]``
        """
        X = self.theta(X)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X
