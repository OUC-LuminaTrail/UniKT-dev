"""
DYGKT 模型（严格复刻原始实现）

关键改动：
1. GRU 不使用 batch_first（与原始一致）
2. 接受 batch 字典输入（包含历史序列）
3. 使用 performance_encoder 编码历史正确率
4. 用户和问题使用各自独立的历史序列
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utils.core import register_model


class TimeDualDecayEncoder(nn.Module):
    """时间双衰减编码器（与原始完全相同）。"""
    
    def __init__(self, dim_time: int, parameter_requires_grad: bool = True) -> None:
        super().__init__()
        self.time_dim = dim_time
        
        # 短期时间衰减权重
        self.w_short = nn.Linear(1, dim_time)
        self.w_short.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_short.bias = nn.Parameter(torch.zeros(dim_time))
        
        # 长期时间衰减权重
        self.w_long = nn.Linear(1, dim_time)
        self.w_long.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_long.bias = nn.Parameter(torch.zeros(dim_time))
        
        # 输出投影层
        self.w_o = nn.Linear(dim_time, dim_time)
        self.w_o.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dim_time * dim_time, dtype=np.float32)).reshape(dim_time, -1)
        )
        self.w_o.bias = nn.Parameter(torch.zeros(dim_time))
        
        self.f = nn.ReLU()
        
        if not parameter_requires_grad:
            self.w_short.weight.requires_grad = False
            self.w_short.bias.requires_grad = False
            self.w_long.weight.requires_grad = False
            self.w_long.bias.requires_grad = False
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """前向传播（与原始完全相同）。"""
        timestamps = timestamps.unsqueeze(dim=2)  # [B, S, 1]
        
        timestamps_right = timestamps.clone()
        timestamps_right = torch.cat(
            [timestamps_right[:, 1:, :], timestamps_right[:, -1, :].unsqueeze(1)],
            dim=1
        )
        timestamps_diff = timestamps_right - timestamps
        
        timestamps_mask = (timestamps_diff > 3600 * 24).float()
        
        timestamps_short = self.f(self.w_short(timestamps_diff * timestamps_mask))
        timestamps_long = self.f(self.w_long(timestamps_diff * (1 - timestamps_mask)))
        
        output = self.w_o(timestamps_short + timestamps_long)
        
        return output


class DyKT_Seq(nn.Module):
    """动态序列更新模块（与原始完全相同）。"""
    
    def __init__(self, edge_dim: int, node_dim: int) -> None:
        super().__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        # 注意：batch_first=True 与原始相同
        self.hid_node_updater = nn.GRU(
            input_size=edge_dim,
            hidden_size=node_dim,
            batch_first=True
        )
    
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """更新节点状态。"""
        outputs, _ = self.hid_node_updater(x)
        return torch.squeeze(outputs, dim=0)


@register_model("DYGKT")
class DYGKT(nn.Module):
    """DYGKT 模型（严格复刻原始实现）。"""
    
    def __init__(self, args: Any, data_metadata: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.args = args
        self.data_metadata = data_metadata
        
        # 模型参数（从原始 model_config 获取）
        self.dim_emb = getattr(args, 'embedding_dim', 128)
        self.dim_time = getattr(args, 'dim_time', 64)
        
        # 原始实现的所有层（严格复刻 L60-67）
        self.performance_encoder = nn.Linear(1, 64)
        self.dual_time_encoder = TimeDualDecayEncoder(self.dim_time)
        self.multiset_indicator = nn.Linear(1, 64)
        self.gru_linear4user = nn.Linear(self.dim_emb, 64)
        # 注意：原始实现没有 batch_first 参数，默认为 False
        self.gru4user = nn.GRU(self.dim_emb, 64)
        self.gru_linear4que = nn.Linear(self.dim_emb, 64)
        self.gru4que = nn.GRU(self.dim_emb, 64)
        
        # 预测层配置（严格对齐 pyedmine PredictorLayer）
        predictor_config = {
            "type": "direct",
            "dim_predict_in": 64 + 64 + self.dim_time,  # user + que + time
            "dim_predict_mid": getattr(args, 'hidden_dim', 128),
            "dim_predict_out": 1,
            "dropout": getattr(args, 'dropout', 0.3),
            "num_predict_layer": getattr(args, 'num_predict_layer', 2),
            "activate_type": getattr(args, 'activate_type', "relu"),
        }
        self.predict_layer = self._create_predictor(predictor_config)
    
    def _create_predictor(self, config):
        """创建预测层（严格复刻 pyedmine PredictorLayer 逻辑）。"""
        dropout = config["dropout"]
        num_predict_layer = config["num_predict_layer"]
        dim_predict_in = config["dim_predict_in"]
        dim_predict_mid = config["dim_predict_mid"]
        activate_type = config["activate_type"]

        if activate_type == "tanh":
            act_func = nn.Tanh
        elif activate_type == "relu":
            act_func = nn.ReLU
        else:
            act_func = nn.Sigmoid

        dim_predict_out = config["dim_predict_out"]
        layers = []
        if num_predict_layer == 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_predict_in, dim_predict_out))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(dim_predict_in, dim_predict_mid))
            for _ in range(num_predict_layer - 1):
                layers.append(act_func())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(dim_predict_mid, dim_predict_mid))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_predict_mid, dim_predict_out))
            layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)

    def _get_batch_tensor(self, batch: dict, keys: list[str]) -> torch.Tensor:
        for key in keys:
            if key in batch:
                return batch[key]
        raise KeyError(f"Missing required batch keys: {keys}")

    def _fit_to_dim_emb(self, x: torch.Tensor) -> torch.Tensor:
        """将输入特征调整到 GRU 需要的 dim_emb。"""
        feat_dim = x.size(-1)
        if feat_dim == self.dim_emb:
            return x
        if feat_dim > self.dim_emb:
            return x[..., : self.dim_emb]
        pad_size = self.dim_emb - feat_dim
        return torch.nn.functional.pad(x, (0, pad_size), value=0.0)

    @staticmethod
    def _select_last_valid(sequence: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        """从 [B, N, D] 中按每个样本的 last_idx 取最后一个有效向量。"""
        batch_size = sequence.size(0)
        safe_last = torch.clamp(last_idx.long() - 1, min=0)
        out = sequence[torch.arange(batch_size, device=sequence.device), safe_last]
        valid = (last_idx > 0).unsqueeze(-1)
        return out * valid
    
    def get_user_que_embedding(self, batch):
        """获取用户和问题的嵌入（严格复刻原始 L69-73）。
        
        原始实现：
            X_se = self.performance_encoder(batch["user_history_correctness_seq"])
            X_qe = self.performance_encoder(batch["que_history_correctness_seq"])
            X_st = self.dual_time_encoder(batch["user_history_time_seq"])
            X_qt = self.dual_time_encoder(batch["que_history_time_seq"])
        """
        # 历史正确率编码（需要添加维度：[B, N] -> [B, N, 1]）
        user_his_correctness = self._get_batch_tensor(
            batch, ["user_his_correctness_seq", "user_history_correctness_seq"]
        ).unsqueeze(-1).float()
        que_his_correctness = self._get_batch_tensor(
            batch, ["que_his_correctness_seq", "que_history_correctness_seq"]
        ).unsqueeze(-1).float()
        
        X_se = self.performance_encoder(user_his_correctness)  # [B, N, 64]
        X_qe = self.performance_encoder(que_his_correctness)   # [B, N, 64]
        
        # 时间编码
        X_st = self.dual_time_encoder(
            self._get_batch_tensor(batch, ["user_his_time_seq", "user_history_time_seq"]).float()
        )  # [B, N, dim_time]
        X_qt = self.dual_time_encoder(
            self._get_batch_tensor(batch, ["que_his_time_seq", "que_history_time_seq"]).float()
        )  # [B, N, dim_time]
        
        return X_se, X_qe, X_st, X_qt
    
    def forward(self, batch: dict) -> torch.Tensor:
        """前向传播（接受 batch 字典）。
        
        Args:
            batch: 包含以下字段的字典
                - user: 用户ID [B]
                - question: 问题ID [B]
                - correctness: 正确性 [B]
                - time: 时间戳 [B]
                - user_his_correctness_seq: 用户历史正确率 [B, N]
                - que_his_correctness_seq: 问题历史正确率 [B, N]
                - user_his_time_seq: 用户历史时间 [B, N]
                - que_his_time_seq: 问题历史时间 [B, N]
                
        Returns:
            score: 预测概率 [B]
        """
        # 获取用户和问题的历史嵌入
        X_se, X_qe, X_st, X_qt = self.get_user_que_embedding(batch)

        user_last_idx = self._get_batch_tensor(
            batch, ["user_his_last_idx", "user_history_last_idx"]
        )
        que_last_idx = self._get_batch_tensor(
            batch, ["que_his_last_idx", "que_history_last_idx"]
        )

        # 组合成 GRU 输入，并匹配 dim_emb
        user_gru_input = self._fit_to_dim_emb(torch.cat([X_se, X_st], dim=-1))
        que_gru_input = self._fit_to_dim_emb(torch.cat([X_qe, X_qt], dim=-1))

        # 原始实现 GRU 为 batch_first=False，先转置到 [N, B, D]
        user_gru_out, _ = self.gru4user(user_gru_input.transpose(0, 1))
        que_gru_out, _ = self.gru4que(que_gru_input.transpose(0, 1))

        user_gru_out = user_gru_out.transpose(0, 1)
        que_gru_out = que_gru_out.transpose(0, 1)

        # 取最后有效状态并叠加线性映射分支
        user_state = self._select_last_valid(user_gru_out, user_last_idx) + self.gru_linear4user(
            self._select_last_valid(user_gru_input, user_last_idx)
        )
        que_state = self._select_last_valid(que_gru_out, que_last_idx) + self.gru_linear4que(
            self._select_last_valid(que_gru_input, que_last_idx)
        )

        # 使用 multiset 指示器增强状态（保留 pyedmine 的层定义语义）
        same_question = self._get_batch_tensor(
            batch, ["user_his_snq_seq", "user_his_snd_seq", "user_history_snq_seq"]
        ).float()
        same_knowledge = self._get_batch_tensor(
            batch, ["user_his_snk_seq", "user_history_snk_seq"]
        ).float()
        indicator = ((same_question > 0) | (same_knowledge > 0)).float().unsqueeze(-1)
        multiset_feat = self.multiset_indicator(indicator).mean(dim=1)
        user_state = user_state + multiset_feat
        que_state = que_state + multiset_feat

        # 时间上下文使用用户/问题最后有效时间编码平均
        time_feat = (
            self._select_last_valid(X_st, user_last_idx)
            + self._select_last_valid(X_qt, que_last_idx)
        ) / 2.0
        
        # 拼接所有特征
        combined = torch.cat([user_state, que_state, time_feat], dim=-1)  # [B, 64+64+dim_time]
        
        # 预测
        score = self.predict_layer(combined).squeeze(-1)  # [B]
        
        return score
