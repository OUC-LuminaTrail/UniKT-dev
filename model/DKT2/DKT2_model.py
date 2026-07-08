import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Linear, ReLU, Sequential
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)

from utils.core import get_logger

logger = get_logger(__name__)


class DKT2(nn.Module):
    """DKT2 主模型。

    每步先构造难度（Rasch 型）调制的交互嵌入，送入 xLSTM 得到隐状态；再按
    “作答正确/错误”将隐状态分流为熟悉/不熟悉能力，与去难度隐状态、概念嵌入
    拼接后经 MLP 输出全概念概率向量，最后按下一概念 one-hot 收集得到该步预测。
    """

    def __init__(
        self,
        num_skills,
        num_questions,
        batch_size,
        seq_len,
        factor=1.3,
        num_blocks=1,
        num_heads=2,
        slstm_at=(0,),
        conv1d_kernel_size=4,
        qkv_proj_blocksize=4,
        embedding_size=64,
        dropout=0.2,
        slstm_backend="cuda",
        length=1,
        joint=False,
        mask_future=False,
    ):
        super().__init__()
        self.num_skills = num_skills
        self.num_questions = num_questions
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.dropout = dropout
        self.factor = factor
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.slstm_at = list(slstm_at)
        self.conv1d_kernel_size = conv1d_kernel_size
        self.qkv_proj_blocksize = qkv_proj_blocksize
        self.slstm_backend = slstm_backend
        self.length = length
        self.joint = joint
        self.mask_future = mask_future

        # 题目难度参数（Rasch 型）。padding 题目映射到 num_questions 行。
        self.difficult_param = nn.Embedding(self.num_questions + 1, 1)
        self.q_embed_diff = nn.Embedding(self.num_skills + 1, self.embedding_size)
        self.qa_embed_diff = nn.Embedding(2 * self.num_skills + 1, self.embedding_size)

        # 概念 / 交互嵌入，第 0 行为 padding。
        self.q_embed = nn.Embedding(self.num_skills, self.embedding_size)
        self.qa_embed = nn.Embedding(2, self.embedding_size)

        self.xlstm_stack = self._build_xlstm()

        self.dropout_layer = Dropout(dropout)
        self.out = Sequential(
            Linear(
                2 * self.embedding_size + 2 * self.hidden_size, 2 * self.hidden_size
            ),
            ReLU(),
            Dropout(self.dropout),
            Linear(2 * self.hidden_size, self.hidden_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.hidden_size, self.num_skills),
        )

    def _build_xlstm(self):
        """构造 xLSTM 块栈。

        sLSTM 的 cuda 后端依赖即时编译的 fused kernel，部分工具链下可能链接失败。
        """
        try:
            return xLSTMBlockStack(self._make_xlstm_config(self.slstm_backend))
        except (RuntimeError, OSError) as e:
            if self.slstm_backend == "cuda":
                logger.warning(
                    f"sLSTM 'cuda' backend unavailable ({type(e).__name__}: {e}); "
                    "falling back to 'vanilla' backend."
                )
                self.slstm_backend = "vanilla"
                return xLSTMBlockStack(self._make_xlstm_config("vanilla"))
            raise

    def _make_xlstm_config(self, backend):
        return xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    qkv_proj_blocksize=self.qkv_proj_blocksize,
                    num_heads=self.num_heads,
                    proj_factor=self.factor,
                    dropout=self.dropout,
                    embedding_dim=self.embedding_size,
                    _inner_embedding_dim=2 * self.embedding_size,
                    _num_blocks=1,
                    round_proj_up_dim_up=True,
                    _proj_up_dim=None,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=self.num_heads,
                    conv1d_kernel_size=self.conv1d_kernel_size,
                    bias_init="powerlaw_blockdependent",
                    recurrent_weight_init="zeros",
                    embedding_dim=self.embedding_size,
                    dropout=self.dropout,
                    group_norm_weight=True,
                    gradient_recurrent_cut=False,
                    gradient_recurrent_clipval=None,
                    forward_clipval=None,
                    batch_size=self.batch_size,
                ),
                feedforward=FeedForwardConfig(proj_factor=self.factor, act_fn="relu"),
            ),
            context_length=self.seq_len - 1,
            num_blocks=self.num_blocks,
            embedding_dim=self.embedding_size,
            add_post_blocks_norm=True,
            bias=True,
            dropout=self.dropout,
            slstm_at=self.slstm_at,
        )

    def forward(self, questions, skills, responses, mask):
        """返回 (预测概率, 对齐标签)。

        output[t] 为看到前 t+1 步交互后对“下一概念”的预测概率；
        r_shft[t] 为对应的下一作答标签。二者逐位对齐，长度均为 S-length。
        """
        # next-item 偏移：输入 [0..S-length-1]，目标 [length..S-1]
        pid_data = questions[:, : -self.length]
        q_data = skills[:, : -self.length]
        q_shft = skills[:, self.length :]
        r_shft = responses[:, self.length :]
        target = responses[:, : -self.length]

        # padding 题目映射到独立的难度行，避免借用真实题目 0 的难度
        cur_mask = mask[:, : -self.length]
        pid_data = pid_data.masked_fill(~cur_mask, self.num_questions)

        # 基础交互嵌入：c_ct + g_rt
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(target) + q_embed_data

        # 难度调制：mu * diff，分别加到概念嵌入与交互嵌入上
        q_embed_diff_data = self.q_embed_diff(q_data)
        pid_embed_data = self.difficult_param(pid_data)
        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

        qa_embed_diff_data = self.qa_embed_diff(target)
        qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data

        qa_embed_data = self.dropout_layer(qa_embed_data)
        d_output = self.xlstm_stack(qa_embed_data)

        # 按作答正误将隐状态分流为熟悉/不熟悉能力向量
        familiar_ability = torch.zeros_like(d_output)
        unfamiliar_ability = torch.zeros_like(d_output)
        familiar_position = target == 1
        unfamiliar_position = target == 0
        familiar_ability[familiar_position] = d_output[familiar_position]
        unfamiliar_ability[unfamiliar_position] = d_output[unfamiliar_position]

        # 去掉难度分量后，与概念嵌入、熟悉/不熟悉能力拼接，经 MLP 输出全概念 logits
        d_output = d_output - pid_embed_data
        concat_q = torch.cat(
            [d_output, q_embed_data, familiar_ability, unfamiliar_ability], dim=-1
        )
        output = self.out(concat_q)

        if self.joint:
            seq_len = q_data.size(1)
            mid = seq_len // 2
            output[:, mid:, :] = output[:, mid : mid + 1, :].expand(
                -1, seq_len - mid, -1
            )

        output = torch.sigmoid(output)
        # 按下一概念 one-hot 收集，得到该步对下一题的预测概率
        output = (output * F.one_hot(q_shft.long(), self.num_skills)).sum(-1)

        if self.mask_future:
            output = output[:, -self.length :]
            r_shft = r_shft[:, -self.length :]
        elif self.joint:
            mid = q_data.size(1) // 2
            output = output[:, mid:]
            r_shft = r_shft[:, mid:]

        return output, r_shft
