from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import BaseModelData


class SkillModelData(BaseModelData):
    """
    技能序列数据基类

    用于构建基于技能（skill/concept）的知识追踪模型数据
    """

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def build_sequence_data(self, max_seq_len: int):
        r"""
        构建用户答题序列，将问题ID映射到技能ID，并展开多知识点

        参数:
            max_seq_len: 最大序列长度

        返回:
            user_sequence: 用户技能ID序列，shape为(num_users, max_seq_len)
            user_response: 用户响应序列，shape为(num_users, max_seq_len)
            user_mask: 用户掩码序列，shape为(num_users, max_seq_len)
            user_id_sequence: 用户ID序列，shape为(num_users, max_seq_len)
        """
        import numpy as np
        from tqdm import tqdm

        data = self.data_src.get_sequence_data()
        question_data = self.data_src.get_question_data()
        num_users = self.data_src.get_metadata("num_users")

        # 构建问题ID到技能ID列表的映射
        # 在数据预处理中，question_data已经将多知识点展开为多个技能ID
        # 例如：question_id=1, skill=10; question_id=1, skill=20
        question_to_skills = {}
        for row in question_data.itertuples():
            qid = row.question
            sid = row.skill
            if qid not in question_to_skills:
                question_to_skills[qid] = []
            question_to_skills[qid].append(sid)

        # 初始化序列数组
        user_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((num_users, max_seq_len), dtype=int)
        user_response = np.zeros((num_users, max_seq_len), dtype=int)
        user_mask = np.zeros((num_users, max_seq_len), dtype=int)

        # 用户序列长度计数
        num_sequence = [0] * num_users

        # 按用户分组构建序列，展开多知识点
        for row in tqdm(
            data.itertuples(), total=data.shape[0], desc="Building skill sequences"
        ):
            user_idx = row.user
            question_idx = row.question
            label = row.label

            # 获取该问题对应的所有技能
            skills = question_to_skills.get(question_idx, [0])

            # 展开多知识点：一个问题对应多个技能时，展开成多个独立的交互
            for skill_idx in skills:
                # 如果当前用户的序列长度未达到最大长度，则添加数据
                if num_sequence[user_idx] < max_seq_len:
                    user_sequence[user_idx, num_sequence[user_idx]] = skill_idx
                    user_id_sequence[user_idx, num_sequence[user_idx]] = user_idx
                    user_response[user_idx, num_sequence[user_idx]] = label
                    user_mask[user_idx, num_sequence[user_idx]] = 1
                    num_sequence[user_idx] += 1
                else:
                    # 如果序列已满，跳过剩余的技能
                    break

        self.logger.debug(
            f"Built skill sequences for {num_users} users, max_len={max_seq_len}"
        )
        self.logger.debug(
            f"Multi-skill expansion applied: {len(question_to_skills)} questions mapped to skills"
        )

        return user_sequence, user_response, user_mask, user_id_sequence

    def load_windowlate_data(self, max_seq_len: int):
        r"""
        加载用于 windowlateauc_mean 评估的样本。

        从预处理的 Parquet 文件加载滑动窗口数据，并转换为 numpy 数组。

        参数:
            max_seq_len: 最大序列长度（窗口大小）

        返回:
            user_sequence: 技能序列，shape=(num_samples, max_seq_len)
            user_response: 响应序列，shape=(num_samples, max_seq_len)
            user_mask: 预测掩码，shape=(num_samples, max_seq_len)，1 表示需要预测
            user_id_sequence: 用户ID序列，shape=(num_samples, max_seq_len)
            late_group_id: 题目级分组ID，shape=(num_samples, max_seq_len)
            user_true_labels: 真实标签序列，shape=(num_samples, max_seq_len)
        """
        import numpy as np

        # 从预处理文件加载长格式数据
        data = self.data_src.get_windowlate_data()

        if data is None or len(data) == 0:
            raise ValueError(
                "No windowlate data available. Please re-run preprocessing with K-fold labels."
            )

        required_cols = [
            "sample_id",
            "position",
            "skill",
            "response",
            "mask",
            "user_id",
            "group_id",
            "true_label",
        ]
        data = data[required_cols]

        # 转为 numpy 数组
        sample_ids = data["sample_id"].to_numpy(copy=False)
        positions = data["position"].to_numpy(copy=False)

        # 获取样本数量
        sample_id_series = data["sample_id"]
        num_samples = sample_id_series.n_unique()
        num_samples = int(num_samples)

        # sample_id 直接作为行索引，要求是 [0, num_samples) 连续编号
        max_sample_id = int(sample_ids.max())
        if max_sample_id >= num_samples:
            raise ValueError(
                "Invalid sample_id values. Expected contiguous sample_id in "
                f"[0, {num_samples}), but got max(sample_id)={max_sample_id}."
            )

        # 初始化数组
        user_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        user_response = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_mask = np.zeros((num_samples, max_seq_len), dtype=np.int8)
        user_id_sequence = np.zeros((num_samples, max_seq_len), dtype=np.int32)
        late_group_id = np.full((num_samples, max_seq_len), -1, dtype=np.int64)
        user_true_labels = np.zeros((num_samples, max_seq_len), dtype=np.int8)

        # 填充数据
        user_sequence[sample_ids, positions] = data["skill"].to_numpy(copy=False)
        user_response[sample_ids, positions] = data["response"].to_numpy(copy=False)
        user_mask[sample_ids, positions] = data["mask"].to_numpy(copy=False)
        user_id_sequence[sample_ids, positions] = data["user_id"].to_numpy(copy=False)
        late_group_id[sample_ids, positions] = data["group_id"].to_numpy(copy=False)
        user_true_labels[sample_ids, positions] = data["true_label"].to_numpy(
            copy=False
        )

        self.logger.debug(
            f"Loaded windowlate data: samples={num_samples}, max_seq_len={max_seq_len}"
        )

        return (
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            late_group_id,
            user_true_labels,
        )
