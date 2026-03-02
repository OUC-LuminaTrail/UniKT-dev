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

    def build_windowlate_data(self, max_seq_len: int):
        r"""
          构建用于 windowlateauc_mean 的评估样本。

          处理流程：
          1. 仅使用测试集用户交互（fold == -1）。
          2. 对每个"题目交互"按其关联技能展开为多个目标样本；同题样本共享 group_id。
          3. 滑动窗口构建：生成固定长度为 max_seq_len 的窗口序列，每个窗口包含
             max_seq_len 个交互。对于展开后的多技能，一个窗口可能超过 max_seq_len 个位置。
          4. 第一个窗口（Win-0）：所有位置都预测（mask=1）
          5. 后续窗口（Win-1, Win-2, ...）：只预测最后一个位置（mask=1），其他位置为0

        参数:
            max_seq_len: 最大序列长度

        返回:
            user_sequence: 评估序列，shape=(num_samples, max_seq_len)
            user_response: 响应序列，shape=(num_samples, max_seq_len)
            user_mask: 仅目标位置有效的掩码，shape=(num_samples, max_seq_len)
            user_id_sequence: 用户ID序列，shape=(num_samples, max_seq_len)
            late_group_id: 逐时间步分组ID，shape=(num_samples, max_seq_len)
        """
        import numpy as np

        data = self.data_src.get_sequence_data().copy()
        question_data = self.data_src.get_question_data()

        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # fold == -1 代表测试集用户
        data = data[data["fold"] == -1].copy()
        if data.empty:
            raise ValueError("No test-set interactions (fold == -1) found")

        q_skill = question_data[["question", "skill"]].drop_duplicates()
        q_skill_map = (
            q_skill.groupby("question", sort=False)["skill"].apply(list).to_dict()
        )

        win_sequences = []
        win_responses = []
        win_masks = []
        win_users = []
        win_group_ids = []

        # 同一用户的交互展开记录
        global_group_id = 0

        for user, user_df in data.groupby("user", sort=False):
            # 获取用户的题目和标签序列
            questions = user_df["question"].to_numpy()
            labels = user_df["label"].to_numpy(dtype=int)
            n_interactions = len(questions)
            # 如果用户交互数不足以构建至少一个完整窗口则跳过
            if n_interactions < max_seq_len:
                global_group_id += n_interactions
                self.logger.warning(
                    f"User {user} has only {n_interactions} interactions, less than max_seq_len={max_seq_len}. Skipping user for window_late evaluation."
                )
                continue

            # 将题目ID映射到技能ID列表，构建交互对应的技能列表
            inter_skills = [
                np.asarray(q_skill_map[question], dtype=int) for question in questions
            ]
            # 计算每个交互对应的技能数和前缀和
            skill_counts = np.asarray(
                [skills.size for skills in inter_skills], dtype=int
            )
            prefix_counts = np.cumsum(skill_counts)

            # 每个题目交互都拥有唯一 group_id；同题多技能共享 group_id
            interaction_group_ids = np.arange(
                global_group_id,
                global_group_id + n_interactions,
                dtype=np.int64,
            )
            global_group_id += n_interactions

            # 滑动窗口：每个窗口包含 max_seq_len 个交互
            # 从位置 0 到 n_interactions - max_seq_len
            num_windows = n_interactions - max_seq_len + 1

            # 构造展开后的全历史序列（用于快速提取窗口）
            flat_skills = np.concatenate(inter_skills)
            flat_labels = np.concatenate(
                [
                    np.full(skills.size, labels[idx], dtype=int)
                    for idx, skills in enumerate(inter_skills)
                ]
            )
            flat_user_ids = np.concatenate(
                [np.full(skills.size, int(user), dtype=int) for skills in inter_skills]
            )
            flat_group_ids = np.concatenate(
                [
                    np.full(skills.size, interaction_group_ids[idx], dtype=np.int64)
                    for idx, skills in enumerate(inter_skills)
                ]
            )

            for window_idx in range(num_windows):
                # 确定窗口在交互序列中的起止位置
                window_start_inter = window_idx
                window_end_inter = window_idx + max_seq_len

                # 确定窗口在技能展开序列中的起止位置
                window_start_skill = (
                    0
                    if window_start_inter == 0
                    else int(prefix_counts[window_start_inter - 1])
                )
                window_end_skill = int(prefix_counts[window_end_inter - 1])

                window_skills = flat_skills[window_start_skill:window_end_skill]
                window_labels = flat_labels[window_start_skill:window_end_skill]
                window_users = flat_user_ids[window_start_skill:window_end_skill]
                window_group_ids = flat_group_ids[window_start_skill:window_end_skill]

                window_len = len(window_skills)

                # 如果窗口展开后的技能数超过 max_seq_len，截取最近的 max_seq_len 个
                if window_len > max_seq_len:
                    window_skills = window_skills[-max_seq_len:]
                    window_labels = window_labels[-max_seq_len:]
                    window_users = window_users[-max_seq_len:]
                    window_group_ids = window_group_ids[-max_seq_len:]
                    window_len = max_seq_len

                # 创建一个样本
                seq = np.zeros((1, max_seq_len), dtype=int)
                rsp = np.zeros((1, max_seq_len), dtype=int)
                msk = np.zeros((1, max_seq_len), dtype=int)
                uid = np.zeros((1, max_seq_len), dtype=int)
                gid = np.full((1, max_seq_len), -1, dtype=np.int64)

                # 填充窗口数据
                seq[0, :window_len] = window_skills
                rsp[0, :window_len] = window_labels
                uid[0, :window_len] = window_users
                gid[0, :window_len] = window_group_ids

                # 设置 mask：
                # - 第一个窗口（window_idx == 0）：所有位置都预测
                # - 后续窗口：只预测最后一个位置
                if window_idx == 0:
                    msk[0, :window_len] = 1
                else:
                    msk[0, window_len - 1] = 1

                win_sequences.append(seq)
                win_responses.append(rsp)
                win_masks.append(msk)
                win_users.append(uid)
                win_group_ids.append(gid)

        if len(win_sequences) == 0:
            raise ValueError(
                "No valid window_late evaluation samples generated for test set"
            )

        user_sequence = np.concatenate(win_sequences, axis=0).astype(int, copy=False)
        user_response = np.concatenate(win_responses, axis=0).astype(int, copy=False)
        user_mask = np.concatenate(win_masks, axis=0).astype(int, copy=False)
        user_id_sequence = np.concatenate(win_users, axis=0).astype(int, copy=False)
        late_group_id = np.concatenate(win_group_ids, axis=0).astype(
            np.int64, copy=False
        )

        self.logger.debug(
            f"Built window_late data: samples={user_sequence.shape[0]}, max_seq_len={max_seq_len}"
        )

        return (
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            late_group_id,
        )
