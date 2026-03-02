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
          2. 对每个“题目交互”按其关联技能展开为多个目标样本；同题样本共享 group_id。
          3. 使用“纯问题窗口”确定历史范围：每个目标仅使用此前最多 max_seq_len-1 个题目，
              然后将该问题窗口内历史按技能展开并填入序列；历史从左侧开始填充，目标紧邻历史末尾。
          4. 仅目标位置置 mask=1，用于 late_mean 聚合评估。

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

        history_width = max_seq_len - 1
        if history_width < 1:
            raise ValueError(f"max_seq_len must be at least 2, got {max_seq_len}")

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
            # 如果用户交互数不足以构建至少一个样本则跳过
            if n_interactions <= 1:
                global_group_id += n_interactions
                continue

            # 将题目ID映射到技能ID列表，构建交互对应的技能列表
            inter_skills = [
                np.asarray(q_skill_map[question], dtype=int) for question in questions
            ]
            # 计算每个交互对应的技能数
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

            # 按 PyKT 语义选择需要评估的目标交互：
            # 1) 首窗（长度<=max_seq_len）中除首题外全部目标
            # 2) 当序列超过 max_seq_len 后，后续滑窗仅取窗口末位作为目标
            if n_interactions <= max_seq_len:
                selected_interactions = list(range(1, n_interactions))
            else:
                selected_interactions = list(range(1, max_seq_len)) + list(
                    range(max_seq_len, n_interactions)
                )

            num_samples = int(sum(skill_counts[idx] for idx in selected_interactions))
            if num_samples == 0:
                self.logger.warning(
                    f"User {user} has no valid samples after multi-skill expansion, skipping"
                )
                continue

            seq = np.zeros((num_samples, max_seq_len), dtype=int)
            rsp = np.zeros((num_samples, max_seq_len), dtype=int)
            msk = np.zeros((num_samples, max_seq_len), dtype=int)
            uid = np.zeros((num_samples, max_seq_len), dtype=int)
            gid = np.full((num_samples, max_seq_len), -1, dtype=np.int64)

            # 构造展开后的全历史序列
            flat_skills = np.concatenate(inter_skills)
            flat_labels = np.concatenate(
                [
                    np.full(skills.size, labels[idx], dtype=int)
                    for idx, skills in enumerate(inter_skills)
                ]
            )

            row_ptr = 0
            for inter_idx in selected_interactions:
                # 按问题窗口确定历史边界
                if inter_idx < max_seq_len:
                    history_start_inter = 0
                else:
                    history_start_inter = inter_idx - history_width
                history_end_inter = inter_idx

                history_start_skill = (
                    0
                    if history_start_inter == 0
                    else int(prefix_counts[history_start_inter - 1])
                )
                history_end_skill = int(prefix_counts[history_end_inter - 1])

                hist_skill_full = flat_skills[history_start_skill:history_end_skill]
                hist_label_full = flat_labels[history_start_skill:history_end_skill]

                # 若问题窗口展开后技能数超过 history_width，仅保留最近的 history_width 个
                if hist_skill_full.size > history_width:
                    hist_skill = hist_skill_full[-history_width:]
                    hist_label = hist_label_full[-history_width:]
                else:
                    hist_skill = hist_skill_full
                    hist_label = hist_label_full

                hist_len = int(hist_skill.size)
                target_pos = hist_len

                target_skills = inter_skills[inter_idx]
                target_label = labels[inter_idx]
                target_gid = interaction_group_ids[inter_idx]

                for target_skill in target_skills:
                    if hist_len > 0:
                        seq[row_ptr, :hist_len] = hist_skill
                        rsp[row_ptr, :hist_len] = hist_label
                        uid[row_ptr, :hist_len] = int(user)

                    seq[row_ptr, target_pos] = int(target_skill)
                    rsp[row_ptr, target_pos] = int(target_label)
                    uid[row_ptr, target_pos] = int(user)
                    msk[row_ptr, target_pos] = 1
                    gid[row_ptr, target_pos] = int(target_gid)
                    row_ptr += 1

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
