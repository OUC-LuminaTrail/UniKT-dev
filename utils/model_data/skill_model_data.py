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
        构建用于 windowlateauc_mean 评估的样本。

        核心设计：
        1. 多技能展开：将涉及多个知识点的题目拆分为多个独立交互
        2. 历史隔离：同一题目的多个技能共享相同的历史信息
        3. 窗口处理：长序列使用滑动窗口，每个窗口只预测最后一个位置

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
        from tqdm import tqdm

        # ==================== 步骤 1: 数据准备 ====================
        data = self.data_src.get_sequence_data().copy()
        question_data = self.data_src.get_question_data()

        # 检查 fold 列是否存在
        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # 筛选测试集数据
        data = data[data["fold"] == -1].copy()
        if data.empty:
            raise ValueError("No test-set interactions (fold == -1) found")

        # 构建题目到技能列表的映射
        q_skill = question_data[["question", "skill"]].drop_duplicates()
        q_skill_map = (
            q_skill.groupby("question", sort=False)["skill"].apply(list).to_dict()
        )

        # ==================== 步骤 2: 第一遍 - 计算样本数并收集元数据 ====================
        user_data_list = []
        total_samples = 0
        global_group_id = 0
        total_users = int(data["user"].nunique())

        for user, user_df in tqdm(
            data.groupby("user", sort=False),
            total=total_users,
            desc="Counting samples",
        ):
            questions = user_df["question"].to_numpy()
            labels = user_df["label"].to_numpy(dtype=int)

            # 过滤无效题目
            valid_mask = np.array([q in q_skill_map for q in questions])
            if not valid_mask.any():
                continue

            questions = questions[valid_mask]
            labels = labels[valid_mask]
            n_interactions = len(questions)

            if n_interactions == 0:
                continue

            # 多技能展开
            inter_skills = [np.asarray(q_skill_map[q], dtype=int) for q in questions]
            skill_counts = np.asarray([s.size for s in inter_skills], dtype=int)
            inter_boundaries = np.cumsum(np.concatenate([[0], skill_counts]))

            flat_skills = np.concatenate(inter_skills)
            flat_labels = np.concatenate(
                [
                    np.full(s.size, labels[i], dtype=int)
                    for i, s in enumerate(inter_skills)
                ]
            )

            # 计算该用户的样本数
            user_sample_count = 0
            sample_info = []

            for inter_idx in range(n_interactions):
                n_skills = skill_counts[inter_idx]
                history_end = inter_boundaries[inter_idx]

                if history_end == 0:
                    continue

                for skill_offset in range(n_skills):
                    seq_len = history_end + 1

                    if seq_len <= max_seq_len:
                        user_sample_count += 1
                        sample_info.append((inter_idx, skill_offset, seq_len, None))
                    else:
                        # 滑动窗口：只有最后一个窗口（selectmask[-1] == 1）
                        num_windows = seq_len - max_seq_len + 1
                        win_idx = num_windows - 1  # 只保留最后一个窗口
                        user_sample_count += 1
                        sample_info.append((inter_idx, skill_offset, seq_len, win_idx))

            if user_sample_count > 0:
                user_data_list.append(
                    {
                        "user": int(user),
                        "questions": questions,
                        "labels": labels,
                        "inter_skills": inter_skills,
                        "skill_counts": skill_counts,
                        "inter_boundaries": inter_boundaries,
                        "flat_skills": flat_skills,
                        "flat_labels": flat_labels,
                        "sample_info": sample_info,
                        "global_group_id_start": global_group_id,
                    }
                )
                global_group_id += n_interactions
                total_samples += user_sample_count

        # 检查是否生成了有效样本
        if total_samples == 0:
            raise ValueError(
                "No valid windowlate evaluation samples generated for test set"
            )

        # ==================== 步骤 3: 预分配并填充数组 ====================
        user_sequence = np.zeros((total_samples, max_seq_len), dtype=int)
        user_response = np.zeros((total_samples, max_seq_len), dtype=int)
        user_mask = np.zeros((total_samples, max_seq_len), dtype=int)
        user_id_sequence = np.zeros((total_samples, max_seq_len), dtype=int)
        late_group_id = np.full((total_samples, max_seq_len), -1, dtype=np.int64)
        user_true_labels = np.zeros((total_samples, max_seq_len), dtype=int)

        sample_idx = 0

        for user_data in tqdm(
            user_data_list,
            desc="Building samples",
        ):
            user = user_data["user"]
            inter_skills = user_data["inter_skills"]
            skill_counts = user_data["skill_counts"]
            inter_boundaries = user_data["inter_boundaries"]
            flat_skills = user_data["flat_skills"]
            flat_labels = user_data["flat_labels"]
            sample_info = user_data["sample_info"]
            global_group_id_start = user_data["global_group_id_start"]

            n_interactions = len(inter_skills)
            n_total_skills = inter_boundaries[-1]

            # 展开用户ID和group_id
            flat_user_ids = np.full(n_total_skills, user, dtype=int)
            inter_group_ids = np.arange(
                global_group_id_start,
                global_group_id_start + n_interactions,
                dtype=np.int64,
            )
            flat_group_ids = np.concatenate(
                [
                    np.full(s.size, inter_group_ids[i], dtype=np.int64)
                    for i, s in enumerate(inter_skills)
                ]
            )

            for inter_idx, skill_offset, seq_len, win_idx in sample_info:
                history_end = inter_boundaries[inter_idx]
                current_skill_pos = inter_boundaries[inter_idx] + skill_offset
                current_skill = flat_skills[current_skill_pos]
                current_label = flat_labels[current_skill_pos]
                current_group_id = flat_group_ids[current_skill_pos]

                if seq_len <= max_seq_len:
                    # 短序列：直接填充
                    # 填充技能序列
                    user_sequence[sample_idx, :history_end] = flat_skills[:history_end]
                    user_sequence[sample_idx, history_end] = current_skill

                    # 填充响应序列（历史用真实标签，目标用0）
                    user_response[sample_idx, :history_end] = flat_labels[:history_end]

                    # 填充用户ID
                    user_id_sequence[sample_idx, :history_end] = flat_user_ids[
                        :history_end
                    ]
                    user_id_sequence[sample_idx, history_end] = user

                    # 填充group_id
                    late_group_id[sample_idx, :history_end] = flat_group_ids[
                        :history_end
                    ]
                    late_group_id[sample_idx, history_end] = current_group_id

                    # 填充真实标签
                    user_true_labels[sample_idx, :history_end] = flat_labels[
                        :history_end
                    ]
                    user_true_labels[sample_idx, history_end] = current_label

                    # 设置mask（只有最后一个位置预测）
                    user_mask[sample_idx, history_end] = 1

                else:
                    # 长序列：滑动窗口
                    win_start = win_idx
                    win_end = win_idx + max_seq_len

                    # 构建完整预测序列
                    pred_skills = np.concatenate(
                        [flat_skills[:history_end], [current_skill]]
                    )
                    pred_labels = np.concatenate([flat_labels[:history_end], [0]])
                    pred_user_ids = np.concatenate(
                        [flat_user_ids[:history_end], [user]]
                    )
                    pred_group_ids = np.concatenate(
                        [flat_group_ids[:history_end], [current_group_id]]
                    )
                    pred_true_labels = np.concatenate(
                        [flat_labels[:history_end], [current_label]]
                    )

                    # 切片窗口
                    user_sequence[sample_idx] = pred_skills[win_start:win_end]
                    user_response[sample_idx] = pred_labels[win_start:win_end]
                    user_id_sequence[sample_idx] = pred_user_ids[win_start:win_end]
                    late_group_id[sample_idx] = pred_group_ids[win_start:win_end]
                    user_true_labels[sample_idx] = pred_true_labels[win_start:win_end]
                    user_mask[sample_idx, -1] = 1

                sample_idx += 1

        self.logger.debug(
            f"Built windowlate data: samples={user_sequence.shape[0]}, "
            f"max_seq_len={max_seq_len}"
        )

        return (
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            late_group_id,
            user_true_labels,
        )
