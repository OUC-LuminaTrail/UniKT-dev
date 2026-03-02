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
        2. 历史隔离：同一题目的多个技能共享相同的历史信息（避免数据泄露）
        3. 窗口处理：长序列使用滑动窗口，每个窗口只预测最后一个位置

        DKT 预测语义说明：
        - DKT 模型 y_hat[:, t] 是基于历史 [0, t) 预测位置 t 的结果
        - 因此，预测位置 t 时，输入序列应包含 [0, t) 的历史
        - 当前技能作为预测目标，不应包含在历史中

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

        # 边界条件 1: 检查 fold 列是否存在
        if "fold" not in data.columns:
            raise ValueError(
                "K-fold labels not found in data. Please call data_src.add_kfold_labels() first."
            )

        # 边界条件 2: 筛选测试集数据
        data = data[data["fold"] == -1].copy()
        if data.empty:
            raise ValueError("No test-set interactions (fold == -1) found")

        # 构建题目到技能列表的映射（保持顺序）
        q_skill = question_data[["question", "skill"]].drop_duplicates()
        q_skill_map = (
            q_skill.groupby("question", sort=False)["skill"].apply(list).to_dict()
        )

        # 边界条件 3: 确保所有题目都有对应的技能
        missing_questions = set(data["question"].unique()) - set(q_skill_map.keys())
        if missing_questions:
            self.logger.warning(
                f"Found {len(missing_questions)} questions without skill mapping, "
                "they will be skipped"
            )

        # ==================== 步骤 2: 初始化结果容器 ====================
        win_sequences = []
        win_responses = []
        win_masks = []
        win_users = []
        win_group_ids = []
        win_true_labels = []  # 新增：真实标签容器

        global_group_id = 0  # 全局 group_id 计数器

        # ==================== 步骤 3: 按用户处理 ====================
        total_users = int(data["user"].nunique())

        for user, user_df in tqdm(
            data.groupby("user", sort=False),
            total=total_users,
            desc="Building windowlate data",
        ):
            # 获取用户的答题序列
            questions = user_df["question"].to_numpy()
            labels = user_df["label"].to_numpy(dtype=int)
            n_interactions = len(questions)

            # 边界条件 4: 跳过空序列
            if n_interactions == 0:
                continue

            # 边界条件 5: 跳过所有题目都没有技能映射的用户
            valid_interactions = [
                i for i, q in enumerate(questions) if q in q_skill_map
            ]
            if not valid_interactions:
                continue

            # 只处理有效的交互
            questions = questions[valid_interactions]
            labels = labels[valid_interactions]
            n_interactions = len(questions)

            # ----- 多技能展开 -----
            # 获取每个交互涉及的技能列表
            inter_skills = [np.asarray(q_skill_map[q], dtype=int) for q in questions]
            skill_counts = np.asarray([s.size for s in inter_skills], dtype=int)

            # 计算交互边界（在展开后的技能序列中的位置）
            # inter_boundaries[i] 表示第 i 个交互的技能在展开序列中的起始位置
            inter_boundaries = np.cumsum(np.concatenate([[0], skill_counts]))
            n_total_skills = inter_boundaries[-1]

            # 展开技能和标签
            flat_skills = np.concatenate(inter_skills)
            flat_labels = np.concatenate(
                [
                    np.full(s.size, labels[i], dtype=int)
                    for i, s in enumerate(inter_skills)
                ]
            )
            flat_user_ids = np.full(n_total_skills, int(user), dtype=int)

            # 为每个交互分配 group_id（同一题目的多个技能共享 group_id）
            inter_group_ids = np.arange(
                global_group_id, global_group_id + n_interactions, dtype=np.int64
            )
            flat_group_ids = np.concatenate(
                [
                    np.full(s.size, inter_group_ids[i], dtype=np.int64)
                    for i, s in enumerate(inter_skills)
                ]
            )
            global_group_id += n_interactions

            # ----- 步骤 4: 为每个技能位置构建预测序列 -----
            # 核心逻辑：预测位置 t 时，使用历史 [0, t)
            # 同一题目的多个技能共享相同的历史边界

            for inter_idx in range(n_interactions):
                n_skills = skill_counts[inter_idx]

                # 确定历史边界：历史应截止到当前交互之前
                # 关键设计：同一题目的多个技能共享相同的历史
                # 即历史 = 当前交互之前的所有技能
                history_end = inter_boundaries[inter_idx]

                # 为当前交互的每个技能构建预测序列
                for skill_offset in range(n_skills):
                    current_skill_pos = inter_boundaries[inter_idx] + skill_offset
                    current_skill = flat_skills[current_skill_pos]
                    current_label = flat_labels[current_skill_pos]
                    current_group_id = flat_group_ids[current_skill_pos]

                    # 构建预测序列：历史 [0, history_end) + 当前技能
                    # 注意：DKT 的 y_hat[:, t] 使用输入 [0, t] 预测位置 t 的标签
                    # 输入 x[t] = sequence[t] + num_c * response[t]
                    # 因此 response[t] 会影响位置 t 的编码
                    #
                    # 关键：为避免数据泄露，预测目标的 response 必须为 0！
                    # - 历史位置 [0, history_end)：response = 实际标签
                    # - 目标位置 [history_end]：response = 0（待预测）

                    if history_end == 0:
                        # 边界条件 6: 第一个交互的第一个技能
                        # 历史为空，只有当前技能
                        # DKT 模型的 y[:, 0] = 0，无法进行有效预测
                        # 跳过这种情况
                        continue
                    else:
                        # 正常情况：历史 + 当前技能
                        pred_skills = np.concatenate(
                            [flat_skills[:history_end], [current_skill]]
                        )
                        # 历史位置用实际标签，目标位置用 0
                        pred_labels = np.concatenate(
                            [
                                flat_labels[:history_end],
                                [0],  # ✅ 目标位置的 response = 0，避免数据泄露
                            ]
                        )
                        pred_user_ids = np.concatenate(
                            [flat_user_ids[:history_end], [int(user)]]
                        )
                        pred_group_ids = np.concatenate(
                            [flat_group_ids[:history_end], [current_group_id]]
                        )
                        # 真实标签：历史位置用实际标签，目标位置用真实标签
                        pred_true_labels = np.concatenate(
                            [
                                flat_labels[:history_end],
                                [current_label],  # 目标位置的真实标签
                            ]
                        )

                    seq_len = len(pred_skills)

                    # 构建 selectmask：只有最后一个位置预测
                    # selectmask[i] = 1 表示位置 i 需要预测
                    selectmask = np.zeros(seq_len, dtype=int)
                    selectmask[-1] = 1

                    # ----- 步骤 5: 窗口处理 -----
                    if seq_len <= max_seq_len:
                        # 短序列：填充到 max_seq_len
                        pad_len = max_seq_len - seq_len

                        padded_skills = np.concatenate(
                            [pred_skills, np.zeros(pad_len, dtype=int)]
                        )
                        padded_labels = np.concatenate(
                            [pred_labels, np.zeros(pad_len, dtype=int)]
                        )
                        padded_user_ids = np.concatenate(
                            [pred_user_ids, np.zeros(pad_len, dtype=int)]
                        )
                        padded_group_ids = np.concatenate(
                            [pred_group_ids, np.full(pad_len, -1, dtype=np.int64)]
                        )
                        # selectmask 填充 0
                        padded_selectmask = np.concatenate(
                            [selectmask, np.zeros(pad_len, dtype=int)]
                        )
                        # true_labels 填充 0
                        padded_true_labels = np.concatenate(
                            [pred_true_labels, np.zeros(pad_len, dtype=int)]
                        )

                        win_sequences.append(padded_skills)
                        win_responses.append(padded_labels)
                        win_masks.append(padded_selectmask)
                        win_users.append(padded_user_ids)
                        win_group_ids.append(padded_group_ids)
                        win_true_labels.append(padded_true_labels)

                    else:
                        # 长序列：滑动窗口
                        # 每个窗口大小为 max_seq_len，只预测最后一个位置
                        num_windows = seq_len - max_seq_len + 1

                        for win_idx in range(num_windows):
                            win_start = win_idx
                            win_end = win_idx + max_seq_len

                            window_skills = pred_skills[win_start:win_end]
                            window_labels = pred_labels[win_start:win_end]
                            window_user_ids = pred_user_ids[win_start:win_end]
                            window_group_ids = pred_group_ids[win_start:win_end]
                            window_true_labels = pred_true_labels[win_start:win_end]

                            # 检查窗口最后一个位置是否需要预测
                            if selectmask[win_end - 1] == 1:
                                # 重置 selectmask：只预测最后一个位置
                                window_selectmask = np.zeros(max_seq_len, dtype=int)
                                window_selectmask[-1] = 1

                                win_sequences.append(window_skills)
                                win_responses.append(window_labels)
                                win_masks.append(window_selectmask)
                                win_users.append(window_user_ids)
                                win_group_ids.append(window_group_ids)
                                win_true_labels.append(window_true_labels)

        # ==================== 步骤 6: 结果整合 ====================
        # 边界条件 7: 检查是否生成了有效样本
        if len(win_sequences) == 0:
            raise ValueError(
                "No valid windowlate evaluation samples generated for test set"
            )

        user_sequence = np.stack(win_sequences, axis=0)
        user_response = np.stack(win_responses, axis=0)
        user_mask = np.stack(win_masks, axis=0)
        user_id_sequence = np.stack(win_users, axis=0)
        late_group_id = np.stack(win_group_ids, axis=0)
        user_true_labels = np.stack(win_true_labels, axis=0)

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
