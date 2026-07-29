import numpy as np
import polars as pl
import torch
from torch.utils.data.dataset import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


class DyGMambaModelData(QuestionModelData):
    @override
    @QuestionModelData.disk_cache("dygmamba_data")
    def prepare_data(self, rc):
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None
        if fold_idx is None:
            raise ValueError("fold_idx must be specified for K-fold cross-validation")

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        if fold_idx < 0 or fold_idx >= kfold_n_splits:
            raise ValueError(f"fold_idx {fold_idx} out of range [0, {kfold_n_splits})")

        logger.info(f"K-fold: fold {fold_idx + 1}/{kfold_n_splits}")

        # 参数
        num_questions = self.data_src.get_metadata("num_questions")
        num_neighbor = rc.model.num_neighbor

        # 构建题目-技能矩阵，用于题目特征的构建
        q_table = self.build_relationship_matrix(("question", "has", "skill"))

        # 加载用户的交互数据，并将时间戳转为统一的 Unix 秒格式
        question_sequence = self.data_src.get_split_question_sequence_data()
        question_sequence = question_sequence.with_columns(
            (pl.col("timestamp").cast(pl.Int64) / 1000.0).alias("timestamp")
        )

        # 提取训练、验证和测试数据
        training_dataframe = question_sequence.filter(
            (pl.col("fold") != fold_idx) & (pl.col("fold") != -1)
        )
        validation_dataframe = question_sequence.filter(pl.col("fold") == fold_idx)
        test_dataframe = question_sequence.filter(pl.col("fold") == -1)

        # 构建数据集
        train_result = self.build_tensors(
            self.prepare_interactions(training_dataframe), num_neighbor, num_questions
        )
        train_dataset = DyGMambaDataset(*train_result, num_neighbor)

        val_result = self.build_tensors(
            self.prepare_interactions(validation_dataframe, training_dataframe),
            num_neighbor,
            num_questions,
        )
        val_dataset = DyGMambaDataset(*val_result, num_neighbor)

        test_result = self.build_tensors(
            self.prepare_interactions(
                test_dataframe, pl.concat([training_dataframe, validation_dataframe])
            ),
            num_neighbor,
            num_questions,
        )
        test_dataset = DyGMambaDataset(*test_result, num_neighbor)

        # 构建元数据
        num_users = self.data_src.get_metadata("num_split_question_users")

        metadata = {
            "num_questions": num_questions,
            "num_users": num_users,
            "question_id_offset": 1,
            "user_id_offset": num_questions + 1,
            "question_features": q_table,
            "num_neighbor": num_neighbor,
            "num_skills": q_table.shape[1],
        }

        return train_dataset, val_dataset, test_dataset, metadata

    def prepare_interactions(
        self,
        target_df: pl.DataFrame,
        context_df: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """合并历史交互与待预测交互，添加 ``is_target`` 标记列。

        Args:
            target_df: 待预测的交互（训练集时为全部交互）。
            context_df: 已观测的交互，仅为预测提供邻居信息
                （val/test 时传入训练数据）。为 None 时 target_df 中
                全部交互都作为预测目标。

        Returns:
            合并后的 DataFrame，包含 ``is_target`` 布尔列。
        """
        target_tagged = target_df.with_columns(pl.lit(True).alias("is_target"))
        if context_df is None:
            return target_tagged
        context_tagged = context_df.with_columns(pl.lit(False).alias("is_target"))
        return pl.concat([context_tagged, target_tagged])

    def build_question_neighbor(
        self,
        grouped_seqs,
        max_len,
        idx_to_user=None,
    ):
        """按分组构建因果邻居序列。

        每个交互的邻居 = 同分组中时间更早的所有交互。
        同一时间戳的交互共享追加前的 running state，保证自身不出现在自己的历史中。

        如果提供了 idx_to_user，则额外排除与当前交互同 user 的邻居，
        且历史不会被预先截断，而是完整保留后过滤再取 max_len。

        Args:
            grouped_seqs: 分组键 → [(global_idx, timestamp), ...] 的映射。
            max_len: 每个交互保留的最大邻居数量。
            idx_to_user: global_idx → user_id 的映射字典（可选）。

        Returns:
            列表，每项包含:
            - history_indices: 邻居的 global_idx 列表
            - history_times: 邻居的时间戳列表
            - current_time: 当前时间戳
            - interaction_indices: 当前时刻涉及的 global_idx 列表
        """
        states: list[dict] = []
        for seq_list in grouped_seqs.values():
            seq_sorted = sorted(seq_list, key=lambda x: x[1])
            running_indices: list[int] = []
            running_times: list[int] = []
            running_entries: list[
                tuple[int, int, int]
            ] = []  # (global_idx, timestamp, user)
            i = 0
            seq_len = len(seq_sorted)

            while i < seq_len:
                t = seq_sorted[i][1]
                j = i
                while j < seq_len and seq_sorted[j][1] == t:
                    j += 1

                current_indices = [seq_sorted[k][0] for k in range(i, j)]

                if idx_to_user is not None:
                    for global_idx in current_indices:
                        current_user = idx_to_user[global_idx]
                        hist_indices = []
                        hist_times = []
                        for entry in reversed(running_entries):
                            if entry[2] != current_user:
                                hist_indices.append(entry[0])
                                hist_times.append(entry[1])
                                if len(hist_indices) == max_len:
                                    break
                        hist_indices.reverse()
                        hist_times.reverse()
                        states.append(
                            {
                                "history_indices": hist_indices,
                                "history_times": hist_times,
                                "current_time": t,
                                "interaction_indices": [global_idx],
                            }
                        )
                else:
                    states.append(
                        {
                            "history_indices": list(running_indices),
                            "history_times": list(running_times),
                            "current_time": t,
                            "interaction_indices": current_indices,
                        }
                    )

                for k in range(i, j):
                    idx = seq_sorted[k][0]
                    running_indices.append(idx)
                    running_times.append(t)
                    if idx_to_user is not None:
                        running_entries.append((idx, t, idx_to_user[idx]))

                if (
                    idx_to_user is None
                    and max_len > 0
                    and len(running_indices) > max_len
                ):
                    running_indices = running_indices[-max_len:]
                    running_times = running_times[-max_len:]

                i = j

        return states

    def build_tensors(
        self,
        data_frame,
        num_neighbor,
        num_questions,
    ):
        total_interactions = len(data_frame)
        n_history = total_interactions - int(data_frame["is_target"].sum())

        # 统计信息
        # total_interactions 包含历史和目标交互的总数
        # n_history 是历史交互的数量
        # total_interactions - n_history 是待预测交互的数量
        logger.info(
            "Building data (total=%d, history=%d, target=%d)...",
            total_interactions,
            n_history,
            total_interactions - n_history,
        )

        # 排序并添加全局索引列
        all_df = data_frame.sort(["user", "seq_pos", "is_target"]).with_row_index(
            "global_idx", offset=1
        )

        idx_arr = all_df["global_idx"].to_numpy().astype(np.int32)
        user_arr = (num_questions + 1 + all_df["user"]).to_numpy().astype(np.int32)
        question_arr = (all_df["question"] + 1).to_numpy().astype(np.int32)
        time_arr = all_df["timestamp"].to_numpy().astype(np.int64)
        correctness_arr = all_df["label"].to_numpy().astype(np.int8)
        if "hint_count" in all_df.columns:
            hint_count_arr = all_df["hint_count"].to_numpy().astype(np.int32)
        else:
            hint_count_arr = np.zeros(total_interactions, dtype=np.int32)
        is_target_arr = all_df["is_target"].to_numpy()

        # 构建用户邻居序列
        logger.debug("Building user neighbor sequences...")
        user_his_padded = np.zeros((total_interactions, num_neighbor), dtype=np.int32)
        user_history: dict[int, list[int]] = {}

        for i in range(total_interactions):
            uid = int(user_arr[i])
            n = int(idx_arr[i])

            hist = user_history.get(uid, [])
            if hist:
                clip = hist[-num_neighbor:]
                user_his_padded[i, : len(clip)] = clip

            hist.append(n)
            if len(hist) > num_neighbor:
                hist = hist[-num_neighbor:]
            user_history[uid] = hist

        # 构建问题邻居序列
        idx_to_user = dict(
            zip(all_df["global_idx"].to_list(), all_df["user"].to_list())
        )
        logger.debug("Building question neighbor sequences...")
        grouped = all_df.group_by("question").agg(
            [
                pl.col("global_idx"),
                pl.col("timestamp"),
            ]
        )
        question_seqs = {q: list(zip(idx, ts)) for q, idx, ts in grouped.iter_rows()}

        question_neighbor_states = self.build_question_neighbor(
            question_seqs, num_neighbor, idx_to_user=idx_to_user
        )

        # 构建问题邻居padding数组
        max_idx = idx_arr.max()
        row_by_idx = np.zeros(max_idx + 1, dtype=np.int32)
        row_by_idx[idx_arr] = np.arange(total_interactions, dtype=np.int32)

        que_his_padded = np.zeros((total_interactions, num_neighbor), dtype=np.int32)
        for state in question_neighbor_states:
            hist = state["history_indices"][-num_neighbor:]
            if not hist:
                continue
            rows = row_by_idx[np.array(state["interaction_indices"])]
            que_his_padded[rows, : len(hist)] = hist

        # 计算邻居长度
        user_his_len = np.sum(user_his_padded != 0, axis=1).astype(np.int32)
        que_his_len = np.sum(que_his_padded != 0, axis=1).astype(np.int32)

        # 构建索引数组
        lookup_user = np.zeros(max_idx + 1, dtype=np.int32)
        lookup_question = np.zeros(max_idx + 1, dtype=np.int32)
        lookup_time = np.zeros(max_idx + 1, dtype=np.int64)
        lookup_correctness = np.zeros(max_idx + 1, dtype=np.int8)
        lookup_hint_count = np.zeros(max_idx + 1, dtype=np.int32)

        # 填充索引数组
        lookup_user[idx_arr] = user_arr
        lookup_question[idx_arr] = question_arr
        lookup_time[idx_arr] = time_arr
        lookup_correctness[idx_arr] = correctness_arr
        lookup_hint_count[idx_arr] = hint_count_arr

        # 确定预测的交互位置
        target_positions = np.where(is_target_arr)[0]

        # 将所有数据转换为 PyTorch 张量
        tensors = {
            "idx": torch.from_numpy(idx_arr),
            "user": torch.from_numpy(user_arr).long(),
            "question": torch.from_numpy(question_arr).long(),
            "time": torch.from_numpy(time_arr).float(),
            "correctness": torch.from_numpy(correctness_arr).float(),
            "user_his_idx": torch.from_numpy(user_his_padded).long(),
            "user_his_len": torch.from_numpy(user_his_len).long(),
            "que_his_idx": torch.from_numpy(que_his_padded).long(),
            "que_his_len": torch.from_numpy(que_his_len).long(),
            "lookup_user": torch.from_numpy(lookup_user).long(),
            "lookup_question": torch.from_numpy(lookup_question).long(),
            "lookup_time": torch.from_numpy(lookup_time).float(),
            "lookup_correctness": torch.from_numpy(lookup_correctness).float(),
            "lookup_hint_count": torch.from_numpy(lookup_hint_count).float(),
        }

        logger.debug(
            "DyGMamba dataset: total=%d, targets=%d",
            total_interactions,
            len(target_positions),
        )

        return tensors, target_positions


class DyGMambaDataset(Dataset):
    def __init__(
        self,
        tensors,
        target_indices,
        num_neighbor,
    ):
        self.tensors = tensors
        self.target_indices = target_indices
        self.num_neighbor = num_neighbor

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        return idx

    def get_batch(self, batch_indices):
        data_indices = self.target_indices[batch_indices]
        t = self.tensors

        user_his_idx = t["user_his_idx"][data_indices]
        que_his_idx = t["que_his_idx"][data_indices]

        return {
            "idx": t["idx"][data_indices],
            "user": t["user"][data_indices],
            "question": t["question"][data_indices],
            "time": t["time"][data_indices],
            "correctness": t["correctness"][data_indices],
            "src_neighbor_node_ids": t["lookup_question"][user_his_idx],
            "src_neighbor_times": t["lookup_time"][user_his_idx],
            "src_neighbor_edge_feats": t["lookup_correctness"][user_his_idx],
            "src_neighbor_len": t["user_his_len"][data_indices],
            "dst_neighbor_node_ids": t["lookup_user"][que_his_idx],
            "dst_neighbor_times": t["lookup_time"][que_his_idx],
            "dst_neighbor_edge_feats": t["lookup_correctness"][que_his_idx],
            "dst_neighbor_len": t["que_his_len"][data_indices],
            "src_neighbor_hint_count": t["lookup_hint_count"][user_his_idx],
            "dst_neighbor_hint_count": t["lookup_hint_count"][que_his_idx],
        }
