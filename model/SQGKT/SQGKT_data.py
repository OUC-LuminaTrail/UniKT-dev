from collections import defaultdict
from functools import partial
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


def sample_hist_neighbors(
    batch_size, max_seq_len, hist_neighbor_num, skill_index, pad_index=None
):
    """同技能历史采样：位置 t 在历史 [0, t-1] 中随机选 M 个技能相同的位置。"""
    if pad_index is None:
        pad_index = max_seq_len
    if hist_neighbor_num == 0:
        return np.zeros((batch_size, max_seq_len, 0), dtype=np.int64)

    skills = (
        skill_index.numpy()
        if isinstance(skill_index, torch.Tensor)
        else np.asarray(skill_index)
    )
    result = np.full(
        (batch_size, max_seq_len, hist_neighbor_num), pad_index, dtype=np.int64
    )
    result[:, 0, :] = pad_index

    for b in range(batch_size):
        seq_skills = skills[b]
        same_skill = seq_skills[np.newaxis, :] == seq_skills[:, np.newaxis]
        causal = np.tril(np.ones((max_seq_len, max_seq_len), dtype=bool), k=-1)
        valid = same_skill & causal
        for t in range(1, max_seq_len):
            candidates = np.where(valid[t])[0]
            if len(candidates) >= hist_neighbor_num:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=False
                )
            elif len(candidates) > 0:
                result[b, t] = np.random.choice(
                    candidates, hist_neighbor_num, replace=True
                )
    return result


class SQGKTModelData(QuestionModelData):
    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, rc: Any):
        user_sequence, user_response, user_mask, user_id_sequence = (
            self.load_sequence_data()
        )

        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")
        # load_sequence_data returns split sequences (windows) as rows; user id is the row index.
        num_users = user_sequence.shape[0]

        logger.info("Building question-skill neighbors...")
        question_neighbors, skill_neighbors = self.build_qs_neighbors(
            question_skill_matrix,
            num_skills,
            num_questions,
            rc.model.question_neighbor_num,
            rc.model.skill_neighbor_num,
        )

        logger.info("Building student-question graph...")
        q_neighbors_2, uq_stat_q = self.build_sq_graph(
            num_questions, rc.model.user_neighbor_num, rc.data.fold
        )

        train_data, val_data, test_data = self.split_kfold_data(
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            fold_idx=rc.data.fold,
        )
        train_seq, train_res, train_mask, train_uid = train_data
        val_seq, val_res, val_mask, val_uid = val_data
        test_seq, test_res, test_mask, test_uid = test_data

        train_skills = self._extract_skills(train_seq, question_skill_matrix)
        val_skills = self._extract_skills(val_seq, question_skill_matrix)
        test_skills = self._extract_skills(test_seq, question_skill_matrix)

        train_dataset = SQGKTDataset(
            train_seq, train_res, train_mask, train_uid, train_skills
        )
        val_dataset = SQGKTDataset(val_seq, val_res, val_mask, val_uid, val_skills)
        test_dataset = SQGKTDataset(
            test_seq, test_res, test_mask, test_uid, test_skills
        )

        collate = partial(
            sqgkt_collate_fn, hist_neighbor_num=rc.model.hist_neighbor_num
        )

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        graph_data = {
            "question_neighbors": torch.from_numpy(question_neighbors).long(),
            "skill_neighbors": torch.from_numpy(skill_neighbors).long(),
            "q_neighbors_2": torch.from_numpy(q_neighbors_2).long(),
            "uq_stat_q": torch.from_numpy(uq_stat_q).float(),
            "next_neighbor_num": rc.model.next_neighbor_num,
            "feature_embedding": None,
        }
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            graph_data,
            num_skills,
            num_questions,
            num_users,
            collate,
            collate,
        )

    @staticmethod
    def build_qs_neighbors(question_skill_matrix, num_skills, num_questions, qn, sn):
        """问题-技能二部图邻居表（偏移节点 id，对应 GIKTGraphAggregator）。

        节点 id 布局：技能 [0, num_skills)，题目 [num_skills, num_skills+num_questions)。
        question_neighbors 以组合 id 为索引（题目行偏移 +num_skills），skill_neighbors 以技能 id 索引。
        """
        qs_num = num_skills + num_questions
        question_to_skills = [
            np.where(question_skill_matrix[q] == 1)[0] for q in range(num_questions)
        ]
        skill_to_questions = [
            np.where(question_skill_matrix[:, s] == 1)[0] + num_skills
            for s in range(num_skills)
        ]

        question_neighbors = np.zeros([qs_num, qn], dtype=np.int32)
        skill_neighbors = np.zeros([num_skills, sn], dtype=np.int32)

        for q_id, neighbors in enumerate(question_to_skills):
            if len(neighbors) == 0:
                continue
            question_neighbors[num_skills + q_id] = np.random.choice(
                neighbors, qn, replace=len(neighbors) < qn
            )
        for s_id, neighbors in enumerate(skill_to_questions):
            if len(neighbors) == 0:
                continue
            skill_neighbors[s_id] = np.random.choice(
                neighbors, sn, replace=len(neighbors) < sn
            )
        return question_neighbors, skill_neighbors

    def build_sq_graph(self, num_questions, k, fold_idx):
        """学生-问题图（论文 §4.1–4.2）。

        对每个问题 q_j 采样 k 个答过它的学生，并按论文式 (6) 计算边权 g_ij 的三个分量：
          c_i   : 学生 i 的整体作答正确率（学习能力，式 1）
          g^p   : 基于 attempt_count（泊松）的知识获取因子（式 2–3）
          g^n   : 基于 hint_count（泊松）的知识获取因子（式 4–5）
        三分量按位存储（g_ij = w_c·c + w_p·g^p + w_n·g^n 中的可学习权重在模型中）。
        统计量仅基于训练折（fold != fold_idx 且 fold != -1）。
        返回 q_neighbors_2[num_questions, k]（采样学生 id）与 uq_stat_q[num_questions, k, 3]。
        """
        from scipy.stats import poisson

        data = self.data_src.get_split_question_sequence_data()
        data = data.filter(
            (pl.col("fold") != fold_idx) & (pl.col("fold") != -1)
        ).to_pandas()
        eta, alpha, beta = 10.0, 0.3, 0.7

        # Per-student overall accuracy c_i (Eq.1)
        stu_total = data.groupby("user")["label"].size()
        stu_correct = data.groupby("user")["label"].sum()
        c_i = (stu_correct / stu_total.clip(lower=1)).to_dict()

        # Per-question Poisson λ for attempt/hint counts (MLE = mean)
        lam_p = data.groupby("question")["attempt_count"].mean().to_dict()
        lam_n = data.groupby("question")["hint_count"].mean().to_dict()

        # Cumulative attempt/hint per (student, question)
        pq = data.groupby(["user", "question"])["attempt_count"].sum().to_dict()
        nq = data.groupby(["user", "question"])["hint_count"].sum().to_dict()
        # Students who answered each question
        q_to_students = defaultdict(list)
        for u, q in pq:
            q_to_students[q].append(u)

        def factor(count, lam):
            pc = 0.0 if lam <= 0 else 1.0 - poisson.sf(int(count) - 1, lam)
            return alpha + (1.0 - alpha) / (1.0 + np.exp(eta * (pc - beta)))

        q_neighbors_2 = np.zeros([num_questions, k], dtype=np.int32)
        uq_stat_q = np.zeros([num_questions, k, 3], dtype=np.float32)
        for q in range(num_questions):
            students = q_to_students.get(q, [])
            if not students:
                continue
            n = len(students)
            idx = np.random.choice(n, k, replace=(n < k))
            sampled = np.array(students, dtype=np.int32)[idx]
            q_neighbors_2[q] = sampled
            for slot, u in enumerate(sampled):
                uq_stat_q[q, slot, 0] = c_i.get(int(u), 0.0)
                uq_stat_q[q, slot, 1] = factor(
                    pq.get((int(u), q), 0), lam_p.get(q, 0.0)
                )
                uq_stat_q[q, slot, 2] = factor(
                    nq.get((int(u), q), 0), lam_n.get(q, 0.0)
                )
        return q_neighbors_2, uq_stat_q

    @staticmethod
    def _extract_skills(user_sequence, question_skill_matrix):
        first_skill = np.argmax(question_skill_matrix, axis=1)
        has_skill = np.any(question_skill_matrix == 1, axis=1)
        question_to_skill = np.where(has_skill, first_skill, 0).astype(
            user_sequence.dtype, copy=False
        )
        return question_to_skill[user_sequence]


class SQGKTDataset(Dataset):
    def __init__(self, user_sequence, user_response, user_mask, user_id, user_skills):
        self.user_sequence = torch.from_numpy(np.asarray(user_sequence)).long()
        self.user_response = torch.from_numpy(np.asarray(user_response)).long()
        self.user_mask = torch.from_numpy(np.asarray(user_mask)).bool()
        self.user_id = torch.from_numpy(np.asarray(user_id)[:, 0]).long()
        self.user_skills = torch.from_numpy(np.asarray(user_skills)).long()

    def __len__(self):
        return len(self.user_sequence)

    def __getitem__(self, idx):
        return {
            "sequence": self.user_sequence[idx],
            "response": self.user_response[idx],
            "mask": self.user_mask[idx],
            "user_id": self.user_id[idx],
            "skills": self.user_skills[idx],
        }


def sqgkt_collate_fn(batch, hist_neighbor_num=3):
    from torch.utils.data.dataloader import default_collate

    batched = default_collate(batch)
    batch_size, full_seq_len = batched["skills"].shape
    model_seq_len = full_seq_len - 1
    batched["hist_neighbor_index"] = torch.from_numpy(
        sample_hist_neighbors(
            batch_size,
            model_seq_len,
            hist_neighbor_num,
            batched["skills"][:, :model_seq_len],
            pad_index=model_seq_len,
        )
    ).long()
    return batched
