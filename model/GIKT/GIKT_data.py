"""GIKT 数据准备：构建问题-技能二部图邻居表（采样）与同技能历史邻居索引。"""

from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.core import get_logger
from utils.model_data import QuestionModelData

logger = get_logger(__name__)


def sample_hist_neighbors(
    batch_size, max_seq_len, hist_neighbor_num, skill_index, pad_index=None
):
    """同技能历史邻居采样：对位置 t，在历史 [0, t-1] 中找技能相同的位置随机选 M 个。"""
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
    result[:, 0, :] = pad_index  # 位置 0 无历史

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


class GIKTModelData(QuestionModelData):
    def __init__(self, data_src):
        super().__init__(data_src)

    def prepare_data(self, args):
        user_sequence, user_response, user_mask, user_id_sequence = (
            self.load_sequence_data()
        )

        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        logger.info("Building question-skill neighbors (bipartite, sampled)")
        question_neighbors, skill_neighbors = self.build_qs_neighbors(
            question_skill_matrix,
            num_skills,
            num_questions,
            args.question_neighbor_num,
            args.skill_neighbor_num,
        )

        train_data, val_data, test_data = self.split_kfold_data(
            user_sequence,
            user_response,
            user_mask,
            user_id_sequence,
            fold_idx=args.fold,
        )
        train_sequence, train_response, train_mask, _ = train_data
        val_sequence, val_response, val_mask, _ = val_data
        test_sequence, test_response, test_mask, _ = test_data

        # 每位置首个技能 id（用于同技能历史采样与 sim_emb=skill_emb）
        train_skills = self._extract_skills(train_sequence)
        val_skills = self._extract_skills(val_sequence)
        test_skills = self._extract_skills(test_sequence)

        train_dataset = GIKTDataset(
            train_sequence, train_response, train_mask, train_skills
        )
        val_dataset = GIKTDataset(val_sequence, val_response, val_mask, val_skills)
        test_dataset = GIKTDataset(test_sequence, test_response, test_mask, test_skills)

        train_collate_fn = partial(
            gikt_collate_fn, hist_neighbor_num=args.hist_neighbor_num
        )
        val_collate_fn = partial(
            gikt_collate_fn, hist_neighbor_num=args.hist_neighbor_num
        )

        logger.info(
            f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
        )

        graph_data = {
            "question_neighbors": torch.from_numpy(question_neighbors).long(),
            "skill_neighbors": torch.from_numpy(skill_neighbors).long(),
            "next_neighbor_num": args.next_neighbor_num,
            "feature_embedding": None,  # 训练器中绑定为 model.feature_embedding.weight
        }
        return (
            train_dataset,
            val_dataset,
            test_dataset,
            graph_data,
            num_skills,
            num_questions,
            train_collate_fn,
            val_collate_fn,
        )

    def build_qs_neighbors(
        self,
        question_skill_matrix,
        num_skills,
        num_questions,
        question_neighbor_num,
        skill_neighbor_num,
    ):
        """构建问题-技能二部图邻居表并固定采样（对应原 extract_qs_relations）。

        节点 id 布局：技能 [0, num_skills)，题目 [num_skills, num_skills+num_questions)。
        """
        qs_num = num_skills + num_questions
        question_to_skills = [
            np.where(question_skill_matrix[q] == 1)[0] for q in range(num_questions)
        ]
        skill_to_questions = [
            np.where(question_skill_matrix[:, s] == 1)[0] + num_skills
            for s in range(num_skills)
        ]

        question_neighbors = np.zeros([qs_num, question_neighbor_num], dtype=np.int32)
        skill_neighbors = np.zeros([num_skills, skill_neighbor_num], dtype=np.int32)

        for q_id, neighbors in enumerate(question_to_skills):
            if len(neighbors) == 0:
                continue
            replace = len(neighbors) < question_neighbor_num
            question_neighbors[num_skills + q_id] = np.random.choice(
                neighbors, question_neighbor_num, replace=replace
            )
        for s_id, neighbors in enumerate(skill_to_questions):
            if len(neighbors) == 0:
                continue
            replace = len(neighbors) < skill_neighbor_num
            skill_neighbors[s_id] = np.random.choice(
                neighbors, skill_neighbor_num, replace=replace
            )
        return question_neighbors, skill_neighbors

    def _extract_skills(self, user_sequence):
        """题目 id -> 首个技能 id。"""
        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        first_skill = np.argmax(question_skill_matrix, axis=1)
        has_skill = np.any(question_skill_matrix == 1, axis=1)
        question_to_skill = np.where(has_skill, first_skill, 0).astype(
            user_sequence.dtype, copy=False
        )
        return question_to_skill[user_sequence]


class GIKTDataset(Dataset):
    def __init__(self, user_sequence, user_response, user_mask, user_skills):
        self.user_sequence = torch.from_numpy(np.asarray(user_sequence)).long()
        self.user_response = torch.from_numpy(np.asarray(user_response)).long()
        self.user_mask = torch.from_numpy(np.asarray(user_mask)).bool()
        self.user_skills = torch.from_numpy(np.asarray(user_skills)).long()

    def __len__(self):
        return len(self.user_sequence)

    def __getitem__(self, idx):
        return {
            "sequence": self.user_sequence[idx],
            "response": self.user_response[idx],
            "mask": self.user_mask[idx],
            "skills": self.user_skills[idx],
        }


def gikt_collate_fn(batch, hist_neighbor_num=3):
    """批拼接并为每个 batch 计算 hist_neighbor_index。"""
    from torch.utils.data.dataloader import default_collate

    batched = default_collate(batch)
    batch_size, full_seq_len = batched["skills"].shape
    model_seq_len = full_seq_len - 1  # GIKT 在 max_step = full_seq_len - 1 上运行
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
