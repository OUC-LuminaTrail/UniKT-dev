"""
SGKT data preparation module.

Implements data preparation for Session Graph-based Knowledge Tracing model.
Builds the Heterogeneous Relation Graph (HRG) following the original author's strategy.
"""

from torch.utils.data import Dataset, DataLoader
from utils.net_data import GraphModelData
from utils.core import get_logger
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data


def sample_hist_neighbors(batch_size, max_seq_len, hist_neighbor_num, skill_index):
    """
    Sample historical neighbors based on skill matching.

    Following the original TF implementation (data_process.py:198-220).
    For each position t, find historical positions [0, t-1] with the same skill,
    then randomly select M positions as neighbors.

    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        hist_neighbor_num: Number of historical neighbors to sample (M)
        skill_index: [batch_size, max_seq_len] Skill indices for each position

    Returns:
        hist_neighbor_index: [batch_size, max_seq_len, hist_neighbor_num]
                            Pre-computed historical neighbor indices
    """
    hist_neighbors_index = []

    for i in range(batch_size):
        seq_hist_index = []
        seq_skill_index = skill_index[i]

        for t in range(1, max_seq_len):
            # Find historical positions with the same skill
            # same_skill_index = [k for k in range(t) if seq_skill_index[k] == seq_skill_index[t]]
            current_skill = seq_skill_index[t].item()
            same_skill_indices = [
                k for k in range(t) if seq_skill_index[k].item() == current_skill
            ]

            if hist_neighbor_num != 0:
                if len(same_skill_indices) >= hist_neighbor_num:
                    # Enough same-skill positions, sample without replacement
                    selected = np.random.choice(
                        same_skill_indices, hist_neighbor_num, replace=False
                    )
                    seq_hist_index.append(selected.tolist())
                elif len(same_skill_indices) > 0:
                    # Not enough, sample with replacement
                    selected = np.random.choice(
                        same_skill_indices, hist_neighbor_num, replace=True
                    )
                    seq_hist_index.append(selected.tolist())
                else:
                    # No same-skill position found, use fallback indices
                    # Use [max_seq_len-1, max_seq_len-1, ...] as fallback
                    seq_hist_index.append([max_seq_len - 1] * hist_neighbor_num)
            else:
                seq_hist_index.append([])

        # Pad first position with zeros (no history for t=0)
        seq_hist_index = [[0] * hist_neighbor_num] + seq_hist_index
        hist_neighbors_index.append(seq_hist_index)

    return np.array(hist_neighbors_index, dtype=np.int64)


class SGKTModelData(GraphModelData):
    """
    SGKT data preparation class.

    Builds:
    1. User sequence data
    2. Question-skill relationship matrix
    3. HRG (Heterogeneous Relation Graph) with GCNConv-compatible format
    """

    def __init__(self, data_src):
        super().__init__(data_src)
        self.logger = get_logger(__name__)

    def prepare_data(self, args):
        """
        Prepare data for SGKT model.

        Args:
            args: Arguments containing model hyperparameters

        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
            hrg_data: PyG Data object for HRG graph
            question_skill_matrix: Question-skill relationship matrix
            num_skills: Number of skills
            num_questions: Number of questions
        """
        max_seq_len = args.max_seq_len
        min_seq_len = args.min_seq_len

        # 1. Build sequence data (reuse base class method)
        user_sequence, user_response, user_mask, user_id_sequence = (
            self.build_sequence_data(max_seq_len=max_seq_len, min_seq_len=min_seq_len)
        )

        # 2. Build question-skill relationship matrix
        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )

        # Get metadata
        num_questions = self.data_src.get_metadata("num_questions")
        num_skills = self.data_src.get_metadata("num_skills")

        # 3. Build HRG graph (Question-Skill Graph)
        # Following the original author's sampling strategy
        hrg_data = self.build_hrg_graph(
            question_skill_matrix=question_skill_matrix,
            num_skills=num_skills,
            num_questions=num_questions,
            cooc_neighbor_num=getattr(args, "cooc_neighbor_num", 50),
        )

        self.logger.info(
            f"HRG graph built: {hrg_data.num_nodes} nodes, {hrg_data.edge_index.shape[1]} edges"
        )

        # 4. Split data into train/val or k-fold
        if hasattr(args, "fold") and args.fold is not None:
            # K-fold cross validation
            train_data, val_data = self.split_kfold_data(
                user_sequence,
                user_response,
                user_mask,
                user_id_sequence,
                fold_idx=args.fold,
            )
        else:
            # Simple train/val split
            train_data, val_data = self.split_data(
                user_sequence, user_response, user_mask, user_id_sequence, val_ratio=0.2
            )

        # Unpack train/val data
        train_sequence, train_response, train_mask, _ = train_data
        val_sequence, val_response, val_mask, _ = val_data

        # 5. Get skill data for hist_neighbor_index computation
        # Extract skills from sequence data
        train_skills = self._extract_skills(train_sequence)
        val_skills = self._extract_skills(val_sequence)

        # 6. Create datasets and data loaders
        hist_neighbor_num = getattr(args, "hist_neighbor_num", 5)

        train_dataset = SGKTDataset(
            train_sequence, train_response, train_mask, train_skills, hist_neighbor_num
        )

        val_dataset = SGKTDataset(
            val_sequence, val_response, val_mask, val_skills, hist_neighbor_num
        )

        # Create custom collate function with hist_neighbor_num
        from functools import partial

        train_collate_fn = partial(sgkt_collate_fn, hist_neighbor_num=hist_neighbor_num)
        val_collate_fn = partial(sgkt_collate_fn, hist_neighbor_num=hist_neighbor_num)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=val_collate_fn,
        )

        self.logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
        )

        return (
            train_loader,
            val_loader,
            hrg_data,
            num_skills,
            num_questions,
        )

    def build_hrg_graph(
        self,
        question_skill_matrix,
        num_skills,
        num_questions,
        cooc_neighbor_num: int = 50,
    ):
        """
        Build Heterogeneous Relation Graph (HRG) following original author's strategy.

        The HRG graph contains three types of relations:
        1. Question -> Skill (via skill_matrix)
        2. Skill -> Question (reverse)
        3. Question <-> Question (co-occurrence in same sequence)

        Args:
            question_skill_matrix: [num_questions, num_skills] binary matrix
            num_skills: Number of skills
            num_questions: Number of questions

        Returns:
            hrg_data: PyG Data object with unified node indexing
                - Node indexing: [skill_0, ..., skill_S, question_0, ..., question_Q]
                - Total nodes = num_skills + num_questions
        """
        edge_index = [[], []]

        if cooc_neighbor_num is None:
            cooc_neighbor_num = 0
        cooc_neighbor_num = int(cooc_neighbor_num)

        # Relation 1: Question -> Skill (via skill_matrix)
        # Following original author's logic in build_adj_list (line 60)
        for q in range(num_questions):
            skills = np.where(question_skill_matrix[q] == 1)[0].tolist()
            for s in skills:
                edge_index[0].append(num_skills + q)  # Question node ID
                edge_index[1].append(s)  # Skill node ID

        # Relation 2: Skill -> Question (reverse edges)
        # Following original author's logic (line 66-67)
        for s in range(num_skills):
            questions = np.where(question_skill_matrix[:, s] == 1)[0].tolist()
            for q in questions:
                edge_index[0].append(s)  # Skill node ID
                edge_index[1].append(num_skills + q)  # Question node ID

        # Relation 3: Question co-occurrence (questions in same sequence)
        # Following original author's logic (line 68-71)
        # Extract co-occurrence from training data
        data = self.data_src.get_sequence_data()

        # IMPORTANT: naive co-occurrence makes a fully-connected graph per user
        # (O(|Q_u|^2) edges). For large datasets this explodes GPU memory in GCNConv.
        # To match the original author's *bounded neighbor* behavior, we cap the number
        # of co-occurrence neighbors per question.
        if cooc_neighbor_num > 0:
            # Build question -> co-occurrence neighbor set (question ids are 0-based)
            cooc_neighbors: dict[int, set[int]] = {}

            # Group once for performance
            user_to_questions = (
                data.groupby("user")["question"].unique()
            )
            for _user_id, questions_arr in tqdm(
                user_to_questions.items(),
                total=len(user_to_questions),
                desc="Building question co-occurrence edges (capped)",
            ):
                # Deterministic order to keep runs stable
                questions_in_seq = sorted(set(map(int, questions_arr)))
                if len(questions_in_seq) <= 1:
                    continue

                for idx, q1 in enumerate(questions_in_seq):
                    # candidates are all other questions in the same user sequence
                    candidates = questions_in_seq[:idx] + questions_in_seq[idx + 1 :]
                    if len(candidates) > cooc_neighbor_num:
                        candidates = candidates[:cooc_neighbor_num]
                    if not candidates:
                        continue
                    cooc_neighbors.setdefault(q1, set()).update(candidates)

            # Enforce the global cap per question (across all users)
            for q1, neigh_set in cooc_neighbors.items():
                neigh_list = sorted(neigh_set)
                if len(neigh_list) > cooc_neighbor_num:
                    neigh_list = neigh_list[:cooc_neighbor_num]
                src = num_skills + q1
                for q2 in neigh_list:
                    dst = num_skills + q2
                    edge_index[0].append(src)
                    edge_index[1].append(dst)

        # Convert to tensor and remove duplicates
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)

        # Build PyG Data object
        hrg_data = Data(edge_index=edge_index, num_nodes=num_skills + num_questions)

        self.logger.info(
            f"HRG Graph: {num_skills} skills, {num_questions} questions, "
            f"{edge_index.shape[1]} edges"
        )

        return hrg_data

    def _extract_skills(self, user_sequence):
        """
        Extract skill IDs from question IDs using question-skill matrix.

        Args:
            user_sequence: [num_users, max_seq_len] Question IDs

        Returns:
            user_skills: [num_users, max_seq_len] Skill IDs
        """
        question_skill_matrix = self.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )

        num_users, max_seq_len = user_sequence.shape
        user_skills = np.zeros_like(user_sequence)

        for user_id in range(num_users):
            for t in range(max_seq_len):
                question_id = user_sequence[user_id, t]
                # Find skill for this question (assuming single skill per question)
                # Get the skill ID where the matrix has value 1
                skills = np.where(question_skill_matrix[question_id] == 1)[0]
                if len(skills) > 0:
                    user_skills[user_id, t] = skills[0]  # Use first skill

        return user_skills


class SGKTDataset(Dataset):
    """
    Dataset for SGKT model.

    Args:
        user_sequence: [num_users, max_seq_len] Question IDs
        user_response: [num_users, max_seq_len] Responses (0/1)
        user_mask: [num_users, max_seq_len] Valid position mask
        user_skills: [num_users, max_seq_len] Skill IDs
        hist_neighbor_num: Number of historical neighbors (M)
    """

    def __init__(
        self, user_sequence, user_response, user_mask, user_skills, hist_neighbor_num
    ):
        self.user_sequence = torch.from_numpy(user_sequence).long()
        self.user_response = torch.from_numpy(user_response).long()
        self.user_mask = torch.from_numpy(user_mask).long()
        self.user_skills = torch.from_numpy(user_skills).long()
        self.hist_neighbor_num = hist_neighbor_num
        self.max_seq_len = user_sequence.shape[1]

        assert len(user_sequence) == len(user_response) == len(user_mask) == len(
            user_skills
        ), "Sequence, response, mask, and skills must have the same length"

    def __len__(self):
        return len(self.user_sequence)

    def __getitem__(self, idx):
        return {
            "sequence": self.user_sequence[idx],
            "response": self.user_response[idx],
            "mask": self.user_mask[idx],
            "skills": self.user_skills[idx],
        }


def sgkt_collate_fn(batch, hist_neighbor_num=5):
    """
    Custom collate function for SGKT that computes hist_neighbor_index per batch.

    Args:
        batch: List of dicts with keys: sequence, response, mask, skills
        hist_neighbor_num: Number of historical neighbors (M)

    Returns:
        Batched dict with hist_neighbor_index added
    """
    from torch.utils.data.dataloader import default_collate

    # First, batch the regular tensors
    batched = default_collate(batch)

    # Compute hist_neighbor_index for this batch
    batch_size = batched["skills"].shape[0]
    max_seq_len = batched["skills"].shape[1]

    # Compute hist_neighbor_index using skill-based matching
    hist_neighbor_index = sample_hist_neighbors(
        batch_size, max_seq_len, hist_neighbor_num, batched["skills"]
    )

    batched["hist_neighbor_index"] = torch.from_numpy(hist_neighbor_index).long()

    return batched
