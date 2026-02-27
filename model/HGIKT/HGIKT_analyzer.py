"""HGIKT Case Analyzer.

Provides inference-only capabilities for HGIKT model case analysis.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import get_logger, register_analyzer
from utils.data_process import DataSource

from .HGIKT_data import HGIKTModelData
from .HGIKT_model import HGIKT

logger = get_logger(__name__)


@register_analyzer("HGIKT")
class HGIKTAnalyzer(BaseCaseAnalyzer):
    """HGIKT-specific case analyzer for inference and visualization."""

    def __init__(self, args, data_src: DataSource, checkpoint_path: str):
        """Initialize HGIKT analyzer.

        Args:
            args: Model arguments
            data_src: Data source instance
            checkpoint_path: Path to model checkpoint
        """
        self.args = args
        self.data_src = data_src

        model_data = HGIKTModelData(data_src)
        data_dict = model_data.prepare_data(args)

        # Unpack data
        val_dataset = data_dict["val_dataset"]
        hypergraph = data_dict["skill_hypergraph"]
        hetero_graph = data_dict["hetero_graph"]
        question_skill_matrix = data_dict["question_skill_matrix"]

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")

        # Store full question-skill matrix for multi-skill extraction
        self.question_skill_matrix_np = (
            question_skill_matrix.numpy()
        )  # [num_questions, num_skills]

        model = HGIKT(args, data_src.get_metadata(), hetero_graph.metadata())

        super().__init__(model, checkpoint_path)

        self.with_inference(val_dataset, args.batch_size, args.device)
        self.build()

        # Store graph data
        self.hetero_graph = hetero_graph
        self.hypergraph = hypergraph
        self.question_skill_matrix = question_skill_matrix

        # Move graphs to device
        self.hetero_graph = self.hetero_graph.to(self.device_)
        self.hypergraph = self.hypergraph.to(self.device_)
        self.question_skill_matrix = self.question_skill_matrix.to(self.device_)

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """HGIKT forward pass for inference.

        Args:
            batch_data: Tuple of (users, sequence, response, mask)

        Returns:
            Dictionary with y_hat (flattened), y_label (flattened), y_predict,
            full_y_hat (unflattened), and knowledge_states (for visualization)
        """
        users, sequence, response, mask = batch_data
        users = self._move_tensor_to_device(users)
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Model forward pass with return_states=True
        y_hat_full, skill_hetero_conv, lstm_output = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            self.question_skill_matrix,
            return_states=True,
        )

        # Extract valid predictions (skip first position)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

        # Handle empty batch
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # Generate binary predictions
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        # 计算知识状态
        knowledge_states = F.cosine_similarity(
            lstm_output.unsqueeze(2),  # [B, S, 1, H]
            skill_hetero_conv.unsqueeze(0).unsqueeze(0),  # [1, 1, num_skills, H]
            dim=-1,
        )  # [B, S, num_skills]
        knowledge_states = (knowledge_states + 1) / 2  # 将[-1, 1]映射到[0, 1]

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_hat_full": y_hat_full,
            "knowledge_states": knowledge_states,
        }

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract case data from batch outputs.

        Args:
            batch_data: Raw batch data (users, sequences, responses, masks tuple)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        users, sequences, responses, masks = batch_data

        batch_size, seq_len = sequences.shape

        y_label = outputs["y_label"]
        y_predict = outputs["y_predict"]
        y_hat = outputs["y_hat"]
        knowledge_states = outputs["knowledge_states"]  # [B, S, num_skills]

        masks_bool = masks.bool()
        if seq_len > 1:
            masks_bool[:, 0] = False

        valid_indices = masks_bool.view(-1).nonzero(as_tuple=True)[0]

        question_ids_flat = sequences.view(-1)[valid_indices].cpu().numpy()
        user_ids_flat = users.view(-1)[valid_indices].cpu().numpy()

        # 将知识状态展平：[B, S, num_skills] -> [B*S, num_skills]，取有效位置
        num_skills = knowledge_states.shape[-1]
        knowledge_states_flat = (
            knowledge_states.view(-1, num_skills)[valid_indices].cpu().numpy()
        )

        # Get all skills for each question (returns list of lists)
        skill_ids_list = self._get_all_skills_for_questions(question_ids_flat)

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "skills": skill_ids_list,
            "labels": y_label.cpu().numpy(),
            "predictions": y_predict.cpu().numpy(),
            "logits": y_hat.cpu().numpy(),
            "masks": masks_bool.view(-1)[valid_indices].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }

    def _get_all_skills_for_questions(
        self, question_ids: np.ndarray
    ) -> list[list[int]]:
        """Get all skills for each question, returning as list of lists.

        Args:
            question_ids: Array of question IDs

        Returns:
            List of lists, where each inner list contains all skill IDs for that question.
            Returns [0] if no skills found.
        """
        skills_list = []
        for q_id in question_ids:
            # Find all skills for this question (where matrix value is 1)
            question_skills = np.where(self.question_skill_matrix_np[q_id] == 1)[
                0
            ].tolist()
            skills_list.append(question_skills if question_skills else [0])
        return skills_list
