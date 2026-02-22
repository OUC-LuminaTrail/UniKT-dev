"""HGIKT Case Analyzer.

Provides inference-only capabilities for HGIKT model case analysis.
"""

from typing import Any

import numpy as np
import torch

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

        # Build question -> first skill lookup array: shape [num_questions]
        qs_matrix = question_skill_matrix.numpy()  # [num_questions, num_skills]
        first_skill = np.argmax(qs_matrix, axis=1)
        has_skill_per_q = qs_matrix.sum(axis=1) > 0
        self.question_to_skill = np.where(has_skill_per_q, first_skill, 0).astype(
            np.int64
        )

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

        # 计算知识状态：sigmoid(lstm_output @ skill_hetero_conv.T)
        # skill_hetero_conv: [num_skills, H], lstm_output: [B, S, H]
        # -> knowledge_states: [B, S, num_skills]，值域 [0, 1] 表示掌握程度
        knowledge_states = torch.sigmoid(torch.matmul(lstm_output, skill_hetero_conv.T))

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

        # 根据 question_id 查询对应的 skill_id（取第一个关联技能）
        skill_ids_flat = self.question_to_skill[question_ids_flat]

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "skills": skill_ids_flat,
            "labels": y_label.cpu().numpy(),
            "predictions": y_predict.cpu().numpy(),
            "logits": y_hat.cpu().numpy(),
            "masks": masks_bool.view(-1)[valid_indices].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }
