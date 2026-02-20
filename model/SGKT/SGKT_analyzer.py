"""SGKT Case Analyzer.

Provides inference-only capabilities for SGKT model case analysis.
"""

from typing import Any

import numpy as np
import torch

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import get_logger, register_analyzer
from utils.data_process import DataSource

from .SGKT_data import SGKTModelData
from .SGKT_model import SGKT

logger = get_logger(__name__)


@register_analyzer("SGKT")
class SGKTAnalyzer(BaseCaseAnalyzer):
    """SGKT-specific case analyzer for inference and visualization."""

    def __init__(self, args, data_src: DataSource, checkpoint_path: str):
        """Initialize SGKT analyzer.

        Args:
            args: Model arguments
            data_src: Data source instance
            checkpoint_path: Path to model checkpoint
        """
        self.args = args
        self.data_src = data_src

        model_data = SGKTModelData(data_src)
        (
            train_data,
            val_data,
            hrg_data,
            num_skills,
            num_questions,
            train_collate_fn,
            val_collate_fn,
        ) = model_data.prepare_data(args)

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")

        # Build question -> first skill lookup array: shape [num_questions]
        question_skill_matrix = model_data.build_relationship_matrix(
            ("question", "has", "skill"), value_type="binary"
        )
        first_skill = np.argmax(question_skill_matrix, axis=1)
        has_skill_per_q = question_skill_matrix.sum(axis=1) > 0
        self.question_to_skill = np.where(has_skill_per_q, first_skill, 0).astype(
            np.int64
        )

        model = SGKT(args=args, data_metadata=data_src.metadata)

        super().__init__(model, checkpoint_path)

        self.with_inference(
            data=val_data,
            batch_size=args.batch_size,
            device=args.device,
            collate_fn=val_collate_fn,
        ).build()

        # Move HRG context to device and bind feature embedding table
        self.hrg_data = {
            key: value.to(self.device_) if hasattr(value, "to") else value
            for key, value in hrg_data.items()
        }
        self.hrg_data["feature_embedding"] = self.model.feature_embedding.weight

    def forward_pass(
        self, batch_data: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """SGKT forward pass for inference.

        Args:
            batch_data: Dictionary with keys 'sequence', 'response', 'mask', 'hist_neighbor_index'

        Returns:
            Dictionary with y_hat (flattened), y_label (flattened), y_predict,
            full_y_hat (unflattened), and knowledge_states (for visualization)
        """
        # Unpack batch data
        sequence = batch_data["sequence"]
        response = batch_data["response"]
        mask = batch_data["mask"]
        hist_neighbor_index = batch_data.get("hist_neighbor_index")

        # Move to device
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Move hist_neighbor_index to device if provided
        if hist_neighbor_index is not None:
            hist_neighbor_index = self._move_tensor_to_device(hist_neighbor_index)

        # Model forward pass with return_states=True
        # Output at time t predicts label at time t+1
        # Model returns [B, S-1] (shifted predictions)
        y_hat_full, skill_embeddings, output_series = self.model(
            user_sequence=sequence,
            user_response=response,
            user_mask=mask,
            hrg_data=self.hrg_data,
            hist_neighbor_index=hist_neighbor_index,
            return_states=True,
        )

        # Extract valid predictions
        # Model already returns [B, S-1] (shifted predictions)
        # So we shift response and mask to match: [B, S] -> [B, S-1]
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full,
            response[:, 1:],  # Shift to match predictions
            mask[:, 1:],  # Shift to match predictions
            skip_first=False,
        )

        # Handle empty batch
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # Generate binary predictions
        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        # 计算知识状态：sigmoid(output_series @ skill_embeddings.T)
        # skill_embeddings: [num_skills, H], output_series: [B, S, H]
        # -> knowledge_states: [B, S, num_skills]，值域 [0, 1] 表示掌握程度
        knowledge_states = torch.sigmoid(
            torch.matmul(output_series, skill_embeddings.T)
        )

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
            batch_data: Raw batch data (dictionary with keys: sequence, response, mask, skills, hist_neighbor_index)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        sequences = batch_data["sequence"]
        masks = batch_data["mask"]

        batch_size, seq_len = sequences.shape

        y_label = outputs["y_label"]
        y_predict = outputs["y_predict"]
        y_hat = outputs["y_hat"]
        knowledge_states = outputs["knowledge_states"]  # [B, S, num_skills]

        # Create valid mask for shifted predictions
        # Model outputs [B, S-1], so we need to shift original mask
        masks_bool = masks.bool()
        if seq_len > 1:
            masks_bool = masks_bool[:, 1:]  # Shift to match predictions

        valid_indices = masks_bool.reshape(-1).nonzero(as_tuple=True)[0]

        question_ids_flat = sequences[:, 1:].reshape(-1)[valid_indices].cpu().numpy()

        # Generate dummy user IDs (batch indices) since SGKT doesn't track users
        user_ids_flat = (
            torch.arange(batch_size)
            .reshape(-1, 1)
            .expand(-1, seq_len - 1)
            .reshape(-1)[valid_indices]
            .cpu()
            .numpy()
        )

        # 将知识状态展平：[B, S, num_skills] -> [B*S, num_skills]，取有效位置
        num_skills = knowledge_states.shape[-1]
        knowledge_states_flat = (
            knowledge_states.reshape(-1, num_skills)[valid_indices].cpu().numpy()
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
            "masks": masks_bool.reshape(-1)[valid_indices].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }
