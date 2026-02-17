"""GIKT Case Analyzer.

Provides inference-only capabilities for GIKT model case analysis.
"""

from typing import Any

import torch

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import get_logger, register_analyzer
from utils.data_process import DataSource

from .GIKT_data import GIKTModelData
from .GIKT_model import GIKT

logger = get_logger(__name__)


@register_analyzer("GIKT")
class GIKTAnalyzer(BaseCaseAnalyzer):
    """GIKT-specific case analyzer for inference and visualization."""

    def __init__(self, args, data_src: DataSource, checkpoint_path: str):
        """Initialize GIKT analyzer.

        Args:
            args: Model arguments
            data_src: Data source instance
            checkpoint_path: Path to model checkpoint
        """
        self.args = args
        self.data_src = data_src

        model_data = GIKTModelData(data_src)
        train_data, val_data, graph, question_skill_matrix = model_data.prepare_data(
            args
        )

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")

        model = GIKT(args, data_src.metadata)

        super().__init__(model, checkpoint_path)

        self.with_inference(
            val_data, args.batch_size, args.device
        ).build_for_inference()

        self.graph = graph
        self.question_skill_matrix = question_skill_matrix

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """GIKT forward pass for inference.

        Args:
            batch_data: Tuple of (users, sequence, response, mask)

        Returns:
            Dictionary with y_hat (flattened), y_label (flattened), y_predict, and full_y_hat (unflattened)
        """
        users, sequence, response, mask = batch_data
        users = self._move_tensor_to_device(users)
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask, dtype=torch.bool)

        y_hat_full = self.model(
            sequence, response, mask, self.graph, self.question_skill_matrix
        )

        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        y_predict = self._generate_binary_predictions(y_hat, threshold=0.0)

        return {
            "y_hat": y_hat,
            "y_label": y_label,
            "y_predict": y_predict,
            "y_hat_full": y_hat_full,
        }

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract case data from batch outputs.

        Args:
            batch_data: Raw batch data (users, sequences, responses, masks)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        users, sequences, responses, masks = batch_data

        batch_size, seq_len = sequences.shape

        y_label = outputs["y_label"]
        y_predict = outputs["y_predict"]
        y_hat = outputs["y_hat"]

        if seq_len > 1:
            masks[:, 0] = False

        valid_indices = masks.view(-1).nonzero(as_tuple=True)[0]

        question_ids_flat = sequences.view(-1)[valid_indices].cpu().numpy()
        user_ids_flat = users.view(-1)[valid_indices].cpu().numpy()

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "labels": y_label.cpu().numpy(),
            "predictions": y_predict.cpu().numpy(),
            "logits": y_hat.cpu().numpy(),
            "masks": masks.view(-1)[valid_indices].cpu().numpy(),
        }
