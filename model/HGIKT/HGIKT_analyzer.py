"""HGIKT Case Analyzer.

Provides inference-only capabilities for HGIKT model case analysis.
"""

from typing import Any

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
        self, batch_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """HGIKT forward pass for inference.

        Args:
            batch_data: Tuple of (sequence, response, mask)

        Returns:
            Dictionary with y_hat (flattened), y_label (flattened), y_predict, and full_y_hat (unflattened)
        """
        sequence, response, mask = batch_data
        sequence = self._move_tensor_to_device(sequence)
        response = self._move_tensor_to_device(response)
        mask = self._move_tensor_to_device(mask)

        # Model forward pass
        y_hat_full = self.model(
            sequence,
            response,
            mask,
            self.hetero_graph,
            self.hypergraph,
            self.question_skill_matrix,
        )

        # Extract valid predictions (skip first position)
        y_hat, y_label, _ = self._extract_valid_predictions(
            y_hat_full, response, mask, skip_first=True
        )

        # Handle empty batch
        y_hat, y_label = self._handle_empty_batch(y_hat, y_label)

        # Generate binary predictions
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
            batch_data: Raw batch data (sequence, response, mask tuple)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        sequences, responses, masks = batch_data

        batch_size, seq_len = sequences.shape

        y_label = outputs["y_label"]
        y_predict = outputs["y_predict"]
        y_hat = outputs["y_hat"]

        masks_bool = masks.bool()
        if seq_len > 1:
            masks_bool[:, 0] = False

        valid_indices = masks_bool.view(-1).nonzero(as_tuple=True)[0]

        question_ids_flat = sequences.view(-1)[valid_indices].cpu().numpy()

        # Generate dummy user IDs (batch indices) since HGIKT doesn't track users
        user_ids_flat = (
            torch.arange(batch_size)
            .view(-1, 1)
            .expand(-1, seq_len)
            .reshape(-1)[valid_indices]
            .cpu()
            .numpy()
        )

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "labels": y_label.cpu().numpy(),
            "predictions": y_predict.cpu().numpy(),
            "logits": y_hat.cpu().numpy(),
            "masks": masks_bool.view(-1)[valid_indices].cpu().numpy(),
        }
