"""Base analyzer class for case analysis.

Provides inference-only capabilities extending BaseTrainer.
"""

from abc import abstractmethod
from typing import Any

import torch
from rich.live import Live
from typing_extensions import override

from ..config import DataConfig, TrainingConfig, create_optimized_dataloader
from ..core import get_logger
from ..progress import create_progress
from ..training import BaseTrainer
from .result_collector import ResultCollector

logger = get_logger(__name__)


class BaseCaseAnalyzer(BaseTrainer):
    """Base class for case analysis with inference-only capabilities.

    Subclasses must implement:
    - forward_pass: Model forward pass (inherited from BaseTrainer)
    - extract_case_data: Extract model-specific data from batch outputs
    """

    def __init__(self, model: torch.nn.Module, checkpoint_path: str):
        """Initialize the case analyzer with a model and checkpoint path."""
        super().__init__(model)
        self.checkpoint_path = checkpoint_path
        self.result_collector = None
        self._is_built_for_inference = False

    def with_inference(
        self, data, batch_size: int, collate_fn=None, device: torch.device | None = None
    ) -> "BaseCaseAnalyzer":
        """Configure inference data and device settings."""
        self._data_config = DataConfig(
            train_data=None, val_data=data, batch_size=batch_size, collate_fn=collate_fn
        )
        self._training_config = TrainingConfig(
            epochs=1, seed=None, device=device, checkpoint_path=self.checkpoint_path
        )
        return self

    @override
    def build(self) -> "BaseCaseAnalyzer":
        if self._is_built_for_inference:
            logger.warning("Analyzer already built for inference. Skipping rebuild.")
            return self

        if self._data_config is None:
            raise ValueError("Data configuration not set. Call with_inference() first.")

        if self._training_config.device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(self._training_config.device)

        val_data = self._data_config.val_data
        if isinstance(val_data, torch.utils.data.Dataset):
            self.val_data = create_optimized_dataloader(
                val_data,
                batch_size=self._data_config.batch_size,
                shuffle=False,
                device=self.device_,
                collate_fn=self._data_config.collate_fn,
            )
        else:
            self.val_data = val_data

        from utils.training.checkpoint import CheckpointManager

        CheckpointManager.load_weights(self.checkpoint_path, self.model, self.device_)

        self.model.to(self.device_)
        self.model.eval()
        self.result_collector = ResultCollector(self.device_)
        self._is_built_for_inference = True
        logger.debug("Analyzer built successfully for inference")
        return self

    @abstractmethod
    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract model-specific data from batch outputs.

        Returns dictionary with at minimum:
        - user_ids: User identifiers
        - question_ids: Question identifiers
        - labels: Ground truth labels
        - predictions: Model predictions
        - logits: Raw model outputs

        Args:
            batch_data: Raw batch data from dataloader
            outputs: Output dict from forward_pass (y_hat, y_label, y_predict)

        Returns:
            Dictionary with extracted data
        """
        raise NotImplementedError("Subclasses must implement extract_case_data method")

    @torch.no_grad()
    def run_inference(self) -> ResultCollector:
        """Run inference over the configured data and collect results."""
        if not self._is_built_for_inference:
            raise RuntimeError(
                "Analyzer not built for inference. Call build_for_inference() first."
            )

        logger.info("Running inference...")

        progress = create_progress()

        with Live(progress):
            inference_task = progress.add_task(
                "[bold cyan]Inference", total=len(self.val_data)
            )

            for batch_data in self.val_data:
                outputs = self.forward_pass(batch_data)
                case_data = self.extract_case_data(batch_data, outputs)
                self.result_collector.add_batch(case_data)
                progress.advance(inference_task)

        logger.info("Inference complete")
        return self.result_collector


__all__ = ["BaseCaseAnalyzer"]
