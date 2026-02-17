"""Base analyzer class for case analysis.

Provides inference-only capabilities extending BaseTrainer.
"""

from abc import abstractmethod
from typing import Any

import torch

from ..config import DataConfig, TrainingConfig, create_optimized_dataloader
from ..core import get_logger
from ..training import BaseTrainer
from .result_collector import ResultCollector

logger = get_logger(__name__)


class BaseCaseAnalyzer(BaseTrainer):
    """Base class for case analysis with inference-only capabilities.

    This class extends BaseTrainer to provide simplified inference-only
    functionality for analyzing model predictions on datasets.

    Subclasses must implement:
    - forward_pass: Model forward pass (inherited from BaseTrainer)
    - extract_case_data: Extract model-specific data from batch outputs
    """

    def __init__(self, model: torch.nn.Module, checkpoint_path: str):
        """Initialize the analyzer.

        Args:
            model: PyTorch model to analyze
            checkpoint_path: Path to model checkpoint to load
        """
        super().__init__(model)
        self.checkpoint_path = checkpoint_path
        self.result_collector = None
        self._is_built_for_inference = False

    def with_inference(
        self, data, batch_size: int, device: torch.device | None = None
    ) -> "BaseCaseAnalyzer":
        """Configure for inference (simplified alternative to with_training).

        Args:
            data: Dataset to run inference on
            batch_size: Batch size for inference
            device: Device to run inference on (None for auto-detect)

        Returns:
            Self for method chaining
        """
        self._data_config = DataConfig(
            train_data=None, val_data=data, batch_size=batch_size
        )
        self._training_config = TrainingConfig(
            epochs=1, seed=None, device=device, checkpoint_path=self.checkpoint_path
        )
        return self

    def build_for_inference(self) -> "BaseCaseAnalyzer":
        """Build analyzer without optimizer/loss (inference-only).

        Returns:
            Self for method chaining
        """
        if self._is_built_for_inference:
            logger.warning("Analyzer already built for inference. Skipping rebuild.")
            return self

        # Validate required configurations
        if self._data_config is None:
            raise ValueError("Data configuration not set. Call with_inference() first.")

        # 1. Setup device
        if self._training_config.device is None:
            self.device_ = self._try_gpu()
        else:
            self.device_ = torch.device(self._training_config.device)

        # 2. Create dataloader
        val_data = self._data_config.val_data
        batch_size = self._data_config.batch_size

        if isinstance(val_data, torch.utils.data.Dataset):
            self.val_data = create_optimized_dataloader(
                val_data,
                batch_size=batch_size,
                shuffle=False,
                device=self.device_,
                collate_fn=None,
            )
        else:
            self.val_data = val_data

        # 3. Load checkpoint
        logger.info(f"Loading checkpoint from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device_)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        logger.info("Checkpoint loaded successfully")

        # 4. Set model to eval mode
        self.model.to(self.device_)
        self.model.eval()

        # 5. Initialize ResultCollector
        self.result_collector = ResultCollector(self.device_)

        self._is_built_for_inference = True
        logger.info("Analyzer built successfully for inference")
        return self

    @abstractmethod
    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract model-specific data from batch outputs.

        This method allows each model analyzer to customize what data
        is collected during inference. At minimum, it should return
        a dictionary with:
        - user_ids: User identifiers
        - question_ids: Question identifiers
        - labels: Ground truth labels
        - predictions: Model predictions
        - logits: Raw model outputs

        Args:
            batch_data: Raw batch data from dataloader
            outputs: Output dict from forward_pass (contains y_hat, y_label, y_predict)

        Returns:
            Dictionary with extracted data
        """
        raise NotImplementedError("Subclasses must implement extract_case_data method")

    @torch.no_grad()
    def run_inference(self) -> ResultCollector:
        """Run inference and collect all results.

        Returns:
            ResultCollector with all prediction results
        """
        if not self._is_built_for_inference:
            raise RuntimeError(
                "Analyzer not built for inference. Call build_for_inference() first."
            )

        logger.info("Running inference...")
        self.model.eval()

        for batch_idx, batch_data in enumerate(self.val_data):
            # Forward pass
            outputs = self.forward_pass(batch_data)

            # Extract model-specific data
            case_data = self.extract_case_data(batch_data, outputs)

            # Add to collector
            self.result_collector.add_batch(case_data)

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")

        logger.info("Inference complete")
        return self.result_collector


__all__ = ["BaseCaseAnalyzer"]
