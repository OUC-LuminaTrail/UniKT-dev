"""Base analyzer class for case analysis.

Provides inference-only capabilities extending BaseTrainer. The analyzer owns
a different lifecycle from training: it is built from an archived RunConfig,
loads a trained checkpoint, and runs forward passes without an optimizer or
training loop. It therefore bypasses :meth:`BaseTrainer.__init__` and manages
its own subset of trainer state.
"""

from abc import abstractmethod
from typing import Any, override

import torch
from rich.live import Live

from ..config import create_optimized_dataloader
from ..core import get_logger
from ..progress import create_progress
from ..training import BaseTrainer
from ..training.runtime_components import RuntimeComponents
from .result_collector import ResultCollector

logger = get_logger(__name__)


class BaseCaseAnalyzer(BaseTrainer):
    """Base class for case analysis with inference-only capabilities.

    Subclasses must implement:

    - :meth:`forward_pass`: Model forward pass (inherited contract).
    - :meth:`extract_case_data`: Extract model-specific data from batch outputs.
    """

    def __init__(self, model: torch.nn.Module, checkpoint_path: str):
        """Initialize the analyzer with a model and checkpoint path.

        Does **not** call :meth:`BaseTrainer.__init__` — the analyzer runs no
        training loop, so the training template (build_components / build /
        exp_manager) does not apply. Inference setup happens in
        :meth:`configure_inference` followed by :meth:`build`.
        """
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.run_config = None
        self._components = RuntimeComponents(model=model)
        self._inference_device: str | None = None
        self._inference_batch_size: int = 1
        self._is_built_for_inference = False
        self.result_collector: ResultCollector | None = None

    def configure_inference(
        self,
        rc: Any,
        data: Any,
        batch_size: int,
        collate_fn=None,
        device: str | None = None,
    ) -> "BaseCaseAnalyzer":
        """Configure inference data, batch size, and device override.

        Args:
            rc: RunConfig (OmegaConf ``DictConfig``) — typically loaded from the
                run's archived ``run_config.yaml``.
            data: Validation/inference dataset (or a ready DataLoader).
            batch_size: Inference batch size.
            collate_fn: Optional collator for the dataset.
            device: Optional device override; falls back to ``rc.general.device``
                during :meth:`build`.

        Returns:
            Self for chaining into :meth:`build`.
        """
        self.run_config = rc
        self._components = RuntimeComponents(
            model=self.model, val_data=data, collate_fn=collate_fn
        )
        self._inference_device = device
        self._inference_batch_size = batch_size
        return self

    @override
    def build(self) -> "BaseCaseAnalyzer":
        """Finalize the analyzer: device, loader, checkpoint, eval mode."""
        if self._is_built_for_inference:
            logger.warning("Analyzer already built for inference. Skipping rebuild.")
            return self

        if self.run_config is None:
            raise ValueError("RunConfig not set. Call configure_inference() first.")
        c = self._components
        if c.val_data is None:
            raise ValueError(
                "Inference data not set. Call configure_inference() first."
            )

        dev = self._inference_device or self.run_config.general.device
        self.device_ = torch.device(dev) if dev else self._try_gpu()

        val_data = c.val_data
        if isinstance(val_data, torch.utils.data.Dataset):
            self.val_data = create_optimized_dataloader(
                val_data,
                batch_size=self._inference_batch_size,
                shuffle=False,
                device=self.device_,
                collate_fn=c.collate_fn,
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
            raise RuntimeError("Analyzer not built for inference. Call build() first.")

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
