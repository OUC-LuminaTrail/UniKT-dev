"""Result collector for case analysis.

Collects, organizes, and filters inference results.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve

from ..core import get_logger

logger = get_logger(__name__)


class ResultCollector:
    """Collects, organizes, and filters inference results.

    This class accumulates batch results during inference and provides
    methods for analysis, filtering, and visualization preparation.
    """

    def __init__(self, device: torch.device):
        """Initialize the result collector.

        Args:
            device: Device where tensors are stored
        """
        self.device = device
        self.data = {
            "user_ids": [],
            "question_ids": [],
            "skills": [],
            "labels": [],
            "predictions": [],
            "logits": [],
            "mask": [],
            "knowledge_states": [],
        }
        self._df = None  # Cached DataFrame

    def add_batch(self, case_data: dict):
        required_keys = ["user_ids", "question_ids", "labels", "predictions", "logits"]
        missing_keys = [k for k in required_keys if k not in case_data]
        if missing_keys:
            raise ValueError(f"Missing required keys in case_data: {missing_keys}")

        for key, value in case_data.items():
            if key in self.data:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                self.data[key].extend(
                    value.tolist() if hasattr(value, "tolist") else [value]
                )

        for key, default in [("mask", 1), ("skills", 0), ("knowledge_states", None)]:
            if key not in case_data or len(case_data[key]) == 0:
                current_len = len(self.data[key])
                target_len = len(self.data["user_ids"])
                if current_len < target_len:
                    self.data[key].extend([default] * (target_len - current_len))

        self._df = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected data to pandas DataFrame.

        Returns:
            DataFrame with columns: user_id, position, question_id, skill,
            label, prediction, logit, mask
        """
        if self._df is not None:
            return self._df

        # Verify all lists have same length
        lengths = {k: len(v) for k, v in self.data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Inconsistent data lengths: {lengths}. "
                "All arrays must have the same number of samples."
            )

        df = pd.DataFrame(
            {
                "user_id": self.data["user_ids"],
                "question_id": self.data["question_ids"],
                "skill": self.data["skills"],
                "label": self.data["labels"],
                "prediction": self.data["predictions"],
                "logit": self.data["logits"],
                "mask": self.data["mask"],
                "knowledge_state": self.data["knowledge_states"],
            }
        )

        # Add position column per user (preserve insertion order as sequence order)
        df["position"] = df.groupby("user_id").cumcount()

        self._df = df
        return df

    def save(self, output_path: str):
        """Save collected results to parquet file.

        Args:
            output_path: Path to save parquet file
        """
        df = self.to_dataframe()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    @staticmethod
    def load(input_path: str) -> "ResultCollector":
        """Load results from parquet file.

        Args:
            input_path: Path to parquet file

        Returns:
            ResultCollector instance with loaded data
        """
        logger.info(f"Loading results from {input_path}...")
        df = pd.read_parquet(input_path)

        # Create a dummy device (CPU since we're loading from disk)
        collector = ResultCollector(device=torch.device("cpu"))

        # Populate data from DataFrame
        for col in [
            "user_id",
            "question_id",
            "skill",
            "label",
            "prediction",
            "logit",
            "mask",
        ]:
            if col in df.columns:
                target_col = col if col != "user_id" else "user_ids"
                target_col = target_col if target_col != "skill" else "skills"
                collector.data[target_col] = df[col].tolist()

        # Load knowledge_state column (list of lists, must not contain None)
        if "knowledge_state" in df.columns:
            if df["knowledge_state"].isna().any():
                raise ValueError(
                    "knowledge_state column contains None values. "
                    "Model must return knowledge states for case analysis."
                )
            collector.data["knowledge_states"] = (
                df["knowledge_state"].apply(list).tolist()
            )

        # Cache DataFrame
        collector._df = df
        logger.info(f"Loaded {len(df)} records from {input_path}")

        return collector

    def calculate_user_metrics(self) -> pd.DataFrame:
        """Calculate per-user metrics.

        Returns:
            DataFrame with columns:
            - user_id: User identifier
            - num_attempts: Number of attempts
            - correct_rate: Actual correct rate
            - predicted_correct_rate: Average predicted correct rate
            - accuracy: Prediction accuracy
            - auc: Area under ROC curve (if enough variance)
            - avg_confidence: Average prediction confidence
            - error_rate: Proportion of incorrect predictions
            - calibration_error: |predicted_correct_rate - correct_rate|
        """
        df = self.to_dataframe()

        user_metrics = []
        for user_id in df["user_id"].unique():
            user_df = df[df["user_id"] == user_id]

            n_attempts = len(user_df)
            correct_rate = user_df["label"].mean()
            predicted_correct_rate = user_df["prediction"].mean()
            accuracy = (user_df["prediction"].round() == user_df["label"]).mean()
            avg_confidence = (
                user_df["prediction"] * user_df["label"]
                + (1 - user_df["prediction"]) * (1 - user_df["label"])
            ).mean()
            error_rate = 1 - accuracy
            calibration_error = abs(predicted_correct_rate - correct_rate)

            # Calculate AUC if possible
            auc_score = np.nan
            if user_df["label"].nunique() > 1:
                try:
                    fpr, tpr, _ = roc_curve(user_df["label"], user_df["prediction"])
                    auc_score = auc(fpr, tpr)
                except ValueError:
                    pass

            user_metrics.append(
                {
                    "user_id": user_id,
                    "num_attempts": n_attempts,
                    "correct_rate": correct_rate,
                    "predicted_correct_rate": predicted_correct_rate,
                    "accuracy": accuracy,
                    "auc": auc_score,
                    "avg_confidence": avg_confidence,
                    "error_rate": error_rate,
                    "calibration_error": calibration_error,
                }
            )

        return pd.DataFrame(user_metrics)

    def select_users(
        self,
        min_num_attempts: int = 10,
        error_rate_range: tuple[float, float] = (0.1, 0.9),
        confidence_range: tuple[float, float] = (0.3, 0.95),
        max_users: int = 20,
        strategy: Literal["diverse", "extreme", "random"] = "diverse",
    ) -> list[int]:
        """Select users based on filtering criteria.

        Args:
            min_num_attempts: Minimum attempt count required
            error_rate_range: Range of valid error rates (min, max)
            confidence_range: Range of valid avg confidence (min, max)
            max_users: Maximum number of users to select
            strategy: Selection strategy:
                - diverse: Sample from different error rate bins
                - extreme: Select users with highest errors (for debugging)
                - random: Random selection from filtered pool

        Returns:
            List of selected user IDs
        """
        user_metrics = self.calculate_user_metrics()

        # Apply filters
        filtered = user_metrics[
            (user_metrics["num_attempts"] >= min_num_attempts)
            & (user_metrics["error_rate"] >= error_rate_range[0])
            & (user_metrics["error_rate"] <= error_rate_range[1])
            & (user_metrics["avg_confidence"] >= confidence_range[0])
            & (user_metrics["avg_confidence"] <= confidence_range[1])
        ].copy()

        if len(filtered) == 0:
            logger.warning(
                "No users match the filtering criteria. Returning empty list."
            )
            return []

        if len(filtered) <= max_users:
            logger.info(
                f"Only {len(filtered)} users match criteria, returning all of them."
            )
            return filtered["user_id"].tolist()

        # Apply selection strategy
        if strategy == "random":
            selected = filtered.sample(n=max_users, random_state=42)["user_id"].tolist()

        elif strategy == "extreme":
            # Select users with highest error rates
            selected = filtered.nlargest(max_users, "error_rate")["user_id"].tolist()

        elif strategy == "diverse":
            # Sample from different error rate bins
            filtered["error_bin"] = pd.cut(
                filtered["error_rate"],
                bins=5,
                labels=["very_low", "low", "medium", "high", "very_high"],
            )

            selected = []
            per_bin = max(1, max_users // filtered["error_bin"].nunique())

            for bin_label in filtered["error_bin"].unique():
                bin_users = filtered[filtered["error_bin"] == bin_label]
                n_take = min(per_bin, len(bin_users))
                sampled = (
                    bin_users.sample(n=n_take, random_state=42)["user_id"].tolist()
                    if len(bin_users) > n_take
                    else bin_users["user_id"].tolist()
                )
                selected.extend(sampled)

            # If we still need more users, randomly sample from remaining
            if len(selected) < max_users:
                remaining = filtered[~filtered["user_id"].isin(selected)]
                additional = remaining.sample(
                    n=min(max_users - len(selected), len(remaining)), random_state=42
                )["user_id"].tolist()
                selected.extend(additional)

            # Trim if we have too many
            selected = selected[:max_users]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(
            f"Selected {len(selected)} users using '{strategy}' strategy "
            f"(from {len(filtered)} filtered users)"
        )

        return selected

    def get_user_sequence(self, user_id: int) -> pd.DataFrame:
        """Get full sequence data for a specific user.

        Args:
            user_id: User identifier

        Returns:
            DataFrame with user's answer sequence, sorted by position
        """
        df = self.to_dataframe()
        user_df = df[df["user_id"] == user_id].copy()
        user_df = user_df.sort_values("position").reset_index(drop=True)
        return user_df


__all__ = ["ResultCollector"]
