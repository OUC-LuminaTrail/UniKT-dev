"""Per-user metric computation for case analysis results."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve


def compute_user_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute diagnostic metrics per user from canonical case results.

    Args:
        df: Case analysis DataFrame with at least ``user_id``, ``label``
            and ``prediction`` columns (one row per attempt).

    Returns:
        DataFrame indexed by user with columns: ``num_attempts``,
        ``correct_rate``, ``predicted_correct_rate``, ``accuracy``,
        ``auc``, ``avg_confidence``, ``error_rate``,
        ``calibration_error``.
    """
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

        # AUC is undefined when the user has only one class of labels
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


__all__ = ["compute_user_metrics"]
