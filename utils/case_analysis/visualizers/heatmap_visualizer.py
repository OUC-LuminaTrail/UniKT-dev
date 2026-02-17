"""Heatmap visualization for case analysis.

Generates heatmap visualizations for user answer sequences.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from ...core import get_logger

logger = get_logger(__name__)


class HeatmapVisualizer:
    """Generates heatmap visualizations for case analysis.

    Creates visual representations of user answer sequences showing:
    - Actual correctness patterns
    - Model prediction probabilities
    - Combined views with error highlighting
    """

    def __init__(self):
        """Initialize the visualizer with default styling."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

        # Create custom colormaps
        self._create_colormaps()

    def _create_colormaps(self):
        """Create custom colormaps for visualization."""
        # Correctness colormap: light red (0) to light green (1)
        self.cmap_correctness = LinearSegmentedColormap.from_list(
            "correctness",
            ["#ffcccc", "#ccffcc"],  # light red to light green
            N=256,
        )

        # Prediction colormap: dark blue (low) to yellow (high)
        self.cmap_prediction = LinearSegmentedColormap.from_list(
            "prediction",
            ["#1a237e", "#4fc3f7", "#ffeb3b"],  # dark blue -> light blue -> yellow
            N=256,
        )

        # Error highlight colormap
        self.cmap_error = ListedColormap(
            ["#4caf50", "#f44336", "#ffffff"]  # green, red, white
        )

    def plot_user_heatmap(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        output_path: str | None = None,
        show_skill_names: bool = False,
    ) -> plt.Figure:
        """Plot 3-panel heatmap for a single user.

        Args:
            user_data: DataFrame with user's answer sequence. Must contain columns:
                - position: Position in sequence
                - label: Ground truth (0/1)
                - prediction: Model prediction probability
                - logit: Raw model output
            user_id: User identifier for title
            output_path: Path to save figure (None to skip saving)
            show_skill_names: Whether to show skill names on y-axis (requires 'skill' column)

        Returns:
            matplotlib Figure object
        """
        # Validate data
        required_cols = ["position", "label", "prediction", "logit"]
        missing = [c for c in required_cols if c not in user_data.columns]
        if missing:
            raise ValueError(f"Missing required columns in user_data: {missing}")

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))
        fig.suptitle(
            f"User {user_id} - Answer Sequence Analysis", fontsize=14, fontweight="bold"
        )

        seq_len = len(user_data)

        # Panel 1: Actual correctness
        ax1 = axes[0]
        correctness_matrix = user_data["label"].values.reshape(1, -1)
        im1 = ax1.imshow(
            correctness_matrix,
            aspect="auto",
            cmap=self.cmap_correctness,
            vmin=0,
            vmax=1,
        )

        # Mark positions
        ax1.set_xticks(range(seq_len))
        ax1.set_yticks([0])
        ax1.set_yticklabels(["Correctness"])
        ax1.set_ylabel("Actual", fontweight="bold")
        ax1.grid(axis="x", alpha=0.3)

        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, orientation="vertical", pad=0.02)
        cbar1.set_ticks([0, 1])
        cbar1.set_ticklabels(["Incorrect", "Correct"])

        # Panel 2: Prediction probabilities
        ax2 = axes[1]
        pred_matrix = user_data["prediction"].values.reshape(1, -1)
        im2 = ax2.imshow(
            pred_matrix, aspect="auto", cmap=self.cmap_prediction, vmin=0, vmax=1
        )

        ax2.set_xticks(range(seq_len))
        ax2.set_yticks([0])
        ax2.set_yticklabels(["Probability"])
        ax2.set_ylabel("Predicted", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        cbar2 = plt.colorbar(im2, ax=ax2, orientation="vertical", pad=0.02)
        cbar2.set_label("Probability")

        # Panel 3: Combined view with error markers
        ax3 = axes[2]

        # Create combined matrix: [logits, predictions, labels]
        # Normalize to [0, 1] for visualization
        logit_normalized = (user_data["logit"] - user_data["logit"].min()) / (
            user_data["logit"].max() - user_data["logit"].min() + 1e-8
        )

        combined_matrix = np.vstack(
            [
                logit_normalized.values.reshape(1, -1),
                user_data["prediction"].values.reshape(1, -1),
                user_data["label"].values.reshape(1, -1),
            ]
        )

        im3 = ax3.imshow(combined_matrix, aspect="auto", cmap="viridis")

        ax3.set_xticks(range(seq_len))
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(["Logits", "Prediction", "Label"])
        ax3.set_ylabel("Combined", fontweight="bold")
        ax3.grid(axis="x", alpha=0.3)

        cbar3 = plt.colorbar(im3, ax=ax3, orientation="vertical", pad=0.02)
        cbar3.set_label("Normalized Value")

        # Mark misclassifications with X
        predictions_binary = (user_data["prediction"] >= 0.5).astype(int)
        misclassified = predictions_binary != user_data["label"]
        for pos, is_error in enumerate(misclassified):
            if is_error:
                ax3.text(
                    pos,
                    1,
                    "✗",
                    ha="center",
                    va="center",
                    color="red",
                    fontsize=16,
                    fontweight="bold",
                )

        # Overall styling
        for ax in axes:
            ax.set_xlabel("Position in Sequence", fontsize=11)

        plt.tight_layout()

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved user {user_id} heatmap to {output_path}")

        return fig


__all__ = ["HeatmapVisualizer"]
