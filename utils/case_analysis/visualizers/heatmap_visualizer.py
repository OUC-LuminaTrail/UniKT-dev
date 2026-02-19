"""Knowledge state heatmap visualization for case analysis.

Generates heatmap visualizations showing per-skill knowledge state changes
over a student's answer sequence, styled after the reference design.
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...core import get_logger

logger = get_logger(__name__)


class HeatmapVisualizer:
    """Generates knowledge state heatmap visualizations for case analysis.

    Creates a visual representation of a student's answer sequence showing:
    - Top row: skill label for each time step
    - Second row: correctness marker (✓/✗) for each time step
    - Main body: heatmap where each row is a skill's knowledge state over time
      (colour encodes mastery level, green=high, red=low)

    Requires the predictions DataFrame to contain a 'knowledge_state' column
    (list of floats per row, one value per skill).
    """

    def __init__(self) -> None:
        """Initialize the visualizer with default styling."""
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["font.size"] = 10

    def plot_user_heatmap(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        output_path: str | None = None,
        show_skill_names: bool = False,
    ) -> plt.Figure:
        """Plot knowledge-state heatmap for a single user.

        Layout (top to bottom):
          - Skill row  – skill label (c0, c1, …) for each answered question
          - Resp row   – ✓ (green) / ✗ (red) correctness markers
          - Heatmap    – [num_unique_skills × T] matrix of knowledge states
                         coloured with RdYlGn (red=low mastery, green=high)
          - x-axis     – 1-based position index
          - y-axis     – skill labels (c0, c1, …)
          - colorbar   – placed to the right of the heatmap

        Args:
            user_data: DataFrame for a single user. Required columns:
                - position: 0-based position in sequence
                - skill: skill ID (int) for each answered question
                - label: ground truth correctness (0/1)
                - knowledge_state: list[float] per-skill mastery values
                  (index == skill ID).
            user_id: User identifier used in the figure title.
            output_path: If given, save figure to this path (PNG, 300 dpi).
            show_skill_names: Unused; kept for API compatibility.

        Returns:
            matplotlib Figure object.

        Raises:
            ValueError: If required columns are missing or knowledge_state is invalid.
        """
        required_cols = {"position", "skill", "label", "knowledge_state"}
        missing = required_cols - set(user_data.columns)
        if missing:
            raise ValueError(f"Missing required columns in user_data: {missing}")

        if user_data["knowledge_state"].isna().any():
            raise ValueError(
                "knowledge_state column contains None values. "
                "Model must return knowledge states for case analysis."
            )

        user_data = user_data.sort_values("position").reset_index(drop=True)

        return self._plot_knowledge_state_heatmap(user_data, user_id, output_path)

    def _plot_knowledge_state_heatmap(
        self,
        user_data: pd.DataFrame,
        user_id: int,
        output_path: str | None,
    ) -> plt.Figure:
        """Full knowledge-state heatmap matching the reference design."""
        T = len(user_data)

        # Collect unique skills in order of first appearance
        seen: dict[int, None] = {}
        for s in user_data["skill"]:
            seen.setdefault(int(s), None)
        unique_skills: list[int] = list(seen.keys())
        num_skills = len(unique_skills)
        skill_to_row = {s: i for i, s in enumerate(unique_skills)}

        # Build knowledge-state matrix [num_skills × T]
        ks_matrix = np.full((num_skills, T), np.nan)
        for t, (_, row) in enumerate(user_data.iterrows()):
            ks = row.get("knowledge_state")
            if ks is None:
                continue
            for skill_id, row_idx in skill_to_row.items():
                if skill_id < len(ks):
                    ks_matrix[row_idx, t] = float(ks[skill_id])

        # Compute actual min/max from matrix (ignoring NaN) for dynamic color scaling
        ks_valid = ks_matrix[~np.isnan(ks_matrix)]
        if ks_valid.size > 0:
            ks_min = float(ks_valid.min())
            ks_max = float(ks_valid.max())
            # Handle case where all values are identical
            if ks_max - ks_min < 1e-8:
                ks_min = ks_max - 1.0
        else:
            ks_min, ks_max = 0.0, 1.0

        # Figure dimensions – scale with sequence length and skill count
        skill_row_h = 0.55
        resp_row_h = 0.55
        cell_h = max(0.45, min(0.85, 10.0 / max(num_skills, 1)))
        heatmap_h = num_skills * cell_h
        fig_w = max(14, T * 0.38 + 3.0)
        fig_h = max(4.5, skill_row_h + resp_row_h + heatmap_h + 1.2)

        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(
            f"User {user_id} – Knowledge State Heatmap",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )

        # GridSpec: 3 rows × 2 cols (main area + narrow colorbar strip)
        gs = gridspec.GridSpec(
            3,
            2,
            figure=fig,
            height_ratios=[skill_row_h, resp_row_h, heatmap_h],
            width_ratios=[1, 0.015],
            hspace=0.0,
            wspace=0.02,
            left=0.07,
            right=0.93,
            top=0.93,
            bottom=0.10,
        )

        ax_skill = fig.add_subplot(gs[0, 0])
        ax_resp = fig.add_subplot(gs[1, 0])
        ax_main = fig.add_subplot(gs[2, 0])
        ax_cbar = fig.add_subplot(gs[2, 1])

        # Draw annotation rows
        self._draw_skill_row(ax_skill, user_data, unique_skills, T)
        self._draw_resp_row(ax_resp, user_data, T)

        # Main heatmap
        cmap = plt.get_cmap("RdYlGn")
        im = ax_main.imshow(
            ks_matrix,
            aspect="auto",
            cmap=cmap,
            vmin=ks_min,
            vmax=ks_max,
            interpolation="none",
        )

        # y-axis: skill labels aligned left
        ax_main.set_yticks(np.arange(num_skills))
        ax_main.set_yticklabels(
            [self._skill_label(s) for s in unique_skills], fontsize=10
        )
        ax_main.tick_params(axis="y", length=0, pad=4)

        # x-axis: 1-based position labels
        ax_main.set_xticks(np.arange(T))
        ax_main.set_xticklabels([str(t + 1) for t in range(T)], fontsize=8)
        ax_main.tick_params(axis="x", length=0)

        for spine in ax_main.spines.values():
            spine.set_visible(False)

        # Colorbar
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=8, length=2)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
            logger.info(f"Saved user {user_id} heatmap to {output_path}")

        return fig

    def _draw_skill_row(
        self,
        ax: plt.Axes,
        user_data: pd.DataFrame,
        unique_skills: list[int],
        T: int,
    ) -> None:
        """Draw the 'Skill' header row with coloured skill labels."""
        cmap20 = plt.get_cmap("tab20")
        skill_color = {s: cmap20(i % 20) for i, s in enumerate(unique_skills)}

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Skill"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t, (_, row) in enumerate(user_data.iterrows()):
            skill_id = int(row["skill"])
            ax.text(
                t,
                0.5,
                self._skill_label(skill_id),
                ha="center",
                va="center",
                fontsize=9,
                color=skill_color[skill_id],
                fontweight="bold",
            )

    def _draw_resp_row(
        self,
        ax: plt.Axes,
        user_data: pd.DataFrame,
        T: int,
    ) -> None:
        """Draw the 'Resp' row with ✓/✗ correctness markers."""
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0.5])
        ax.set_yticklabels(["Resp"], fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for t, (_, row) in enumerate(user_data.iterrows()):
            correct = int(row["label"]) == 1
            symbol = "✓" if correct else "✗"
            color = "#2e7d32" if correct else "#c62828"  # dark green / dark red
            ax.text(
                t,
                0.5,
                symbol,
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                fontweight="bold",
            )

    @staticmethod
    def _skill_label(skill_id: int) -> str:
        """Return display label for a skill ID, e.g. 'c0', 'c1', …"""
        return f"c{skill_id}"


__all__ = ["HeatmapVisualizer"]
