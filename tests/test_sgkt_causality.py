"""Tests for SGKT historical-neighbor causality guards."""

import numpy as np
import torch

from model.SGKT.SGKT_data import sample_hist_neighbors
from model.SGKT.SGKT_model import SelfAttentionHistory


def test_sample_hist_neighbors_only_uses_past_or_padding():
    """Neighbor indices must be strictly historical or explicit padding."""
    batch_size = 3
    seq_len = 8
    hist_neighbor_num = 4
    pad_index = seq_len

    skill_index = np.array(
        [
            [0, 1, 0, 2, 2, 3, 1, 1],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=np.int64,
    )

    hist_neighbor_index = sample_hist_neighbors(
        batch_size=batch_size,
        max_seq_len=seq_len,
        hist_neighbor_num=hist_neighbor_num,
        skill_index=skill_index,
        pad_index=pad_index,
    )

    for t in range(seq_len):
        current = hist_neighbor_index[:, t, :]
        valid = (current < t) | (current == pad_index)
        assert np.all(valid), f"Found non-causal index at timestep {t}: {current}"


def test_self_attention_history_padding_index_maps_to_zero_feature():
    """Padding index should always gather a zero vector."""
    batch_size = 2
    seq_len = 6
    hidden_dim = 5
    hist_neighbor_num = 3

    module = SelfAttentionHistory(
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        hist_neighbor_num=hist_neighbor_num,
    )

    input_embedding = torch.randn(batch_size, seq_len, hidden_dim)
    hist_neighbor_index = torch.full(
        (batch_size, seq_len, hist_neighbor_num),
        fill_value=seq_len,
        dtype=torch.long,
    )

    output = module(input_embedding, hist_neighbor_index)
    assert output.shape == (batch_size, seq_len, hist_neighbor_num, hidden_dim)
    assert torch.allclose(output, torch.zeros_like(output), atol=0.0, rtol=0.0)
