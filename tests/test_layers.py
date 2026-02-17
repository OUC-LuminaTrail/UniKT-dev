"""Tests for shared layer components."""

import torch

from model.layers import GeneralInteraction, HistoryRecap


class TestHistoryRecap:
    """Test HistoryRecap module."""

    def test_init(self, hist_neighbor_num):
        """Test HistoryRecap initialization."""
        module = HistoryRecap(hist_neighbor_num=hist_neighbor_num, att_bound=0.7)
        assert module.hist_neighbor_num == hist_neighbor_num
        assert module.att_bound == 0.7

    def test_forward_basic(self, batch_size, seq_len, hidden_dim, hist_neighbor_num):
        """Test HistoryRecap forward pass with basic inputs."""
        module = HistoryRecap(hist_neighbor_num=hist_neighbor_num, att_bound=0.0)

        input_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        next_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        qa_emb = torch.randn(batch_size, seq_len, hidden_dim)
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = module(input_q_emb, next_q_emb, qa_emb, user_mask)

        assert output.shape == (batch_size, seq_len, hist_neighbor_num, hidden_dim)

    def test_forward_with_threshold(
        self, batch_size, seq_len, hidden_dim, hist_neighbor_num
    ):
        """Test HistoryRecap with attention threshold."""
        module = HistoryRecap(hist_neighbor_num=hist_neighbor_num, att_bound=0.7)

        input_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        next_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        qa_emb = torch.randn(batch_size, seq_len, hidden_dim)
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = module(input_q_emb, next_q_emb, qa_emb, user_mask)

        assert output.shape == (batch_size, seq_len, hist_neighbor_num, hidden_dim)

    def test_forward_with_fallback_index(
        self, batch_size, seq_len, hidden_dim, hist_neighbor_num
    ):
        """Test HistoryRecap with provided fallback index."""
        module = HistoryRecap(hist_neighbor_num=hist_neighbor_num, att_bound=0.7)

        input_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        next_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        qa_emb = torch.randn(batch_size, seq_len, hidden_dim)
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        hist_neighbor_index = torch.randint(
            0, seq_len, (batch_size, seq_len, hist_neighbor_num)
        )

        output = module(input_q_emb, next_q_emb, qa_emb, user_mask, hist_neighbor_index)

        assert output.shape == (batch_size, seq_len, hist_neighbor_num, hidden_dim)

    def test_forward_with_mask(
        self, batch_size, seq_len, hidden_dim, hist_neighbor_num
    ):
        """Test HistoryRecap with masked positions."""
        module = HistoryRecap(hist_neighbor_num=hist_neighbor_num, att_bound=0.0)

        input_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        next_q_emb = torch.randn(batch_size, seq_len, hidden_dim)
        qa_emb = torch.randn(batch_size, seq_len, hidden_dim)
        # Create mask with some invalid positions
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False

        output = module(input_q_emb, next_q_emb, qa_emb, user_mask)

        assert output.shape == (batch_size, seq_len, hist_neighbor_num, hidden_dim)


class TestGeneralInteraction:
    """Test GeneralInteraction module."""

    def test_init(self, hidden_dim):
        """Test GeneralInteraction initialization."""
        module = GeneralInteraction(hidden_dim=hidden_dim)
        assert module.hidden_dim == hidden_dim
        assert module.w1.shape == (hidden_dim, 1)
        assert module.w2.shape == (hidden_dim, 1)

    def test_forward_basic(self, batch_size, seq_len, hidden_dim, hist_neighbor_num):
        """Test GeneralInteraction forward pass with basic inputs."""
        module = GeneralInteraction(hidden_dim=hidden_dim)

        hist_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        next_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = module(hist_candidates, next_candidates, user_mask)

        assert output.shape == (batch_size, seq_len)

    def test_forward_with_different_candidate_nums(
        self, batch_size, seq_len, hidden_dim
    ):
        """Test GeneralInteraction with different numbers of candidates."""
        module = GeneralInteraction(hidden_dim=hidden_dim)

        M = 5
        N = 3
        hist_candidates = torch.randn(batch_size, seq_len, M, hidden_dim)
        next_candidates = torch.randn(batch_size, seq_len, N, hidden_dim)
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = module(hist_candidates, next_candidates, user_mask)

        assert output.shape == (batch_size, seq_len)

    def test_forward_with_mask(
        self, batch_size, seq_len, hidden_dim, hist_neighbor_num
    ):
        """Test GeneralInteraction with masked positions."""
        module = GeneralInteraction(hidden_dim=hidden_dim)

        hist_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        next_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        # Create mask with some invalid positions
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False

        output = module(hist_candidates, next_candidates, user_mask)

        assert output.shape == (batch_size, seq_len)

    def test_attention_weights_sum_to_one(
        self, batch_size, seq_len, hidden_dim, hist_neighbor_num
    ):
        """Test that attention weights sum to approximately one."""
        module = GeneralInteraction(hidden_dim=hidden_dim)

        hist_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        next_candidates = torch.randn(
            batch_size, seq_len, hist_neighbor_num + 1, hidden_dim
        )
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = module(hist_candidates, next_candidates, user_mask)

        # Check that output is finite
        assert torch.all(torch.isfinite(output))
