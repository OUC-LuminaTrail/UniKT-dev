"""Tests for SGKT model components."""

import pytest
import torch
from torch_geometric.data import Data

from model.SGKT.SGKT_model import (
    SGKT,
    HRGEmbedding,
    NextNeighborSampler,
    SelfAttentionHistory,
    SGEmbedding,
)


class TestHRGEmbedding:
    """Test HRGEmbedding module."""

    @pytest.fixture
    def hrg_data(self, num_skills, num_questions):
        """Create HRG graph data for testing."""
        # Create edges between skills and questions
        num_edges = 100
        skill_ids = torch.randint(0, num_skills, (num_edges,))
        question_ids = torch.randint(0, num_questions, (num_edges,)) + num_skills

        # Add question-question edges
        num_qq_edges = 50
        qq_src = torch.randint(0, num_questions, (num_qq_edges,))
        qq_dst = torch.randint(0, num_questions, (num_qq_edges,))

        # Combine all edges
        all_src = torch.cat([skill_ids, qq_src + num_skills])
        all_dst = torch.cat([question_ids, qq_dst + num_skills])

        edge_index = torch.stack([all_src, all_dst])

        data = Data(edge_index=edge_index)
        return data

    @pytest.fixture
    def hrg_embedding(self, num_skills, num_questions, embedding_dim):
        """Create a HRGEmbedding module for testing."""
        return HRGEmbedding(
            num_skills=num_skills,
            num_questions=num_questions,
            embedding_dim=embedding_dim,
            num_layers=2,
            max_neighbors=20,
            dropout=0.1,
        )

    def test_init(self, hrg_embedding, num_skills, num_questions, embedding_dim):
        """Test HRGEmbedding initialization."""
        assert hrg_embedding.num_skills == num_skills
        assert hrg_embedding.num_questions == num_questions
        assert hrg_embedding.embedding_dim == embedding_dim
        assert len(hrg_embedding.convs) == 2

    def test_forward_basic(
        self, hrg_embedding, hrg_data, batch_size, seq_len, embedding_dim
    ):
        """Test HRGEmbedding forward pass."""
        question_indices = torch.randint(
            0, hrg_embedding.num_questions, (batch_size, seq_len)
        )

        question_features, neighbor_features = hrg_embedding(hrg_data, question_indices)

        assert question_features.shape == (batch_size, seq_len, embedding_dim)
        assert neighbor_features.shape == (batch_size, seq_len, 20, embedding_dim)
        assert torch.all(torch.isfinite(question_features))
        assert torch.all(torch.isfinite(neighbor_features))

    def test_precompute_neighbors(self, hrg_embedding, hrg_data):
        """Test neighbor precomputation."""
        hrg_embedding.precompute_neighbors(hrg_data.edge_index)

        assert hrg_embedding._precomputed
        assert hrg_embedding.neighbor_indices is not None
        assert hrg_embedding.neighbor_mask is not None


class TestSGEmbedding:
    """Test SGEmbedding module."""

    @pytest.fixture
    def sg_embedding(self, embedding_dim, hidden_dim):
        """Create a SGEmbedding module for testing."""
        return SGEmbedding(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=0.1,
        )

    def test_init(self, sg_embedding, embedding_dim, hidden_dim):
        """Test SGEmbedding initialization."""
        assert sg_embedding.embedding_dim == embedding_dim
        assert sg_embedding.hidden_dim == hidden_dim
        assert hasattr(sg_embedding, "gru_cell")

    def test_forward_basic(
        self, sg_embedding, batch_size, seq_len, embedding_dim, hidden_dim
    ):
        """Test SGEmbedding forward pass."""
        question_emb = torch.randn(batch_size, seq_len, embedding_dim)
        answer_emb = torch.randn(batch_size, seq_len, embedding_dim)
        input_trans_embedding = torch.randn(batch_size, seq_len, hidden_dim)
        user_sequence = torch.randint(0, 10, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = sg_embedding(
            question_emb,
            answer_emb,
            input_trans_embedding,
            user_sequence,
            user_mask,
        )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(
        self, sg_embedding, batch_size, seq_len, embedding_dim, hidden_dim
    ):
        """Test SGEmbedding with masked positions."""
        question_emb = torch.randn(batch_size, seq_len, embedding_dim)
        answer_emb = torch.randn(batch_size, seq_len, embedding_dim)
        input_trans_embedding = torch.randn(batch_size, seq_len, hidden_dim)
        user_sequence = torch.randint(0, 10, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False

        output = sg_embedding(
            question_emb,
            answer_emb,
            input_trans_embedding,
            user_sequence,
            user_mask,
        )

        assert output.shape == (batch_size, seq_len, embedding_dim)
        assert torch.all(torch.isfinite(output))


class TestNextNeighborSampler:
    """Test NextNeighborSampler module."""

    @pytest.fixture
    def sampler(self):
        """Create a NextNeighborSampler for testing."""
        return NextNeighborSampler(next_neighbor_num=5)

    def test_init(self, sampler):
        """Test NextNeighborSampler initialization."""
        assert sampler.next_neighbor_num == 5

    def test_forward_basic(self, sampler, batch_size, seq_len, hidden_dim):
        """Test NextNeighborSampler forward pass."""
        total_neighbors = 20
        neighbor_features = torch.randn(
            batch_size, seq_len, total_neighbors, hidden_dim
        )

        output = sampler(neighbor_features)

        assert output.shape == (batch_size, seq_len, 5, hidden_dim)
        assert torch.all(torch.isfinite(output))

    def test_forward_insufficient_neighbors(
        self, sampler, batch_size, seq_len, hidden_dim
    ):
        """Test NextNeighborSampler with insufficient neighbors."""
        total_neighbors = 3  # Less than next_neighbor_num=5
        neighbor_features = torch.randn(
            batch_size, seq_len, total_neighbors, hidden_dim
        )

        output = sampler(neighbor_features)

        assert output.shape == (batch_size, seq_len, 5, hidden_dim)
        assert torch.all(torch.isfinite(output))

    def test_forward_no_neighbors(self, sampler, batch_size, seq_len, hidden_dim):
        """Test NextNeighborSampler with no neighbors."""
        total_neighbors = 0
        neighbor_features = torch.randn(
            batch_size, seq_len, total_neighbors, hidden_dim
        )

        output = sampler(neighbor_features)

        assert output.shape == (batch_size, seq_len, 5, hidden_dim)
        # Output should be all zeros
        assert torch.all(output == 0.0)


class TestSelfAttentionHistory:
    """Test SelfAttentionHistory module."""

    @pytest.fixture
    def self_attention(self, hidden_dim, seq_len):
        """Create a SelfAttentionHistory module for testing."""
        return SelfAttentionHistory(
            hidden_dim=hidden_dim,
            seq_len=seq_len,
            hist_neighbor_num=3,
            dropout=0.1,
        )

    def test_init(self, self_attention, hidden_dim, seq_len):
        """Test SelfAttentionHistory initialization."""
        assert self_attention.hidden_dim == hidden_dim
        assert self_attention.seq_len == seq_len
        assert self_attention.hist_neighbor_num == 3

    def test_forward_basic(self, self_attention, batch_size, seq_len, hidden_dim):
        """Test SelfAttentionHistory forward pass."""
        input_embedding = torch.randn(batch_size, seq_len, hidden_dim)
        hist_neighbor_index = torch.randint(0, seq_len, (batch_size, seq_len, 3))

        output = self_attention(input_embedding, hist_neighbor_index)

        assert output.shape == (batch_size, seq_len, 3, hidden_dim)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(self, self_attention, batch_size, seq_len, hidden_dim):
        """Test SelfAttentionHistory with masked positions."""
        input_embedding = torch.randn(batch_size, seq_len, hidden_dim)
        # Create neighbor index with some invalid (-1) positions
        hist_neighbor_index = torch.randint(-1, seq_len, (batch_size, seq_len, 3))

        output = self_attention(input_embedding, hist_neighbor_index)

        assert output.shape == (batch_size, seq_len, 3, hidden_dim)
        assert torch.all(torch.isfinite(output))


class TestSGKT:
    """Test SGKT model."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for SGKT model."""

        class MockArgs:
            embedding_dim = 64
            hidden_dim = 32
            dropout = 0.1
            dropout_gnn = 0.1
            n_hop = 2
            hist_neighbor_num = 3
            next_neighbor_num = 3
            att_bound = 0.7

        return MockArgs()

    @pytest.fixture
    def data_metadata(self, num_questions, num_skills):
        """Create data metadata for SGKT model."""
        return {
            "num_questions": num_questions,
            "num_skills": num_skills,
            "max_seq_len": 10,
        }

    @pytest.fixture
    def hrg_data(self, num_skills, num_questions):
        """Create HRG graph data for SGKT."""
        # Create edges between skills and questions
        num_edges = 100
        skill_ids = torch.randint(0, num_skills, (num_edges,))
        question_ids = torch.randint(0, num_questions, (num_edges,)) + num_skills

        # Add question-question edges
        num_qq_edges = 50
        qq_src = torch.randint(0, num_questions, (num_qq_edges,))
        qq_dst = torch.randint(0, num_questions, (num_qq_edges,))

        # Combine all edges
        all_src = torch.cat([skill_ids, qq_src + num_skills])
        all_dst = torch.cat([question_ids, qq_dst + num_skills])

        edge_index = torch.stack([all_src, all_dst])

        data = Data(edge_index=edge_index)
        return data

    @pytest.fixture
    def sgkt_model(self, mock_args, data_metadata):
        """Create a SGKT model for testing."""
        return SGKT(mock_args, data_metadata)

    def test_init(self, sgkt_model, mock_args):
        """Test SGKT initialization."""
        assert sgkt_model.embedding_dim == mock_args.embedding_dim
        assert sgkt_model.hidden_dim == mock_args.hidden_dim
        assert hasattr(sgkt_model, "hrg_embedding")
        assert hasattr(sgkt_model, "sg_embedding")
        assert hasattr(sgkt_model, "self_attention")
        assert hasattr(sgkt_model, "next_sampler")

    def test_forward_basic(
        self,
        sgkt_model,
        batch_size,
        seq_len,
        hrg_data,
    ):
        """Test SGKT forward pass with basic inputs."""
        user_sequence = torch.randint(
            0, sgkt_model.num_questions, (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        hist_neighbor_index = torch.randint(0, seq_len, (batch_size, seq_len, 3))

        output = sgkt_model(
            user_sequence,
            user_response,
            user_mask,
            hrg_data,
            hist_neighbor_index,
        )

        # Output shape should be [batch_size, seq_len-1]
        assert output.shape == (batch_size, seq_len - 1)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(
        self,
        sgkt_model,
        batch_size,
        seq_len,
        hrg_data,
    ):
        """Test SGKT forward pass with masked positions."""
        user_sequence = torch.randint(
            0, sgkt_model.num_questions, (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False
        hist_neighbor_index = torch.randint(0, seq_len, (batch_size, seq_len, 3))

        output = sgkt_model(
            user_sequence,
            user_response,
            user_mask,
            hrg_data,
            hist_neighbor_index,
        )

        assert output.shape == (batch_size, seq_len - 1)
        assert torch.all(torch.isfinite(output))
