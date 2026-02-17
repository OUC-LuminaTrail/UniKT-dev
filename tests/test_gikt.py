"""Tests for GIKT model components."""

import pytest
import torch
from torch_geometric.data import HeteroData

from model.GIKT.GIKT_model import GIKT, GNN_QS


class TestGNNQS:
    """Test GNN_QS (Question-Skill Graph GNN) module."""

    @pytest.fixture
    def hetero_data(self, num_questions, num_skills, embedding_dim):
        """Create heterogeneous graph data for testing."""
        data = HeteroData()

        # Create some random edges between questions and skills
        num_edges = 100
        question_ids = torch.randint(0, num_questions, (num_edges,))
        skill_ids = torch.randint(0, num_skills, (num_edges,))

        data["question"].node_id = torch.arange(num_questions)
        data["skill"].node_id = torch.arange(num_skills)

        data["question", "has", "skill"].edge_index = torch.stack(
            [question_ids, skill_ids]
        )
        data["skill", "rev_has", "question"].edge_index = torch.stack(
            [skill_ids, question_ids]
        )

        return data

    @pytest.fixture
    def gnn_qs_module(self, embedding_dim):
        """Create a GNN_QS module for testing."""
        return GNN_QS(
            embedding_dim=embedding_dim,
            n_hop=2,
            heads=4,
            dropout=0.1,
        )

    def test_init(self, gnn_qs_module, embedding_dim):
        """Test GNN_QS initialization."""
        assert gnn_qs_module.n_hop == 2
        assert gnn_qs_module.heads == 4
        assert len(gnn_qs_module.convs) == 2

    def test_forward_basic(
        self, gnn_qs_module, hetero_data, num_questions, num_skills, embedding_dim
    ):
        """Test GNN_QS forward pass."""
        x = {
            "question": torch.randn(num_questions, embedding_dim),
            "skill": torch.randn(num_skills, embedding_dim),
        }

        output = gnn_qs_module(x, hetero_data.edge_index_dict)

        assert "question" in output
        assert "skill" in output
        assert output["question"].shape == (num_questions, embedding_dim)
        assert output["skill"].shape == (num_skills, embedding_dim)

    def test_forward_with_dropout(
        self, gnn_qs_module, hetero_data, num_questions, num_skills, embedding_dim
    ):
        """Test GNN_QS forward pass with dropout enabled."""
        gnn_qs_module.train()  # Enable dropout
        x = {
            "question": torch.randn(num_questions, embedding_dim),
            "skill": torch.randn(num_skills, embedding_dim),
        }

        output = gnn_qs_module(x, hetero_data.edge_index_dict)

        assert "question" in output
        assert "skill" in output
        assert output["question"].shape == (num_questions, embedding_dim)
        assert output["skill"].shape == (num_skills, embedding_dim)


class TestGIKT:
    """Test GIKT model."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for GIKT model."""

        class MockArgs:
            embedding_dim = 64
            hidden_dim = 64  # Must equal embedding_dim for some operations
            lstm_layers = 2
            dropout = 0.1
            n_hop = 2
            heads = 4
            history_neighbour = 3
            att_bound = 0.7

        return MockArgs()

    @pytest.fixture
    def data_metadata(self, num_questions, num_skills):
        """Create data metadata for GIKT model."""
        return {
            "num_questions": num_questions,
            "num_skills": num_skills,
        }

    @pytest.fixture
    def hetero_graph(self, num_questions, num_skills):
        """Create heterogeneous graph for GIKT."""
        data = HeteroData()

        # Create some random edges
        num_edges = 100
        question_ids = torch.randint(0, num_questions, (num_edges,))
        skill_ids = torch.randint(0, num_skills, (num_edges,))

        data["question", "has", "skill"].edge_index = torch.stack(
            [question_ids, skill_ids]
        )
        data["skill", "rev_has", "question"].edge_index = torch.stack(
            [skill_ids, question_ids]
        )

        return data

    @pytest.fixture
    def question_skill_matrix(self, num_questions, num_skills):
        """Create question-skill matrix."""
        return torch.randint(0, 2, (num_questions, num_skills)).float()

    @pytest.fixture
    def gikt_model(self, mock_args, data_metadata):
        """Create a GIKT model for testing."""
        return GIKT(mock_args, data_metadata)

    def test_init(self, gikt_model, mock_args, num_questions, num_skills):
        """Test GIKT initialization."""
        assert gikt_model.embedding_dim == mock_args.embedding_dim
        assert gikt_model.hidden_dim == mock_args.hidden_dim
        assert gikt_model.lstm_layers == mock_args.lstm_layers
        assert hasattr(gikt_model, "question_embedding")
        assert hasattr(gikt_model, "skill_embedding")
        assert hasattr(gikt_model, "answer_embedding")
        assert hasattr(gikt_model, "conv")

    def test_forward_basic(
        self,
        gikt_model,
        batch_size,
        seq_len,
        hetero_graph,
        question_skill_matrix,
    ):
        """Test GIKT forward pass with basic inputs."""
        user_sequence = torch.randint(
            0, gikt_model.data_metadata["num_questions"], (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = gikt_model(
            user_sequence,
            user_response,
            user_mask,
            hetero_graph,
            question_skill_matrix,
        )

        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(
        self,
        gikt_model,
        batch_size,
        seq_len,
        hetero_graph,
        question_skill_matrix,
    ):
        """Test GIKT forward pass with masked positions."""
        user_sequence = torch.randint(
            0, gikt_model.data_metadata["num_questions"], (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        # Create mask with some invalid positions
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False

        output = gikt_model(
            user_sequence,
            user_response,
            user_mask,
            hetero_graph,
            question_skill_matrix,
        )

        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_forward_single_sequence(
        self,
        gikt_model,
        hetero_graph,
        question_skill_matrix,
        num_questions,
    ):
        """Test GIKT forward pass with a single sequence (batch_size=1)."""
        batch_size = 1
        seq_len = 5

        user_sequence = torch.randint(0, num_questions, (batch_size, seq_len))
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = gikt_model(
            user_sequence,
            user_response,
            user_mask,
            hetero_graph,
            question_skill_matrix,
        )

        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_embeddings_initialized(self, gikt_model):
        """Test that embeddings are properly initialized."""
        assert (
            gikt_model.question_embedding.weight.shape[0]
            == gikt_model.data_metadata["num_questions"]
        )
        assert (
            gikt_model.skill_embedding.weight.shape[0]
            == gikt_model.data_metadata["num_skills"]
        )
        assert gikt_model.answer_embedding.weight.shape[0] == 2
