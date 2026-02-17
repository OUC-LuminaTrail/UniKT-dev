"""Tests for SQGKT model components."""

import numpy as np
import pytest
import torch

from model.SQGKT.SQGKT_model import SQGKT


class TestSQGKT:
    """Test SQGKT model."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for SQGKT model."""

        class MockArgs:
            dim_emb = 64
            agg_hops = 2
            dropout4lstm = 0.1
            dropout4gnn = 0.1
            rank_k = 3
            qs_question_neighbors = 5
            qs_skill_neighbors = 10
            uq_user_neighbors = 5
            uq_question_neighbors = 5

        return MockArgs()

    @pytest.fixture
    def data_metadata(self, num_questions, num_skills, num_users):
        """Create data metadata for SQGKT model."""
        return {
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_users": num_users,
        }

    @pytest.fixture
    def sqgkt_model(self, mock_args, data_metadata):
        """Create a SQGKT model for testing."""
        return SQGKT(mock_args, data_metadata)

    @pytest.fixture
    def qs_table(self, num_questions, num_skills):
        """Create question-skill relation table."""
        # Random binary matrix indicating which skills each question requires
        return torch.randint(0, 2, (num_questions, num_skills))

    @pytest.fixture
    def q_neighbors_qs(self, num_questions, num_skills):
        """Create question neighbors in QS graph."""
        # Each question has some skill neighbors (indices must be valid skill IDs)
        neighbors = []
        for q_id in range(num_questions):
            # Randomly select skill neighbors (with replacement to ensure we have enough)
            skill_neighbors = np.random.choice(num_skills, size=5, replace=True)
            neighbors.append(skill_neighbors)
        return torch.from_numpy(np.array(neighbors))

    @pytest.fixture
    def c_neighbors_qs(self, num_questions, num_skills):
        """Create skill neighbors in QS graph."""
        # Each skill has some question neighbors
        neighbors = []
        for s_id in range(num_skills):
            # Randomly select question neighbors
            question_neighbors = np.random.choice(num_questions, size=10, replace=True)
            neighbors.append(question_neighbors)
        return torch.from_numpy(np.array(neighbors))

    @pytest.fixture
    def uq_table(self, num_users, num_questions):
        """Create user-question table with 3 factors."""
        # Shape: [num_users, num_questions, 3]
        # Factors: ability, attempt, hint
        return torch.rand(num_users, num_questions, 3)

    @pytest.fixture
    def u_neighbors_uq(self, num_users, num_questions):
        """Create user neighbors in UQ graph."""
        # Each user has some question neighbors
        neighbors = []
        for u_id in range(num_users):
            # Randomly select question neighbors
            question_neighbors = np.random.choice(num_questions, size=5, replace=True)
            neighbors.append(question_neighbors)
        return torch.from_numpy(np.array(neighbors))

    @pytest.fixture
    def q_neighbors_uq(self, num_users, num_questions):
        """Create question neighbors in UQ graph."""
        # Each question has some user neighbors
        neighbors = []
        for q_id in range(num_questions):
            # Randomly select user neighbors
            user_neighbors = np.random.choice(num_users, size=5, replace=True)
            neighbors.append(user_neighbors)
        return torch.from_numpy(np.array(neighbors))

    def test_init(self, sqgkt_model, mock_args):
        """Test SQGKT initialization."""
        assert sqgkt_model.dim_emb == mock_args.dim_emb
        assert sqgkt_model.agg_hops == mock_args.agg_hops
        assert hasattr(sqgkt_model, "embed_question_qs")
        assert hasattr(sqgkt_model, "embed_question_uq")
        assert hasattr(sqgkt_model, "embed_concept")
        assert hasattr(sqgkt_model, "lstm1")

    def test_embeddings_initialized(self, sqgkt_model):
        """Test that embeddings are properly initialized."""
        assert sqgkt_model.embed_question_qs.weight.shape[0] == sqgkt_model.num_question
        assert sqgkt_model.embed_question_uq.weight.shape[0] == sqgkt_model.num_question
        assert sqgkt_model.embed_concept.weight.shape[0] == sqgkt_model.num_concept
        assert sqgkt_model.embed_user.weight.shape[0] == sqgkt_model.num_user
        assert sqgkt_model.embed_correctness.weight.shape[0] == 2

    def test_forward_basic(
        self,
        sqgkt_model,
        batch_size,
        seq_len,
        qs_table,
        q_neighbors_qs,
        c_neighbors_qs,
        uq_table,
        u_neighbors_uq,
        q_neighbors_uq,
        num_users,
        num_questions,
    ):
        """Test SQGKT forward pass with basic inputs."""
        user_seq = torch.randint(0, num_users, (batch_size, seq_len))
        question_seq = torch.randint(0, num_questions, (batch_size, seq_len))
        correctness_seq = torch.randint(0, 2, (batch_size, seq_len))
        mask_seq = torch.ones(batch_size, seq_len)  # All valid

        output = sqgkt_model(
            user_seq,
            question_seq,
            correctness_seq,
            mask_seq,
            qs_table,
            q_neighbors_qs,
            c_neighbors_qs,
            uq_table,
            u_neighbors_uq,
            q_neighbors_uq,
        )

        # Output shape is [batch_size, seq_len]
        # Last position contains zeros (initialized but never updated)
        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(
        self,
        sqgkt_model,
        batch_size,
        seq_len,
        qs_table,
        q_neighbors_qs,
        c_neighbors_qs,
        uq_table,
        u_neighbors_uq,
        q_neighbors_uq,
        num_users,
        num_questions,
    ):
        """Test SQGKT forward pass with masked positions."""
        user_seq = torch.randint(0, num_users, (batch_size, seq_len))
        question_seq = torch.randint(0, num_questions, (batch_size, seq_len))
        correctness_seq = torch.randint(0, 2, (batch_size, seq_len))
        # Create mask with some invalid positions (mask=0 means invalid)
        mask_seq = torch.ones(batch_size, seq_len)
        mask_seq[:, seq_len // 2 :] = 0

        output = sqgkt_model(
            user_seq,
            question_seq,
            correctness_seq,
            mask_seq,
            qs_table,
            q_neighbors_qs,
            c_neighbors_qs,
            uq_table,
            u_neighbors_uq,
            q_neighbors_uq,
        )

        # Output shape is [batch_size, seq_len]
        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_forward_single_sequence(
        self,
        sqgkt_model,
        seq_len,
        qs_table,
        q_neighbors_qs,
        c_neighbors_qs,
        uq_table,
        u_neighbors_uq,
        q_neighbors_uq,
        num_users,
        num_questions,
    ):
        """Test SQGKT forward pass with a single sequence (batch_size=1)."""
        batch_size = 1
        user_seq = torch.randint(0, num_users, (batch_size, seq_len))
        question_seq = torch.randint(0, num_questions, (batch_size, seq_len))
        correctness_seq = torch.randint(0, 2, (batch_size, seq_len))
        mask_seq = torch.ones(batch_size, seq_len)

        output = sqgkt_model(
            user_seq,
            question_seq,
            correctness_seq,
            mask_seq,
            qs_table,
            q_neighbors_qs,
            c_neighbors_qs,
            uq_table,
            u_neighbors_uq,
            q_neighbors_uq,
        )

        # Output shape is [batch_size, seq_len]
        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))
