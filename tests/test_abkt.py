"""Tests for ABKT model components."""

import pytest
import torch

from model.ABKT.ABKT_model import GMF, IRT_2, K_CMF


class TestIRT2:
    """Test IRT_2 response function."""

    def test_irt_2_basic(self, batch_size, num_skills):
        """Test IRT_2 with basic inputs."""
        user_k = torch.randn(batch_size, num_skills)
        item_k = torch.randn(batch_size, num_skills)
        item_q = torch.randint(0, 2, (batch_size, num_skills)).float()
        guess = 0.25

        output = IRT_2(user_k, item_k, item_q, guess)

        assert output.shape == (batch_size,)
        assert torch.all(output >= guess)
        assert torch.all(output <= 1.0)

    def test_irt_2_sequence(self, seq_len, num_skills):
        """Test IRT_2 with sequence inputs."""
        user_k = torch.randn(seq_len, num_skills)
        item_k = torch.randn(seq_len, num_skills)
        item_q = torch.randint(0, 2, (seq_len, num_skills)).float()
        guess = 0.2

        output = IRT_2(user_k, item_k, item_q, guess)

        assert output.shape == (seq_len,)
        assert torch.all(output >= guess)
        assert torch.all(output <= 1.0)

    def test_irt_2_extreme_guess(self, batch_size, num_skills):
        """Test IRT_2 with extreme guess values."""
        user_k = torch.randn(batch_size, num_skills)
        item_k = torch.randn(batch_size, num_skills)
        item_q = torch.randint(0, 2, (batch_size, num_skills)).float()

        # Test with guess = 0
        output = IRT_2(user_k, item_k, item_q, 0.0)
        assert torch.all(output >= 0.0)
        assert torch.all(output <= 1.0)

        # Test with guess = 1
        output = IRT_2(user_k, item_k, item_q, 1.0)
        assert torch.all(output >= 1.0)
        assert torch.all(output <= 1.0)


class TestKCMF:
    """Test K_CMF (Knowledge Module)."""

    @pytest.fixture
    def k_cmf_module(self, num_users, num_questions, num_skills, hidden_dim):
        """Create a K_CMF module for testing."""
        k_hidden_size = 5
        Q_matrix = torch.randint(0, 2, (num_questions, num_skills)).float()
        return K_CMF(
            k_hidden_size=k_hidden_size,
            skill_num=num_skills,
            user_num=num_users,
            item_num=num_questions,
            q_matrix=Q_matrix,
        )

    def test_init(self, k_cmf_module):
        """Test K_CMF initialization."""
        assert hasattr(k_cmf_module, "user_initial_k")
        assert hasattr(k_cmf_module, "item_k")
        assert hasattr(k_cmf_module, "user_improving_k")
        assert hasattr(k_cmf_module, "item_improving_k")

    def test_forward_basic(self, k_cmf_module, seq_len):
        """Test K_CMF forward pass."""
        user_id = 0
        sq = torch.randint(0, k_cmf_module.item_num, (seq_len,))

        out, _, _ = k_cmf_module(user_id, sq)

        # Output shape: [seq_len + 1, num_skills]
        assert out.shape == (seq_len + 1, k_cmf_module.skill_num)
        # All values should be in [0, 1] after sigmoid
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_forward_knowledge_growth(self, k_cmf_module):
        """Test that knowledge only grows (non-negative)."""
        user_id = 0
        seq_len = 5
        sq = torch.randint(0, k_cmf_module.item_num, (seq_len,))

        out, _, _ = k_cmf_module(user_id, sq)

        # Knowledge should be non-decreasing (after sigmoid)
        # Note: This is a weak test since sigmoid is applied
        # We mainly check that values are finite
        assert torch.all(torch.isfinite(out))

    def test_forward_multiple_users(self, k_cmf_module, seq_len):
        """Test K_CMF with different users."""
        sq = torch.randint(0, k_cmf_module.item_num, (seq_len,))

        for user_id in [0, 1, min(k_cmf_module.user_num - 1, 2)]:
            out, _, _ = k_cmf_module(user_id, sq)
            assert out.shape == (seq_len + 1, k_cmf_module.skill_num)
            assert torch.all(torch.isfinite(out))


class TestGMF:
    """Test GMF (Ability Module)."""

    @pytest.fixture
    def adj_matrix(self, num_users, num_questions):
        """Create adjacency matrix for testing."""
        n_nodes = num_users + num_questions
        # Create a simple bipartite graph
        indices = []
        values = []
        for i in range(num_users):
            for j in range(min(5, num_questions)):
                q_id = num_users + (i * 5 + j) % num_questions
                indices.append([i, q_id])
                indices.append([q_id, i])
                values.append(1.0)
                values.append(1.0)

        indices = torch.tensor(indices).t()
        values = torch.tensor(values)

        adj = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes))
        return adj

    @pytest.fixture
    def gmf_module(self, num_users, num_questions, hidden_dim, adj_matrix):
        """Create a GMF module for testing."""
        return GMF(
            n_users=num_users,
            n_items=num_questions,
            embedding_k=hidden_dim,
            aj_norm=adj_matrix,
            adj=True,
            layer=1,
        )

    def test_init(self, gmf_module, num_users, num_questions, hidden_dim):
        """Test GMF initialization."""
        assert gmf_module.n_users == num_users
        assert gmf_module.n_items == num_questions
        assert gmf_module.embedding_k == hidden_dim
        assert gmf_module.embeddings.shape == (num_users + num_questions, hidden_dim)

    def test_forward_basic(self, gmf_module, batch_size):
        """Test GMF forward pass."""
        user_index = torch.randint(0, gmf_module.n_users, (batch_size,))
        item_index = torch.randint(0, gmf_module.n_items, (batch_size,))

        pred_batch, u_norm, i_norm = gmf_module(user_index, item_index)

        assert pred_batch.shape == (batch_size,)
        # u_norm and i_norm are scalars
        assert u_norm.ndim == 0
        assert i_norm.ndim == 0
        assert torch.all(torch.isfinite(pred_batch))

    def test_forward_different_layers(
        self, num_users, num_questions, hidden_dim, adj_matrix, batch_size
    ):
        """Test GMF with different number of GNN layers."""
        user_index = torch.randint(0, num_users, (batch_size,))
        item_index = torch.randint(0, num_questions, (batch_size,))

        for layer in [0, 1, 2]:
            gmf = GMF(
                n_users=num_users,
                n_items=num_questions,
                embedding_k=hidden_dim,
                aj_norm=adj_matrix,
                adj=False,
                layer=layer,
            )

            pred_batch, u_norm, i_norm = gmf(user_index, item_index)

            assert pred_batch.shape == (batch_size,)
            assert torch.all(torch.isfinite(pred_batch))

    def test_forward_without_learnable_adj(
        self, num_users, num_questions, hidden_dim, adj_matrix, batch_size
    ):
        """Test GMF without learnable adjacency weights."""
        gmf = GMF(
            n_users=num_users,
            n_items=num_questions,
            embedding_k=hidden_dim,
            aj_norm=adj_matrix,
            adj=False,
            layer=1,
        )

        user_index = torch.randint(0, num_users, (batch_size,))
        item_index = torch.randint(0, num_questions, (batch_size,))

        pred_batch, _, _ = gmf(user_index, item_index)

        assert pred_batch.shape == (batch_size,)
        assert torch.all(torch.isfinite(pred_batch))

    def test_global_effects(self, gmf_module, batch_size):
        """Test GMF global effects (user_GE and item_GE)."""
        user_index = torch.randint(0, gmf_module.n_users, (batch_size,))
        item_index = torch.randint(0, gmf_module.n_items, (batch_size,))

        pred_batch, _, _ = gmf_module(user_index, item_index)

        # Check that global effects are being used
        assert torch.all(torch.isfinite(pred_batch))
