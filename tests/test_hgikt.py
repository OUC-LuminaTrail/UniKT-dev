"""Tests for HGIKT model components."""

import pytest
import torch
from torch_geometric.data import HeteroData

from model.HGIKT.HGIKT_model import HGIKT, HeteroGNN, HyperGNN, MoEFusion


class TestHeteroGNN:
    """Test HeteroGNN module."""

    @pytest.fixture
    def hetero_metadata(self):
        """Create metadata for heterogeneous graph."""
        return (
            ["question", "skill", "assignment", "template"],
            [
                ("question", "has", "skill"),
                ("skill", "rev_has", "question"),
                ("question", "belongs_to", "assignment"),
                ("assignment", "rev_belongs_to", "question"),
                ("question", "uses", "template"),
                ("template", "rev_uses", "question"),
            ],
        )

    @pytest.fixture
    def hetero_graph(self, num_questions, num_skills):
        """Create heterogeneous graph for testing."""
        data = HeteroData()

        num_edges = 100
        question_ids = torch.randint(0, num_questions, (num_edges,))
        skill_ids = torch.randint(0, num_skills, (num_edges,))

        data["question", "has", "skill"].edge_index = torch.stack(
            [question_ids, skill_ids]
        )
        data["skill", "rev_has", "question"].edge_index = torch.stack(
            [skill_ids, question_ids]
        )

        # Add assignment and template nodes
        num_assignments = 10
        num_templates = 5
        data["question"].node_id = torch.arange(num_questions)
        data["skill"].node_id = torch.arange(num_skills)
        data["assignment"].node_id = torch.arange(num_assignments)
        data["template"].node_id = torch.arange(num_templates)

        # Create edges for assignment
        assignment_ids = torch.randint(0, num_assignments, (num_edges,))
        data["question", "belongs_to", "assignment"].edge_index = torch.stack(
            [question_ids, assignment_ids]
        )
        data["assignment", "rev_belongs_to", "question"].edge_index = torch.stack(
            [assignment_ids, question_ids]
        )

        # Create edges for template
        template_ids = torch.randint(0, num_templates, (num_edges,))
        data["question", "uses", "template"].edge_index = torch.stack(
            [question_ids, template_ids]
        )
        data["template", "rev_uses", "question"].edge_index = torch.stack(
            [template_ids, question_ids]
        )

        return data

    @pytest.fixture
    def hetero_gnn(self, hidden_dim, hetero_metadata):
        """Create a HeteroGNN module for testing."""
        return HeteroGNN(
            embedding_dim=hidden_dim,
            n_hop=2,
            heads=4,
            dropout=0.1,
            metadata=hetero_metadata,
        )

    def test_init(self, hetero_gnn, hidden_dim):
        """Test HeteroGNN initialization."""
        assert hetero_gnn.n_hop == 2
        assert hetero_gnn.heads == 4
        assert len(hetero_gnn.convs) == 2

    def test_forward_basic(
        self, hetero_gnn, hetero_graph, hidden_dim, num_questions, num_skills
    ):
        """Test HeteroGNN forward pass."""
        num_assignments = 10
        num_templates = 5
        x_dict = {
            "question": torch.randn(num_questions, hidden_dim),
            "skill": torch.randn(num_skills, hidden_dim),
            "assignment": torch.randn(num_assignments, hidden_dim),
            "template": torch.randn(num_templates, hidden_dim),
        }

        output = hetero_gnn(x_dict, hetero_graph.edge_index_dict)

        assert "question" in output
        assert "skill" in output
        assert "assignment" in output
        assert "template" in output
        assert output["question"].shape == (num_questions, hidden_dim)
        assert output["skill"].shape == (num_skills, hidden_dim)


class TestHyperGNN:
    """Test HyperGNN module."""

    @pytest.fixture
    def hypergraph(self, num_questions):
        """Create a simple hypergraph for testing."""
        from dhg import Hypergraph

        # Create a hypergraph with random hyperedges
        num_edges = 10
        edge_list = []
        for _ in range(num_edges):
            edge_size = torch.randint(2, 5, (1,)).item()
            vertices = torch.randperm(num_questions)[:edge_size].tolist()
            edge_list.append(vertices)

        return Hypergraph(num_questions, edge_list)

    @pytest.fixture
    def hyper_gnn(self, hidden_dim):
        """Create a HyperGNN module for testing."""
        return HyperGNN(
            in_ch=hidden_dim,
            n_hid=hidden_dim,
            n_class=hidden_dim,
            dropout=0.1,
            use_edge_weights=True,
        )

    def test_init(self, hyper_gnn, hidden_dim):
        """Test HyperGNN initialization."""
        assert hasattr(hyper_gnn, "hgc1")
        assert hasattr(hyper_gnn, "hgc2")

    def test_forward_basic(self, hyper_gnn, hypergraph, num_questions, hidden_dim):
        """Test HyperGNN forward pass."""
        x = torch.randn(num_questions, hidden_dim)

        output = hyper_gnn(x, hypergraph)

        assert output.shape == (num_questions, hidden_dim)
        assert torch.all(torch.isfinite(output))


class TestMoEFusion:
    """Test MoEFusion (Mixture-of-Experts Fusion) module."""

    @pytest.fixture
    def moe_fusion(self, hidden_dim):
        """Create a MoEFusion module for testing."""
        return MoEFusion(dim=hidden_dim, dropout=0.1)

    def test_init(self, moe_fusion, hidden_dim):
        """Test MoEFusion initialization."""
        assert moe_fusion.dim == hidden_dim
        assert hasattr(moe_fusion, "expert1")
        assert hasattr(moe_fusion, "expert2")
        assert hasattr(moe_fusion, "expert_shared")
        assert hasattr(moe_fusion, "router")

    def test_forward_basic(self, moe_fusion, batch_size, seq_len, hidden_dim):
        """Test MoEFusion forward pass."""
        view1 = torch.randn(batch_size, seq_len, hidden_dim)
        view2 = torch.randn(batch_size, seq_len, hidden_dim)

        output = moe_fusion(view1, view2)

        assert output.shape == (batch_size, seq_len, hidden_dim)
        assert torch.all(torch.isfinite(output))

    def test_forward_different_shapes(self, moe_fusion, hidden_dim):
        """Test MoEFusion with different input shapes."""
        # Test with 1D input
        view1 = torch.randn(hidden_dim)
        view2 = torch.randn(hidden_dim)

        output = moe_fusion(view1, view2)

        assert output.shape == (hidden_dim,)
        assert torch.all(torch.isfinite(output))


class TestHGIKT:
    """Test HGIKT model."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args for HGIKT model."""

        class MockArgs:
            hidden_dim = 32
            lstm_layers = 2
            dropout = 0.1
            n_hop = 2
            heads = 4
            history_neighbour = 3
            att_bound = 0.7

        return MockArgs()

    @pytest.fixture
    def data_metadata(self, num_questions, num_skills):
        """Create data metadata for HGIKT model."""
        return {
            "num_questions": num_questions,
            "num_skills": num_skills,
            "num_assignments": 10,
            "num_templates": 5,
        }

    @pytest.fixture
    def hetero_metadata(self, num_questions, num_skills):
        """Create metadata for heterogeneous graph."""
        return (
            ["question", "skill", "assignment", "template"],
            [
                ("question", "has", "skill"),
                ("skill", "rev_has", "question"),
                ("question", "belongs_to", "assignment"),
                ("assignment", "rev_belongs_to", "question"),
                ("question", "uses", "template"),
                ("template", "rev_uses", "question"),
            ],
        )

    @pytest.fixture
    def hetero_graph(self, num_questions, num_skills):
        """Create heterogeneous graph for HGIKT."""
        data = HeteroData()

        num_edges = 100
        question_ids = torch.randint(0, num_questions, (num_edges,))
        skill_ids = torch.randint(0, num_skills, (num_edges,))

        data["question", "has", "skill"].edge_index = torch.stack(
            [question_ids, skill_ids]
        )
        data["skill", "rev_has", "question"].edge_index = torch.stack(
            [skill_ids, question_ids]
        )

        num_assignments = 10
        num_templates = 5
        assignment_ids = torch.randint(0, num_assignments, (num_edges,))
        template_ids = torch.randint(0, num_templates, (num_edges,))

        data["question", "belongs_to", "assignment"].edge_index = torch.stack(
            [question_ids, assignment_ids]
        )
        data["assignment", "rev_belongs_to", "question"].edge_index = torch.stack(
            [assignment_ids, question_ids]
        )

        data["question", "uses", "template"].edge_index = torch.stack(
            [question_ids, template_ids]
        )
        data["template", "rev_uses", "question"].edge_index = torch.stack(
            [template_ids, question_ids]
        )

        return data

    @pytest.fixture
    def hypergraph(self, num_questions):
        """Create a hypergraph for HGIKT."""
        from dhg import Hypergraph

        num_edges = 10
        edge_list = []
        for _ in range(num_edges):
            edge_size = torch.randint(2, 5, (1,)).item()
            vertices = torch.randperm(num_questions)[:edge_size].tolist()
            edge_list.append(vertices)

        return Hypergraph(num_questions, edge_list)

    @pytest.fixture
    def question_skill_matrix(self, num_questions, num_skills):
        """Create question-skill matrix."""
        return torch.randint(0, 2, (num_questions, num_skills)).float()

    @pytest.fixture
    def hgikt_model(self, mock_args, data_metadata, hetero_metadata):
        """Create a HGIKT model for testing."""
        return HGIKT(mock_args, data_metadata, hetero_metadata)

    def test_init(self, hgikt_model, mock_args, num_questions, num_skills):
        """Test HGIKT initialization."""
        assert hgikt_model.hidden_dim == mock_args.hidden_dim
        assert hgikt_model.lstm_layers == mock_args.lstm_layers
        assert hasattr(hgikt_model, "question_embedding")
        assert hasattr(hgikt_model, "skill_embedding")
        assert hasattr(hgikt_model, "assignment_embedding")
        assert hasattr(hgikt_model, "template_embedding")
        assert hasattr(hgikt_model, "hetero_conv")
        assert hasattr(hgikt_model, "hgnn_conv")
        assert hasattr(hgikt_model, "fuse")

    def test_forward_basic(
        self,
        hgikt_model,
        batch_size,
        seq_len,
        hetero_graph,
        hypergraph,
        question_skill_matrix,
    ):
        """Test HGIKT forward pass with basic inputs."""
        user_sequence = torch.randint(
            0, hgikt_model.data_metadata["num_questions"], (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = hgikt_model(
            user_sequence,
            user_response,
            user_mask,
            hetero_graph,
            hypergraph,
            question_skill_matrix,
        )

        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))

    def test_forward_with_mask(
        self,
        hgikt_model,
        batch_size,
        seq_len,
        hetero_graph,
        hypergraph,
        question_skill_matrix,
    ):
        """Test HGIKT forward pass with masked positions."""
        user_sequence = torch.randint(
            0, hgikt_model.data_metadata["num_questions"], (batch_size, seq_len)
        )
        user_response = torch.randint(0, 2, (batch_size, seq_len))
        # Create mask with some invalid positions
        user_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        user_mask[:, seq_len // 2 :] = False

        output = hgikt_model(
            user_sequence,
            user_response,
            user_mask,
            hetero_graph,
            hypergraph,
            question_skill_matrix,
        )

        assert output.shape == (batch_size, seq_len)
        assert torch.all(torch.isfinite(output))
