"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest
import torch


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset random seed before each test."""
    set_random_seed(42)
    yield


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Default sequence length for tests."""
    return 10


@pytest.fixture
def hidden_dim():
    """Default hidden dimension for tests."""
    return 32


@pytest.fixture
def embedding_dim():
    """Default embedding dimension for tests."""
    return 64


@pytest.fixture
def num_skills():
    """Default number of skills for tests."""
    return 20


@pytest.fixture
def num_questions():
    """Default number of questions for tests."""
    return 50


@pytest.fixture
def num_users():
    """Default number of users for tests."""
    return 30


@pytest.fixture
def hist_neighbor_num():
    """Default number of historical neighbors for tests."""
    return 3
