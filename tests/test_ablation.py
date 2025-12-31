#!/usr/bin/env python3
"""
Unit tests for the ablation framework.

Tests all ablation strategies and configuration management.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ablation import (
    load_ablation_config,
    list_strategies,
    validate_ablation_config,
)
from utils.ablation.strategies import (
    ModuleDisableStrategy,
    ModuleZeroStrategy,
    FeatureZeroStrategy,
    ParameterFreezeStrategy,
    apply_ablation,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.conv = nn.Linear(64, 64)
        self.lstm = nn.LSTM(64, 64)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x, _ = self.lstm(x)
        return x


def test_strategy_registry():
    """Test that all strategies are registered."""
    print("\nTesting strategy registry...")

    strategies = list_strategies()

    expected_strategies = [
        "module_disable",
        "module_zero",
        "feature_zero",
        "parameter_freeze",
        "module_replace",
    ]

    for strategy in expected_strategies:
        assert strategy in strategies, f"Strategy '{strategy}' not registered"

    print(f"✓ All strategies registered: {strategies}")


def test_module_disable():
    """Test ModuleDisableStrategy."""
    print("\nTesting ModuleDisableStrategy...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    # Normal forward pass
    output_normal = model(input_ids)
    assert output_normal is not None

    # Disable conv module
    strategy = ModuleDisableStrategy(model, "conv")
    strategy.apply()

    output_disabled = model(input_ids)
    assert output_disabled.shape == output_normal.shape

    # Cleanup
    strategy.cleanup()
    output_restored = model(input_ids)
    assert output_restored.shape == output_normal.shape

    print("✓ ModuleDisableStrategy works correctly")


def test_module_zero():
    """Test ModuleZeroStrategy."""
    print("\nTesting ModuleZeroStrategy...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    # Normal forward pass
    output_normal = model.embedding(input_ids)

    # Zero out embedding output
    strategy = ModuleZeroStrategy(model, "embedding")
    strategy.apply()

    output_zeroed = model.embedding(input_ids)
    assert torch.allclose(output_zeroed, torch.zeros_like(output_normal))

    # Cleanup
    strategy.cleanup()
    output_restored = model.embedding(input_ids)
    assert not torch.allclose(output_restored, torch.zeros_like(output_normal))

    print("✓ ModuleZeroStrategy works correctly")


def test_feature_zero():
    """Test FeatureZeroStrategy."""
    print("\nTesting FeatureZeroStrategy...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    # Zero out first 5 dimensions
    params = {"indices": [0, 1, 2, 3, 4], "dim": -1}
    strategy = FeatureZeroStrategy(model, "embedding", params)
    strategy.apply()

    output_zeroed = model.embedding(input_ids)

    # Check first 5 dims are zero
    assert torch.all(output_zeroed[:, :, :5] == 0)

    # Check remaining dims are non-zero (embedding is non-zero)
    assert torch.any(output_zeroed[:, :, 5:] != 0)

    # Cleanup
    strategy.cleanup()

    print("✓ FeatureZeroStrategy works correctly")


def test_feature_zero_bounds_check():
    """Test FeatureZeroStrategy bounds checking."""
    print("\nTesting FeatureZeroStrategy bounds checking...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    # Test with out of bounds indices
    params = {"indices": [0, 1, 100, 200], "dim": -1}
    strategy = FeatureZeroStrategy(model, "embedding", params)
    strategy.apply()

    try:
        _ = model.embedding(input_ids)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "out of bounds" in str(e)
        print(f"✓ Bounds check works: {e}")

    strategy.cleanup()


def test_parameter_freeze():
    """Test ParameterFreezeStrategy."""
    print("\nTesting ParameterFreezeStrategy...")

    model = SimpleModel()

    # Check initial requires_grad
    lstm_params = list(model.lstm.parameters())
    initial_grads = [p.requires_grad for p in lstm_params]
    assert all(initial_grads)

    # Freeze LSTM
    strategy = ParameterFreezeStrategy(model, "lstm")
    strategy.apply()

    frozen_params = list(model.lstm.parameters())
    frozen_grads = [p.requires_grad for p in frozen_params]
    assert all(not grad for grad in frozen_grads)

    # Cleanup
    strategy.cleanup()
    restored_params = list(model.lstm.parameters())
    restored_grads = [p.requires_grad for p in restored_params]
    assert restored_grads == initial_grads

    print("✓ ParameterFreezeStrategy works correctly")


def test_context_manager():
    """Test apply_ablation context manager."""
    print("\nTesting context manager...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    # Use context manager
    with apply_ablation(model, "module_disable", "conv"):
        output = model(input_ids)
        assert output is not None

    # After context, model works normally
    output = model(input_ids)
    assert output is not None

    print("✓ Context manager works correctly")


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    config_path = "configs/ablation/gikt_ablation.json"

    if not Path(config_path).exists():
        print(f"⚠ Config file not found: {config_path}")
        return

    config = load_ablation_config(config_path)

    assert config.model_name == "GIKT"
    assert config.baseline.name == "full_model"
    assert len(config.ablations) > 0

    # Validate config
    is_valid = validate_ablation_config(config.to_dict())
    assert is_valid

    print("✓ Configuration loading works correctly")


def test_combined_strategies():
    """Test combining multiple strategies."""
    print("\nTesting combined strategies...")

    model = SimpleModel()
    input_ids = torch.tensor([[1, 2, 3]])

    from contextlib import ExitStack

    with ExitStack() as stack:
        # Apply multiple strategies
        stack.enter_context(apply_ablation(model, "module_disable", "conv"))
        stack.enter_context(
            apply_ablation(
                model,
                "feature_zero",
                "embedding",
                params={"indices": [0, 1, 2], "dim": -1},
            )
        )

        output = model(input_ids)
        assert output is not None

    print("✓ Combined strategies work correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Ablation Framework Unit Tests")
    print("=" * 60)

    tests = [
        test_strategy_registry,
        test_module_disable,
        test_module_zero,
        test_feature_zero,
        test_feature_zero_bounds_check,
        test_parameter_freeze,
        test_context_manager,
        test_config_loading,
        test_combined_strategies,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Tests passed: {passed}/{len(tests)}")
    if failed > 0:
        print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
