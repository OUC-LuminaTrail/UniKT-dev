"""Tests for InferenceOpsMixin: equivalence with BaseTrainer and helper math."""

import pytest
import torch

from utils.training import BaseTrainer, InferenceOpsMixin


class _Host(InferenceOpsMixin):
    def __init__(self):
        self.device_ = torch.device("cpu")


def test_mixin_methods_shared_with_base_trainer():
    for name in (
        "_try_gpu",
        "_move_tensor_to_device",
        "_extract_valid_predictions",
        "_pad_to_full_sequence",
        "_handle_empty_batch",
        "_generate_binary_predictions",
    ):
        assert getattr(InferenceOpsMixin, name) is getattr(BaseTrainer, name), name


def test_mixin_usable_without_trainer():
    host = _Host()
    assert host._move_tensor_to_device(torch.zeros(1)).device.type == "cpu"
    assert torch.equal(
        host._generate_binary_predictions(torch.tensor([0.5, -0.3])),
        torch.tensor([1, 0]),
    )


def test_extract_valid_predictions_next_item_alignment():
    host = _Host()
    y_hat_full = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    response = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    y_hat, y_label, valid = host._extract_valid_predictions(y_hat_full, response, mask)

    assert y_hat.tolist() == [1.0, 2.0, 5.0, 6.0, 7.0]
    assert y_label.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert valid.tolist() == [[True, True, False], [True, True, True]]


def test_extract_valid_predictions_same_position_normalizes():
    host = _Host()
    # same-position output: out[t] predicts response[t]; first column is padding
    same_pos = torch.tensor([[9.0, 1.0, 2.0, 3.0], [9.0, 5.0, 6.0, 7.0]])
    response = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)

    y_hat, y_label, _ = host._extract_valid_predictions(
        same_pos, response, mask, same_position=True
    )

    assert y_hat.tolist() == [1.0, 2.0, 5.0, 6.0, 7.0]
    assert y_label.tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_pad_to_full_sequence_shape_and_zeros():
    host = _Host()
    out = host._pad_to_full_sequence(torch.ones(2, 3))
    assert out.shape == (2, 4)
    assert torch.equal(out[:, -1], torch.zeros(2))


def test_handle_empty_batch_raises():
    host = _Host()
    with pytest.raises(ValueError, match="Empty valid targets"):
        host._handle_empty_batch(torch.zeros(0), torch.zeros(0))
