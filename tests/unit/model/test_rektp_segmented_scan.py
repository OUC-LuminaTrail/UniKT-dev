import torch

from model.ReKTP.segmented_scan import (
    segmented_affine_exclusive_scan,
    segmented_block_affine_exclusive_scan,
)


def _serial_scan(alpha, beta, segment_ids, valid_mask, initial_state):
    result = torch.zeros_like(alpha)
    for batch in range(alpha.size(0)):
        states = {}
        for position in range(alpha.size(1)):
            if not valid_mask[batch, position]:
                continue
            segment = int(segment_ids[batch, position])
            state = states.get(segment, initial_state[batch, position])
            result[batch, position] = state
            states[segment] = alpha[batch, position] * state + beta[batch, position]
    return result


def _example():
    torch.manual_seed(7)
    segment_ids = torch.tensor([[0, 0, 0, 2, 2, 5, 5, 5, 9]])
    valid_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    alpha = torch.sigmoid(torch.randn(1, 9, 4))
    beta = torch.randn(1, 9, 4)
    initial = torch.randn(1, 9, 4)
    initial[:, 1:3] = initial[:, :1]
    initial[:, 4:5] = initial[:, 3:4]
    initial[:, 6:8] = initial[:, 5:6]
    return alpha, beta, segment_ids, valid_mask, initial


def _serial_block_scan(matrix, bias, segment_ids, valid_mask, initial_state):
    result = torch.zeros_like(bias)
    for batch in range(matrix.size(0)):
        states = {}
        for position in range(matrix.size(1)):
            if not valid_mask[batch, position]:
                continue
            segment = int(segment_ids[batch, position])
            state = states.get(segment, initial_state[batch, position])
            result[batch, position] = state
            states[segment] = (matrix[batch, position] @ state.unsqueeze(-1)).squeeze(
                -1
            ) + bias[batch, position]
    return result


def _block_example():
    torch.manual_seed(11)
    segment_ids = torch.tensor([[0, 0, 0, 2, 2, 5, 5, 5, 9]])
    valid_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    identity = torch.eye(2).view(1, 1, 1, 2, 2)
    matrix = identity + 0.1 * torch.randn(1, 9, 3, 2, 2)
    bias = torch.randn(1, 9, 3, 2)
    initial = torch.randn(1, 9, 3, 2)
    initial[:, 1:3] = initial[:, :1]
    initial[:, 4:5] = initial[:, 3:4]
    initial[:, 6:8] = initial[:, 5:6]
    return matrix, bias, segment_ids, valid_mask, initial


def test_segmented_scan_matches_serial_reference():
    alpha, beta, segment_ids, valid_mask, initial = _example()
    actual = segmented_affine_exclusive_scan(
        alpha, beta, segment_ids, valid_mask, initial
    )
    expected = _serial_scan(alpha, beta, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected)


def test_segment_changes_do_not_cross_kc_boundaries():
    alpha, beta, segment_ids, valid_mask, initial = _example()
    baseline = segmented_affine_exclusive_scan(
        alpha, beta, segment_ids, valid_mask, initial
    )
    changed_beta = beta.clone()
    changed_beta[:, :3] += 100.0
    changed = segmented_affine_exclusive_scan(
        alpha, changed_beta, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 3:], baseline[:, 3:])


def test_exclusive_state_does_not_depend_on_current_transition():
    alpha, beta, segment_ids, valid_mask, initial = _example()
    baseline = segmented_affine_exclusive_scan(
        alpha, beta, segment_ids, valid_mask, initial
    )
    changed_alpha = alpha.clone()
    changed_beta = beta.clone()
    changed_alpha[:, 1] = 0.01
    changed_beta[:, 1] = 99.0
    changed = segmented_affine_exclusive_scan(
        changed_alpha, changed_beta, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_block_scan_matches_serial_reference():
    matrix, bias, segment_ids, valid_mask, initial = _block_example()
    actual = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _serial_block_scan(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected)


def test_block_scan_does_not_cross_segment_boundaries():
    matrix, bias, segment_ids, valid_mask, initial = _block_example()
    baseline = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_bias = bias.clone()
    changed_bias[:, :3] += 100.0
    changed = segmented_block_affine_exclusive_scan(
        matrix, changed_bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 3:], baseline[:, 3:])


def test_block_scan_excludes_current_noncommuting_transition():
    matrix, bias, segment_ids, valid_mask, initial = _block_example()
    baseline = segmented_block_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_matrix = matrix.clone()
    changed_bias = bias.clone()
    changed_matrix[:, 1] = torch.tensor([[0.0, 2.0], [-1.0, 0.5]])
    changed_bias[:, 1] = 99.0
    changed = segmented_block_affine_exclusive_scan(
        changed_matrix, changed_bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_block_scan_supports_pure_additive_write_without_erasure():
    identity = torch.eye(2).view(1, 1, 1, 2, 2).expand(1, 4, 2, 2, 2)
    bias = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 10.0
    segment_ids = torch.zeros(1, 4, dtype=torch.long)
    valid_mask = torch.ones(1, 4, dtype=torch.bool)
    initial = torch.randn(1, 4, 2, 2)
    initial[:] = initial[:, :1]

    actual = segmented_block_affine_exclusive_scan(
        identity, bias, segment_ids, valid_mask, initial
    )
    expected = _serial_block_scan(identity, bias, segment_ids, valid_mask, initial)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[:, 1], initial[:, 1] + bias[:, 0])


def test_block_scan_gradients_match_serial_reference():
    matrix, bias, segment_ids, valid_mask, initial = _block_example()
    weights = torch.randn_like(bias)

    parallel_matrix = matrix.clone().requires_grad_()
    parallel_bias = bias.clone().requires_grad_()
    parallel = segmented_block_affine_exclusive_scan(
        parallel_matrix,
        parallel_bias,
        segment_ids,
        valid_mask,
        initial,
    )
    parallel_gradients = torch.autograd.grad(
        (parallel * weights).sum(), (parallel_matrix, parallel_bias)
    )

    serial_matrix = matrix.clone().requires_grad_()
    serial_bias = bias.clone().requires_grad_()
    serial = _serial_block_scan(
        serial_matrix,
        serial_bias,
        segment_ids,
        valid_mask,
        initial,
    )
    serial_gradients = torch.autograd.grad(
        (serial * weights).sum(), (serial_matrix, serial_bias)
    )

    torch.testing.assert_close(parallel, serial)
    for parallel_gradient, serial_gradient in zip(
        parallel_gradients, serial_gradients, strict=True
    ):
        torch.testing.assert_close(parallel_gradient, serial_gradient)
