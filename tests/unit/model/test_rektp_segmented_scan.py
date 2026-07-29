import torch

from model.ReKTP.segmented_scan import segmented_affine_exclusive_scan


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
