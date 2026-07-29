import torch

from model.ReKTP.segmented_scan import (
    segmented_affine_exclusive_scan,
    segmented_normalized_evidence_exclusive_scan,
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


def _serial_evidence_scan(
    decay,
    write_mass,
    candidate,
    segment_ids,
    valid_mask,
    prior_state,
    prior_mass=1.0,
):
    result = torch.zeros_like(candidate)
    for batch in range(decay.size(0)):
        states = {}
        for position in range(decay.size(1)):
            if not valid_mask[batch, position]:
                continue
            segment = int(segment_ids[batch, position])
            evidence, mass = states.get(
                segment,
                (
                    torch.zeros_like(candidate[batch, position]),
                    torch.zeros_like(write_mass[batch, position]),
                ),
            )
            evidence = decay[batch, position] * evidence
            mass = decay[batch, position] * mass
            result[batch, position] = (
                prior_mass * prior_state[batch, position] + evidence
            ) / (prior_mass + mass)
            states[segment] = (
                evidence
                + write_mass[batch, position] * candidate[batch, position],
                mass + write_mass[batch, position],
            )
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


def _evidence_example():
    alpha, _, segment_ids, valid_mask, initial = _example()
    torch.manual_seed(11)
    write_mass = torch.sigmoid(torch.randn_like(alpha))
    candidate = torch.tanh(torch.randn_like(alpha))
    prior_state = torch.tanh(initial)
    return (
        alpha,
        write_mass,
        candidate,
        segment_ids,
        valid_mask,
        prior_state,
    )


def test_normalized_evidence_scan_matches_serial_reference():
    args = _evidence_example()
    actual = segmented_normalized_evidence_exclusive_scan(*args)
    expected = _serial_evidence_scan(*args)
    torch.testing.assert_close(actual, expected)


def test_normalized_evidence_state_is_bounded():
    args = _evidence_example()
    state = segmented_normalized_evidence_exclusive_scan(*args)
    valid_mask = args[4]
    assert torch.all(state[valid_mask].abs() <= 1.0)


def test_current_write_is_exclusive_but_current_decay_affects_read():
    decay, write_mass, candidate, segment_ids, valid_mask, prior_state = (
        _evidence_example()
    )
    baseline = segmented_normalized_evidence_exclusive_scan(
        decay,
        write_mass,
        candidate,
        segment_ids,
        valid_mask,
        prior_state,
    )

    changed_write = write_mass.clone()
    changed_candidate = candidate.clone()
    changed_write[:, 1] = 0.99
    changed_candidate[:, 1] = -1.0
    after_write_change = segmented_normalized_evidence_exclusive_scan(
        decay,
        changed_write,
        changed_candidate,
        segment_ids,
        valid_mask,
        prior_state,
    )
    torch.testing.assert_close(after_write_change[:, 1], baseline[:, 1])
    assert not torch.allclose(after_write_change[:, 2], baseline[:, 2])

    changed_decay = decay.clone()
    changed_decay[:, 1] = 0.01
    after_decay_change = segmented_normalized_evidence_exclusive_scan(
        changed_decay,
        write_mass,
        candidate,
        segment_ids,
        valid_mask,
        prior_state,
    )
    assert not torch.allclose(after_decay_change[:, 1], baseline[:, 1])
