"""Tests for the 1x1 (scalar) Triton segmented scan in ReKTP.

Semantics of the scalar kernels (``_fwd_kernel_scalar`` /
``_bwd_adj_kernel_scalar`` / ``_bwd_dinit_kernel_scalar``):

- exclusive: the output at position ``n`` is the state *before* applying the
  position's operator;
- segment heads (row start, after an invalid position, or a new segment id)
  reset the carry to the identity, so a head outputs its own initial state;
- the initial state is used per position (``carry @ init_n + carry_bias``),
  not only at segment heads;
- invalid positions output zero and break the segment run.

The pure-PyTorch ``_scalar_scan_reference`` mirrors these semantics, so
``torch.autograd`` covers its backward pass; the Triton results are checked
against it both forward and backward.
"""

import pytest
import torch

from model.ReKTP.triton_scan import segmented_scalar_affine_exclusive_scan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ReKTP segmented scan requires CUDA"
)

DEVICE = torch.device("cuda")


def _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial_state):
    """Serial scan matching the Triton kernel semantics exactly.

    Pure PyTorch ops, so autograd covers the backward pass.
    """
    batch, length, heads, _, _ = matrix.shape
    result = torch.zeros_like(bias)
    for b in range(batch):
        carry_a = torch.ones(heads, 1, device=matrix.device, dtype=matrix.dtype)
        carry_b = torch.zeros(heads, 1, device=matrix.device, dtype=matrix.dtype)
        prev_valid = False
        prev_seg = -1
        for n in range(length):
            v = bool(valid_mask[b, n])
            s = int(segment_ids[b, n])
            if v and (n == 0 or not prev_valid or s != prev_seg):
                carry_a = torch.ones(heads, 1, device=matrix.device, dtype=matrix.dtype)
                carry_b = torch.zeros(
                    heads, 1, device=matrix.device, dtype=matrix.dtype
                )
            if v:
                result[b, n] = carry_a * initial_state[b, n] + carry_b
                a = matrix[b, n].squeeze(-1)  # [heads, 1]
                carry_a = a * carry_a
                carry_b = a * carry_b + bias[b, n]
            prev_valid = v
            prev_seg = s
    return result


def _random_case(
    batch,
    length,
    heads,
    seed,
    valid_density=0.7,
    num_segments=4,
):
    """Random 1x1 scan inputs on CUDA with a fixed seed."""
    torch.manual_seed(seed)
    segment_ids = torch.randint(0, num_segments, (batch, length), device=DEVICE)
    valid_mask = torch.rand(batch, length, device=DEVICE) < valid_density
    matrix = torch.randn(batch, length, heads, 1, 1, device=DEVICE)
    bias = torch.randn(batch, length, heads, 1, device=DEVICE)
    initial = torch.randn(batch, length, heads, 1, device=DEVICE)
    return matrix, bias, segment_ids, valid_mask, initial


def _make_segment_constant_init(initial, segment_ids, valid_mask):
    """Copy each segment run's first valid init to all valid positions in it.

    ``d matrix`` / ``d bias`` of the Triton backward are exact only while the
    initial state is constant within a segment; this helper builds such
    inputs while keeping the per-position init semantics for the forward.
    """
    result = initial.clone()
    for b in range(initial.size(0)):
        seen = {}
        for n in range(initial.size(1)):
            if not valid_mask[b, n]:
                seen = {}
                continue
            s = int(segment_ids[b, n])
            if s not in seen:
                seen[s] = initial[b, n].clone()
            result[b, n] = seen[s]
    return result


# --------------------------------------------------------------------------
# forward correctness
# --------------------------------------------------------------------------


def test_scalar_scan_matches_hand_computed_single_segment():
    matrix = torch.tensor(
        [[[[[0.5]]], [[[2.0]]], [[[1.5]]], [[[0.25]]]]], device=DEVICE
    )
    bias = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]], device=DEVICE)
    initial = torch.full((1, 4, 1, 1), 10.0, device=DEVICE)
    segment_ids = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
    valid_mask = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)

    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    # out0 = init = 10
    # out1 = a0*init + b0 = 0.5*10 + 1 = 6
    # out2 = a1*out1 + b1 = 2*6 + 2 = 14
    # out3 = a2*out2 + b2 = 1.5*14 + 3 = 24
    expected = torch.tensor([10.0, 6.0, 14.0, 24.0], device=DEVICE)
    torch.testing.assert_close(actual.flatten(), expected)


def test_scalar_scan_matches_hand_computed_invalid_break():
    matrix = torch.tensor(
        [[[[[0.5]]], [[[2.0]]], [[[1.5]]], [[[0.25]]], [[[3.0]]]]], device=DEVICE
    )
    bias = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]]], device=DEVICE)
    initial = torch.tensor(
        [[[[10.0]], [[20.0]], [[30.0]], [[40.0]], [[50.0]]]], device=DEVICE
    )
    segment_ids = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
    valid_mask = torch.tensor([[True, False, True, True, True]], device=DEVICE)

    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    # n0: head -> 10; n1: invalid -> 0
    # n2: head (prior invalid) -> init2 = 30
    # n3: carry = (a2, b2) -> a2*init3 + b2 = 1.5*40 + 3 = 63
    # n4: carry = (a3*a2, a3*b2 + b3) = (0.375, 4.75) -> 0.375*50 + 4.75 = 23.5
    expected = torch.tensor([10.0, 0.0, 30.0, 63.0, 23.5], device=DEVICE)
    torch.testing.assert_close(actual.flatten(), expected)


@pytest.mark.parametrize(
    ("batch", "length", "heads"),
    [
        (1, 1, 1),
        (2, 5, 3),
        (3, 7, 64),
        (1, 64, 128),
        (4, 9, 17),
        (2, 33, 1),
        (1, 16, 2),
    ],
)
@pytest.mark.parametrize("valid_density", [1.0, 0.6])
@pytest.mark.parametrize("seed", [3, 17])
def test_scalar_scan_matches_reference_random(
    batch, length, heads, valid_density, seed
):
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        batch, length, heads, seed, valid_density=valid_density
    )
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_scalar_scan_all_invalid_outputs_zeros():
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, 8, 4, seed=9, valid_density=0.0
    )
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(actual, torch.zeros_like(bias))


def test_scalar_scan_empty_sequence():
    matrix, bias, segment_ids, valid_mask, initial = _random_case(2, 0, 4, seed=1)
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(actual, torch.zeros_like(initial))


# --------------------------------------------------------------------------
# semantics
# --------------------------------------------------------------------------


def test_scalar_scan_excludes_current_noncommuting_transition():
    # Single segment, all valid: changing position 1's operator must not
    # change position 1's output (exclusive), only position 2's.
    matrix = torch.randn(1, 4, 2, 1, 1, device=DEVICE)
    bias = torch.randn(1, 4, 2, 1, device=DEVICE)
    initial = torch.randn(1, 4, 2, 1, device=DEVICE)
    segment_ids = torch.zeros(1, 4, dtype=torch.long, device=DEVICE)
    valid_mask = torch.ones(1, 4, dtype=torch.bool, device=DEVICE)
    baseline = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_matrix = matrix.clone()
    changed_bias = bias.clone()
    changed_matrix[:, 1] = torch.full((1, 2, 1, 1), -7.0, device=DEVICE)
    changed_bias[:, 1] = 99.0
    changed = segmented_scalar_affine_exclusive_scan(
        changed_matrix, changed_bias, segment_ids, valid_mask, initial
    )
    # Position 1's output is the state before its own operator.
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_scalar_scan_does_not_cross_segment_boundaries():
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        1, 9, 3, seed=11, valid_density=1.0, num_segments=4
    )
    baseline = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    changed_bias = bias.clone()
    changed_bias[:, :3] += 100.0
    changed = segmented_scalar_affine_exclusive_scan(
        matrix, changed_bias, segment_ids, valid_mask, initial
    )
    # Only positions sharing the first segment's run may change.
    same_run = (segment_ids[:, 3:] == segment_ids[:, 2]) & valid_mask[:, 3:]
    torch.testing.assert_close(changed[:, 3:][~same_run], baseline[:, 3:][~same_run])


def test_scalar_scan_invalid_positions_break_segments():
    # Same segment id on both sides of an invalid position: the position after
    # the gap is a segment head and starts from its own initial state.
    matrix = torch.randn(1, 5, 2, 1, 1, device=DEVICE)
    bias = torch.randn(1, 5, 2, 1, device=DEVICE)
    initial = torch.randn(1, 5, 2, 1, device=DEVICE)
    segment_ids = torch.zeros(1, 5, dtype=torch.long, device=DEVICE)
    valid_mask = torch.tensor([[True, True, False, True, True]], device=DEVICE)

    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    # The position after the gap outputs its own initial state.
    torch.testing.assert_close(actual[:, 3], initial[:, 3])


def test_scalar_scan_uses_per_position_initial_state():
    # Single segment, all valid, per-position initial states: the output at
    # position n is carry_n @ init_n + carry_bias_n, so init participates at
    # every position, not just at the segment head.
    matrix = torch.tensor([[[[[0.5]]], [[[2.0]]], [[[1.5]]]]], device=DEVICE)
    bias = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]], device=DEVICE)
    initial = torch.tensor([[[[10.0]], [[20.0]], [[30.0]]]], device=DEVICE)
    segment_ids = torch.zeros(1, 3, dtype=torch.long, device=DEVICE)
    valid_mask = torch.ones(1, 3, dtype=torch.bool, device=DEVICE)

    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    # out0 = init0 = 10
    # out1 = a0*init1 + b0 = 0.5*20 + 1 = 11
    # out2 = (a1*a0)*init2 + (a1*b0 + b1) = 1*30 + 4 = 34
    expected = torch.tensor([10.0, 11.0, 34.0], device=DEVICE)
    torch.testing.assert_close(actual.flatten(), expected)
    # A segment-head-only reading would give out2 = 1*10 + 4 = 14.
    assert not torch.allclose(
        actual.flatten(), torch.tensor([10.0, 11.0, 14.0], device=DEVICE)
    )


def test_scalar_scan_is_deterministic():
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, 12, 8, seed=21, valid_density=0.5
    )
    first = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    second = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    torch.testing.assert_close(first, second)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


def _grads_of(scan_fn, matrix, bias, initial, segment_ids, valid_mask, weights):
    matrix = matrix.clone().requires_grad_()
    bias = bias.clone().requires_grad_()
    initial = initial.clone().requires_grad_()
    out = scan_fn(matrix, bias, segment_ids, valid_mask, initial)
    (out * weights).sum().backward()
    return (
        matrix.grad.detach(),
        bias.grad.detach(),
        initial.grad.detach(),
    ), out.detach()


@pytest.mark.parametrize(
    ("batch", "length", "heads"),
    [(2, 5, 3), (1, 17, 7), (3, 9, 64)],
)
@pytest.mark.parametrize("valid_density", [1.0, 0.75])
def test_scalar_scan_gradients_match_autograd_reference(
    batch, length, heads, valid_density
):
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        batch,
        length,
        heads,
        seed=13,
        valid_density=valid_density,
        num_segments=2,
    )
    # Force the first row to a single fully-valid segment so every operator
    # except the last has a successor; otherwise a sparse random case can
    # leave the whole matrix/bias tensor without any gradient at all.
    segment_ids[0] = 0
    valid_mask[0] = True
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    (dmat, dbias, dinit), out = _grads_of(
        segmented_scalar_affine_exclusive_scan,
        matrix,
        bias,
        initial,
        segment_ids,
        valid_mask,
        weights,
    )
    (rmat, rbias, rinit), rout = _grads_of(
        _scalar_scan_reference,
        matrix,
        bias,
        initial,
        segment_ids,
        valid_mask,
        weights,
    )

    torch.testing.assert_close(out, rout, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dmat, rmat, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dbias, rbias, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dinit, rinit, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("valid_density", [1.0, 0.6])
def test_scalar_scan_gradcheck(valid_density):
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        1, 5, 2, seed=29, valid_density=valid_density
    )
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    def loss_fn(m, b, init):
        out = segmented_scalar_affine_exclusive_scan(
            m, b, segment_ids, valid_mask, init
        )
        return (out * weights).sum()

    torch.autograd.gradcheck(
        loss_fn,
        (matrix.requires_grad_(), bias.requires_grad_(), initial.requires_grad_()),
        eps=1e-2,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_scalar_scan_finite_difference_gradients():
    """Independent check of the adjoint kernels via central differences."""
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        1, 4, 2, seed=31, valid_density=1.0, num_segments=2
    )
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)
    eps = 1e-2

    def loss(m, b, init):
        out = segmented_scalar_affine_exclusive_scan(
            m, b, segment_ids, valid_mask, init
        )
        return (out * weights).sum()

    def central_diff(fn, tensors, index):
        t = tensors[index]
        grad = torch.zeros_like(t)
        flat = t.flatten()
        for i in range(flat.numel()):
            plus = t.clone().flatten()
            minus = t.clone().flatten()
            plus[i] += eps
            minus[i] -= eps
            tensors_plus = list(tensors)
            tensors_minus = list(tensors)
            tensors_plus[index] = plus.reshape_as(t)
            tensors_minus[index] = minus.reshape_as(t)
            grad.flatten()[i] = (fn(*tensors_plus) - fn(*tensors_minus)) / (2.0 * eps)
        return grad

    (dmat, dbias, dinit), _ = _grads_of(
        segmented_scalar_affine_exclusive_scan,
        matrix,
        bias,
        initial,
        segment_ids,
        valid_mask,
        weights,
    )
    num_dmat = central_diff(loss, (matrix, bias, initial), 0)
    num_dbias = central_diff(loss, (matrix, bias, initial), 1)
    num_dinit = central_diff(loss, (matrix, bias, initial), 2)

    torch.testing.assert_close(dmat, num_dmat, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(dbias, num_dbias, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(dinit, num_dinit, rtol=1e-2, atol=1e-3)


# --------------------------------------------------------------------------
# cross-checks with the block kernels and dispatch
# --------------------------------------------------------------------------


def test_scalar_scan_cuda_uses_triton_kernel(monkeypatch):
    from model.ReKTP import triton_scan as ts

    def boom(*args, **kwargs):
        raise AssertionError("serial fallback must not run for 1x1 on CUDA")

    monkeypatch.setattr(ts, "_serial_scalar_affine_exclusive_scan", boom)
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, 6, 4, seed=41, valid_density=0.7
    )
    out = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    assert out.shape == bias.shape


def test_scalar_scan_cpu_falls_back_to_serial():
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, 6, 4, seed=53, valid_density=0.7
    )
    matrix = matrix.cpu()
    bias = bias.cpu()
    segment_ids = segment_ids.cpu()
    valid_mask = valid_mask.cpu()
    initial = initial.cpu()
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
