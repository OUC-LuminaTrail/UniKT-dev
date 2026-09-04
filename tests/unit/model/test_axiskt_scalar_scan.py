"""Tests for the 1x1 (scalar) Triton segmented scan in AxisKT.

Pinned semantics: exclusive (the output at position ``n`` is the state
before ``n``'s operator), segment heads reset the carry to the identity,
the initial state applies at every position, and invalid positions output
zero and break the segment run. The pure-PyTorch ``_scalar_scan_reference``
mirrors these semantics with full autograd coverage, so the Triton results
are checked against it both forward and backward.
"""

import pytest
import torch

from model.AxisKT.triton_scan import segmented_scalar_affine_exclusive_scan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AxisKT segmented scan requires CUDA"
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


@pytest.mark.parametrize("length", [17, 200])
def test_post_multiply_matches_explicit_transition_and_gradients(length):
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, length, 8, seed=25, valid_density=0.75, num_segments=2
    )
    matrix = matrix.sigmoid()
    segment_ids[0] = 0
    valid_mask[0] = True
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    def fused(m, b, seg, valid, init):
        return segmented_scalar_affine_exclusive_scan(
            m, b, seg, valid, init, post_multiply=True
        )

    def explicit(m, b, seg, valid, init):
        pre = segmented_scalar_affine_exclusive_scan(m, b, seg, valid, init)
        return pre * m[..., 0]

    fused_grads, fused_out = _grads_of(
        fused, matrix, bias, initial, segment_ids, valid_mask, weights
    )
    explicit_grads, explicit_out = _grads_of(
        explicit, matrix, bias, initial, segment_ids, valid_mask, weights
    )

    torch.testing.assert_close(fused_out, explicit_out, rtol=1e-4, atol=1e-5)
    for fused_grad, explicit_grad in zip(fused_grads, explicit_grads, strict=True):
        torch.testing.assert_close(fused_grad, explicit_grad, rtol=1e-3, atol=1e-4)


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
    from model.AxisKT import triton_scan as ts

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


# --------------------------------------------------------------------------
# coded operator mode (codebook + per-position codes)
# --------------------------------------------------------------------------


def _coded_case(batch, length, heads, bins, seed, valid_density=0.7):
    """Random scan inputs whose operators come from a small codebook."""
    torch.manual_seed(seed)
    segment_ids = torch.randint(0, 3, (batch, length), device=DEVICE)
    valid_mask = torch.rand(batch, length, device=DEVICE) < valid_density
    table = torch.rand(bins, heads, device=DEVICE) * 0.4 + 0.6
    codes = torch.randint(0, bins, (batch, length), device=DEVICE)
    bias = torch.randn(batch, length, heads, 1, device=DEVICE)
    initial = torch.randn(batch, length, heads, 1, device=DEVICE)
    return table, codes, bias, segment_ids, valid_mask, initial


@pytest.mark.parametrize(
    ("batch", "length", "heads"),
    [(3, 40, 8), (2, 200, 16), (2, 130, 12)],
)
def test_coded_scan_matches_dense_matrix(batch, length, heads):
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        batch, length, heads, bins=5, seed=7
    )
    dense = table[codes][..., None, None]
    actual = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        initial,
        matrix_table=table,
        matrix_codes=codes,
    )
    expected = segmented_scalar_affine_exclusive_scan(
        dense, bias, segment_ids, valid_mask, initial
    )
    # Identical operator values must give identical kernels and outputs.
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("length", [40, 150])
def test_coded_scan_gradients_match_dense(length):
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        2, length, 8, bins=4, seed=13, valid_density=0.75
    )
    segment_ids[0] = 0
    valid_mask[0] = True
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    table_c = table.clone().requires_grad_()
    bias_c = bias.clone().requires_grad_()
    init_c = initial.clone().requires_grad_()
    out_c = segmented_scalar_affine_exclusive_scan(
        None,
        bias_c,
        segment_ids,
        valid_mask,
        init_c,
        matrix_table=table_c,
        matrix_codes=codes,
    )
    (out_c * weights).sum().backward()

    dense = table[codes][..., None, None]
    matrix_d = dense.clone().requires_grad_()
    bias_d = bias.clone().requires_grad_()
    init_d = initial.clone().requires_grad_()
    out_d = segmented_scalar_affine_exclusive_scan(
        matrix_d, bias_d, segment_ids, valid_mask, init_d
    )
    (out_d * weights).sum().backward()

    torch.testing.assert_close(out_c, out_d)
    expected_table_grad = torch.zeros_like(table).index_add_(
        0, codes.reshape(-1), matrix_d.grad.reshape(-1, table.shape[1])
    )
    torch.testing.assert_close(table_c.grad, expected_table_grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(bias_c.grad, bias_d.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(init_c.grad, init_d.grad, rtol=1e-4, atol=1e-5)


def test_coded_scan_gradcheck():
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        1, 5, 2, bins=3, seed=29, valid_density=0.6
    )
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    def loss_fn(t, b, init):
        out = segmented_scalar_affine_exclusive_scan(
            None,
            b,
            segment_ids,
            valid_mask,
            init,
            matrix_table=t,
            matrix_codes=codes,
        )
        return (out * weights).sum()

    torch.autograd.gradcheck(
        loss_fn,
        (table.requires_grad_(), bias.requires_grad_(), initial.requires_grad_()),
        eps=1e-2,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_coded_scan_ignores_codes_at_invalid_positions():
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        2, 9, 4, bins=3, seed=31, valid_density=0.5
    )
    assert (~valid_mask).any(), "case must contain invalid positions"
    baseline = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        initial,
        matrix_table=table,
        matrix_codes=codes,
    )
    shifted_codes = codes.clone()
    shifted_codes[~valid_mask] = (shifted_codes[~valid_mask] + 1) % table.size(0)
    changed = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        initial,
        matrix_table=table,
        matrix_codes=shifted_codes,
    )
    torch.testing.assert_close(changed, baseline)


def _coded_init_case(
    batch, length, heads, bins, seed, valid_density=0.7, align_codes=False
):
    """Random scan inputs whose initial states come from a small codebook."""
    torch.manual_seed(seed)
    segment_ids = torch.randint(0, 3, (batch, length), device=DEVICE)
    valid_mask = torch.rand(batch, length, device=DEVICE) < valid_density
    matrix = torch.rand(batch, length, heads, 1, 1, device=DEVICE) * 0.4 + 0.6
    bias = torch.randn(batch, length, heads, 1, device=DEVICE)
    init_table = torch.randn(bins, heads, device=DEVICE)
    if align_codes:
        init_codes = segment_ids.clone()
    else:
        init_codes = torch.randint(0, bins, (batch, length), device=DEVICE)
    return matrix, bias, segment_ids, valid_mask, init_table, init_codes


@pytest.mark.parametrize(
    ("batch", "length", "heads"),
    [(3, 40, 8), (2, 200, 16)],
)
def test_coded_init_matches_dense_initial_state(batch, length, heads):
    matrix, bias, segment_ids, valid_mask, init_table, init_codes = _coded_init_case(
        batch, length, heads, bins=4, seed=11
    )
    dense_init = init_table[init_codes][..., None]
    actual = segmented_scalar_affine_exclusive_scan(
        matrix,
        bias,
        segment_ids,
        valid_mask,
        None,
        init_table=init_table,
        init_codes=init_codes,
    )
    expected = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, dense_init
    )
    assert torch.equal(actual, expected)


def test_coded_matrix_and_init_match_dense():
    table, codes, bias, segment_ids, valid_mask, _ = _coded_case(
        2, 150, 8, bins=4, seed=23
    )
    _, _, _, _, init_table, init_codes = _coded_init_case(2, 150, 8, bins=4, seed=29)
    dense = table[codes][..., None, None]
    dense_init = init_table[init_codes][..., None]
    actual = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        None,
        matrix_table=table,
        matrix_codes=codes,
        init_table=init_table,
        init_codes=init_codes,
    )
    expected = segmented_scalar_affine_exclusive_scan(
        dense, bias, segment_ids, valid_mask, dense_init
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("length", [40, 150])
def test_coded_init_gradients_match_dense(length):
    matrix, bias, segment_ids, valid_mask, init_table, init_codes = _coded_init_case(
        2, length, 8, bins=4, seed=17
    )
    weights = torch.randn_like(bias)

    init_table_c = init_table.clone().requires_grad_()
    bias_c = bias.clone().requires_grad_()
    out_c = segmented_scalar_affine_exclusive_scan(
        matrix,
        bias_c,
        segment_ids,
        valid_mask,
        None,
        init_table=init_table_c,
        init_codes=init_codes,
    )
    (out_c * weights).sum().backward()

    dense_init = init_table[init_codes][..., None]
    init_d = dense_init.clone().requires_grad_()
    bias_d = bias.clone().requires_grad_()
    out_d = segmented_scalar_affine_exclusive_scan(
        matrix, bias_d, segment_ids, valid_mask, init_d
    )
    (out_d * weights).sum().backward()

    torch.testing.assert_close(out_c, out_d)
    expected_table_grad = torch.zeros_like(init_table).index_add_(
        0, init_codes.reshape(-1), init_d.grad.reshape(-1, init_table.shape[1])
    )
    torch.testing.assert_close(
        init_table_c.grad, expected_table_grad, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(bias_c.grad, bias_d.grad, rtol=1e-4, atol=1e-5)


def test_coded_init_gradcheck():
    matrix, bias, segment_ids, valid_mask, init_table, init_codes = _coded_init_case(
        1, 5, 2, bins=3, seed=37, valid_density=0.6, align_codes=True
    )
    weights = torch.randn_like(bias)

    def loss_fn(m, b, t):
        out = segmented_scalar_affine_exclusive_scan(
            m,
            b,
            segment_ids,
            valid_mask,
            None,
            init_table=t,
            init_codes=init_codes,
        )
        return (out * weights).sum()

    torch.autograd.gradcheck(
        loss_fn,
        (matrix.requires_grad_(), bias.requires_grad_(), init_table.requires_grad_()),
        eps=1e-2,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_coded_init_ignores_codes_at_invalid_positions():
    matrix, bias, segment_ids, valid_mask, init_table, init_codes = _coded_init_case(
        2, 9, 4, bins=3, seed=41, valid_density=0.5
    )
    assert (~valid_mask).any(), "case must contain invalid positions"
    baseline = segmented_scalar_affine_exclusive_scan(
        matrix,
        bias,
        segment_ids,
        valid_mask,
        None,
        init_table=init_table,
        init_codes=init_codes,
    )
    shifted_codes = init_codes.clone()
    shifted_codes[~valid_mask] = (shifted_codes[~valid_mask] + 1) % init_table.size(0)
    changed = segmented_scalar_affine_exclusive_scan(
        matrix,
        bias,
        segment_ids,
        valid_mask,
        None,
        init_table=init_table,
        init_codes=shifted_codes,
    )
    torch.testing.assert_close(changed, baseline)


@pytest.mark.parametrize("length", [40, 150])
def test_post_multiply_coded_tables_match_explicit_transition(length):
    table, codes, bias, segment_ids, valid_mask, _ = _coded_case(
        2, length, 8, bins=4, seed=73, valid_density=0.75
    )
    segment_ids[0] = 0
    valid_mask[0] = True
    init_table = torch.randn(3, 8, device=DEVICE)
    init_codes = segment_ids.clone()
    weights = torch.randn_like(bias)

    def run(post_multiply):
        table_i = table.clone().requires_grad_()
        bias_i = bias.clone().requires_grad_()
        init_i = init_table.clone().requires_grad_()
        out = segmented_scalar_affine_exclusive_scan(
            None,
            bias_i,
            segment_ids,
            valid_mask,
            None,
            matrix_table=table_i,
            matrix_codes=codes,
            init_table=init_i,
            init_codes=init_codes,
            post_multiply=post_multiply,
        )
        if not post_multiply:
            out = out * table_i[codes][..., None]
        (out * weights).sum().backward()
        return out.detach(), (table_i.grad, bias_i.grad, init_i.grad)

    fused_out, fused_grads = run(True)
    explicit_out, explicit_grads = run(False)

    torch.testing.assert_close(fused_out, explicit_out, rtol=1e-4, atol=1e-5)
    for fused_grad, explicit_grad in zip(fused_grads, explicit_grads, strict=True):
        torch.testing.assert_close(fused_grad, explicit_grad, rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------
# chunked long-sequence coverage (length >= _CHUNKED_MIN_LENGTH)
#
# Every reference comparison above runs at short lengths (legacy kernels) or
# against the dense Triton path (test_coded_scan_matches_dense_matrix and
# friends), which shares the chunked machinery — symmetric chunked bugs
# cancel there. These tests pin the chunked pipeline via monkeypatch (the
# autotuner may otherwise pick the legacy kernels) and compare against the
# independent serial reference on both the dense and coded paths.
# --------------------------------------------------------------------------


def _force_chunked(monkeypatch, block_n, num_warps=4):
    """Pin every scan mode to the three-pass chunked pipeline.

    Without this pin the autotuner may pick the legacy single-program
    kernels, so a long-sequence test may never execute the chunked kernels.
    """
    from model.AxisKT import triton_scan as ts

    monkeypatch.setattr(
        ts, "_pick_chunk_config", lambda *args, **kwargs: (block_n, num_warps)
    )


@pytest.mark.parametrize(
    ("length", "block_n"),
    [(200, 64), (130, 128), (130, 32)],
)
def test_scalar_scan_chunked_matches_reference(length, block_n, monkeypatch):
    _force_chunked(monkeypatch, block_n)
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, length, 16, seed=43, valid_density=0.7
    )
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("length", "block_n"),
    [(200, 64), (130, 128)],
)
def test_coded_scan_chunked_matches_reference(length, block_n, monkeypatch):
    _force_chunked(monkeypatch, block_n)
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        2, length, 16, bins=5, seed=47
    )
    actual = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        initial,
        matrix_table=table,
        matrix_codes=codes,
    )
    dense = table[codes][..., None, None]
    expected = _scalar_scan_reference(dense, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("length", "block_n"),
    [(200, 64), (130, 128), (130, 32)],
)
def test_scalar_scan_chunked_boundary_crossing_matches_reference(
    length, block_n, monkeypatch
):
    """All-valid single-segment rows must compose the incoming carry.

    The incoming ``pa/pb`` carry is only consumed from a tile's start up to
    its first segment head, so a random case can miss the boundary crossing
    by luck (chunk 0 never uses it at all). Constant segments with every
    position valid make every position past a chunk boundary consume the
    carry, so carry-buffer aliasing cannot hide.
    """
    _force_chunked(monkeypatch, block_n)
    torch.manual_seed(61)
    matrix = torch.randn(2, length, 16, 1, 1, device=DEVICE)
    bias = torch.randn(2, length, 16, 1, device=DEVICE)
    initial = torch.randn(2, length, 16, 1, device=DEVICE)
    segment_ids = torch.zeros(2, length, dtype=torch.long, device=DEVICE)
    valid_mask = torch.ones(2, length, dtype=torch.bool, device=DEVICE)
    actual = segmented_scalar_affine_exclusive_scan(
        matrix, bias, segment_ids, valid_mask, initial
    )
    expected = _scalar_scan_reference(matrix, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("length", "block_n"),
    [(200, 64), (130, 128), (130, 32)],
)
def test_coded_scan_chunked_boundary_crossing_matches_reference(
    length, block_n, monkeypatch
):
    """Coded-path twin of the boundary-crossing test above."""
    _force_chunked(monkeypatch, block_n)
    torch.manual_seed(67)
    bins, heads = 5, 16
    table = torch.rand(bins, heads, device=DEVICE) * 0.4 + 0.6
    codes = torch.randint(0, bins, (2, length), device=DEVICE)
    bias = torch.randn(2, length, heads, 1, device=DEVICE)
    initial = torch.randn(2, length, heads, 1, device=DEVICE)
    segment_ids = torch.zeros(2, length, dtype=torch.long, device=DEVICE)
    valid_mask = torch.ones(2, length, dtype=torch.bool, device=DEVICE)
    actual = segmented_scalar_affine_exclusive_scan(
        None,
        bias,
        segment_ids,
        valid_mask,
        initial,
        matrix_table=table,
        matrix_codes=codes,
    )
    dense = table[codes][..., None, None]
    expected = _scalar_scan_reference(dense, bias, segment_ids, valid_mask, initial)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_scalar_scan_chunked_gradients_match_reference(monkeypatch):
    """Lock the chunked adjoint / d-init kernels to the serial reference."""
    _force_chunked(monkeypatch, 64)
    matrix, bias, segment_ids, valid_mask, initial = _random_case(
        2, 200, 8, seed=51, valid_density=0.75, num_segments=2
    )
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

    torch.testing.assert_close(out, rout, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dmat, rmat, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(dbias, rbias, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(dinit, rinit, rtol=1e-3, atol=1e-4)


def test_coded_scan_chunked_gradients_match_reference(monkeypatch):
    """Coded-path chunked gradients against the serial dense reference."""
    _force_chunked(monkeypatch, 64)
    table, codes, bias, segment_ids, valid_mask, initial = _coded_case(
        2, 200, 8, bins=4, seed=59, valid_density=0.75
    )
    segment_ids[0] = 0
    valid_mask[0] = True
    initial = _make_segment_constant_init(initial, segment_ids, valid_mask)
    weights = torch.randn_like(bias)

    table_c = table.clone().requires_grad_()
    bias_c = bias.clone().requires_grad_()
    init_c = initial.clone().requires_grad_()
    out_c = segmented_scalar_affine_exclusive_scan(
        None,
        bias_c,
        segment_ids,
        valid_mask,
        init_c,
        matrix_table=table_c,
        matrix_codes=codes,
    )
    (out_c * weights).sum().backward()

    dense = table[codes][..., None, None].clone().requires_grad_()
    bias_d = bias.clone().requires_grad_()
    init_d = initial.clone().requires_grad_()
    out_d = _scalar_scan_reference(dense, bias_d, segment_ids, valid_mask, init_d)
    (out_d * weights).sum().backward()

    torch.testing.assert_close(out_c, out_d, rtol=1e-4, atol=1e-4)
    expected_table_grad = torch.zeros_like(table).index_add_(
        0, codes.reshape(-1), dense.grad.reshape(-1, table.shape[1])
    )
    torch.testing.assert_close(table_c.grad, expected_table_grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(bias_c.grad, bias_d.grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(init_c.grad, init_d.grad, rtol=1e-3, atol=1e-4)
