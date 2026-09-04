import pytest
import torch

AxisKT = pytest.importorskip("model.AxisKT.AxisKT_model").AxisKT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AxisKT requires CUDA"
)

DEVICE = torch.device("cuda")


def _build_model(device, *, activate_private_writes=True):
    # Skill id 2 is the padding sentinel.
    question_skill_ids = torch.tensor([[0, 2], [1, 2], [0, 1]])
    question_skill_mask = torch.tensor([[True, False], [True, False], [True, True]])
    model = AxisKT(
        data_metadata={"num_questions": 3, "num_skills": 2},
        question_skill_ids=question_skill_ids,
        question_skill_mask=question_skill_mask,
        hidden_dim=16,
        n_blocks=1,
        max_gap_bins=4,
        dropout=0.0,
    )
    if activate_private_writes:
        with torch.no_grad():
            model.answer_embed.weight[0].zero_()
            model.answer_embed.weight[1].fill_(0.25)
            identity = torch.eye(model.hidden_dim)
            model.local_write.weight.copy_(0.05 * identity)
    return model.to(device).eval()


def _dim_kwargs(hidden_dim=16):
    return {
        "data_metadata": {"num_questions": 3, "num_skills": 2},
        "question_skill_ids": torch.tensor([[0, 2], [1, 2], [0, 1]]),
        "question_skill_mask": torch.tensor(
            [[True, False], [True, False], [True, True]]
        ),
        "hidden_dim": hidden_dim,
        "n_blocks": 1,
        "max_gap_bins": 4,
        "dropout": 0.0,
    }


def _activate_head(model):
    """Make the zero-initialized IRT head emit non-constant logits.

    ability_head and question_diff start at zero, so a freshly built model
    returns identically-zero logits and any output comparison passes trivially.
    """
    with torch.no_grad():
        torch.manual_seed(0)
        model.ability_head.weight.normal_(0.0, 0.2)
        model.question_diff.weight.normal_(0.0, 0.2)
    return model


def _position_times(questions: torch.Tensor) -> torch.Tensor:
    """Position-index times reproducing the pre-real-time gap semantics."""
    return (
        torch.arange(questions.size(1), dtype=torch.float64, device=questions.device)
        .unsqueeze(0)
        .expand(questions.shape[0], -1)
    )


def test_default_question_embed_dim_matches_hidden_dim():
    model = AxisKT(**_dim_kwargs())

    # At full width the shared projection is skipped entirely.
    assert model.question_embed_proj is None
    assert model.question_embed.weight.shape == (3, 16)


@pytest.mark.parametrize("dim", [4, 8])
def test_low_dim_question_embed_shrinks_per_question_parameters(dim):
    full = AxisKT(**_dim_kwargs())
    reduced = AxisKT(**_dim_kwargs(), question_embed_dim=dim)

    assert reduced.question_embed.weight.shape == (3, dim)
    assert reduced.question_embed_proj is not None
    # Per-question rows shrink; the shared projection is independent of them.
    assert reduced.question_embed.weight.numel() < full.question_embed.weight.numel()


@pytest.mark.parametrize("dim", [4, 16])
def test_question_vector_is_always_hidden_dim_wide(dim):
    model = AxisKT(**_dim_kwargs(), question_embed_dim=dim).eval()

    with torch.no_grad():
        vector = model._question_vector(torch.tensor([[0, 1, 2]]))

    assert vector.shape == (1, 3, 16)
    assert torch.isfinite(vector).all()


def test_zero_dim_removes_the_question_pathway():
    model = AxisKT(**_dim_kwargs(), question_embed_dim=0)

    assert model.question_embed is None
    assert model.question_embed_proj is None
    vector = model._question_vector(torch.tensor([[0, 1, 2]]))
    assert vector.shape == (1, 3, 16)
    assert torch.all(vector == 0.0)


def test_zero_dim_drops_exactly_the_question_rows():
    full = AxisKT(**_dim_kwargs())
    ablated = AxisKT(**_dim_kwargs(), question_embed_dim=0)

    dropped = sum(p.numel() for p in full.parameters()) - sum(
        p.numel() for p in ablated.parameters()
    )
    assert dropped == 3 * 16


def test_zero_dim_still_runs_and_keeps_the_difficulty_scalar():
    model = AxisKT(**_dim_kwargs(), question_embed_dim=0).to(DEVICE).eval()
    with torch.no_grad():
        # question_diff is zero-initialised, so activate it to expose the scalar.
        model.question_diff.weight.normal_(0.0, 0.5)
    questions = torch.tensor([[0, 1, 2, 0]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 0]], device=DEVICE)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    with torch.no_grad():
        logits = model(questions, responses, times, mask)
        shifted = model(questions.roll(1, dims=1), responses, times, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()
    # Question identity still reaches the output through question_diff and KCs.
    assert not torch.allclose(logits, shifted)


def test_negative_question_embed_dim_is_rejected():
    with pytest.raises(ValueError, match="question_embed_dim must be non-negative"):
        AxisKT(**_dim_kwargs(), question_embed_dim=-2)


def test_local_readout_initializes_as_masked_mean():
    model = _build_model(DEVICE, activate_private_writes=False)
    with torch.no_grad():
        model.local_readout.weight.zero_()
        model.local_readout.bias.zero_()
    B, P, H = 2, 5, model.hidden_dim
    packed_state = torch.randn(B, P, H, device=DEVICE)
    skill_embedding = torch.randn(B, P, H, device=DEVICE)
    question_vector = torch.randn(B, 4, H, device=DEVICE)
    packed_pos = torch.tensor([[0, 0, 1, 2, 2], [0, 1, 1, 3, 3]], device=DEVICE)
    packed_valid = torch.ones(B, P, dtype=torch.bool, device=DEVICE)

    actual = model._packed_question_conditioned_readout(
        packed_state, skill_embedding, packed_pos, packed_valid, question_vector
    )
    # Zeroed weights give uniform scores, so the readout is a per-position
    # mean over the occurrences of that position.
    expected = torch.zeros(B, 4, H, device=DEVICE)
    for b in range(B):
        for s in range(4):
            members = packed_pos[b] == s
            if members.any():
                expected[b, s] = packed_state[b, members].mean(dim=0)

    torch.testing.assert_close(actual, expected)


def test_local_readout_can_weight_kcs_conditionally():
    model = _build_model(DEVICE, activate_private_writes=False)
    H = model.hidden_dim
    packed_state = torch.zeros(1, 2, H, device=DEVICE)
    packed_state[0, 0, 0] = -1.0
    packed_state[0, 1, 0] = 1.0
    skill_embedding = torch.zeros(1, 2, H, device=DEVICE)
    question_vector = torch.zeros(1, 1, H, device=DEVICE)
    packed_pos = torch.zeros(1, 2, dtype=torch.long, device=DEVICE)
    packed_valid = torch.ones(1, 2, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        model.local_readout.weight.zero_()
        model.local_readout.bias.zero_()
        model.local_readout.weight[0, 0] = 8.0

    actual = model._packed_question_conditioned_readout(
        packed_state, skill_embedding, packed_pos, packed_valid, question_vector
    )

    assert actual[0, 0, 0] > 0.9


def test_local_readout_matches_cat_linear_reference():
    torch.manual_seed(0)
    model = _build_model(DEVICE, activate_private_writes=False)
    with torch.no_grad():
        model.local_readout.weight.normal_(0.0, 0.5)
        model.local_readout.bias.normal_(0.0, 0.5)

    B, N, K, H = 2, 3, 2, model.hidden_dim
    local_state = torch.randn(B, N, K, H, device=DEVICE)
    skill_embedding = torch.randn(B, N, K, H, device=DEVICE)
    question_vector = torch.randn(B, N, H, device=DEVICE)
    readout_mask = torch.tensor(
        [[[True, True], [True, False], [False, True]]], device=DEVICE
    ).expand(B, N, K)

    # Pack the (s, k) grid in row-major order, dropping masked occurrences;
    # the (b, s) groups are then the runs of equal ``packed_pos``.
    flat_pos = torch.arange(N, device=DEVICE).view(N, 1).expand(N, K).reshape(-1)
    sel = readout_mask.reshape(B, N * K)
    packed_pos = torch.stack([flat_pos[sel[b]] for b in range(B)])
    packed_state = torch.stack(
        [local_state.reshape(B, N * K, H)[b][sel[b]] for b in range(B)]
    )
    packed_skill = torch.stack(
        [skill_embedding.reshape(B, N * K, H)[b][sel[b]] for b in range(B)]
    )
    packed_valid = torch.ones_like(packed_pos, dtype=torch.bool)

    actual = model._packed_question_conditioned_readout(
        packed_state, packed_skill, packed_pos, packed_valid, question_vector
    )

    # Reference: cat -> Linear(3H, 1) then the same masked-softmax readout.
    weight = model.local_readout.weight
    bias = model.local_readout.bias
    question_embedding = question_vector.unsqueeze(-2).expand_as(local_state)
    score_input = torch.cat((local_state, skill_embedding, question_embedding), dim=-1)
    scores = torch.nn.functional.linear(score_input, weight, bias).squeeze(-1)
    masked_scores = scores.masked_fill(~readout_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    weights = torch.where(readout_mask, weights, torch.zeros_like(weights))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    expected = (local_state * weights.unsqueeze(-1)).sum(dim=-2)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_event_embeddings_pools_skill_change():
    torch.manual_seed(0)
    model = _build_model(DEVICE, activate_private_writes=False)
    with torch.no_grad():
        model.question_diff.weight.normal_(0.0, 0.5)
        model.skill_change.weight.normal_(0.0, 0.5)
    questions = torch.tensor([[0, 1, 2, 0]], device=DEVICE)
    q_features = model._resolve_question_features(questions)

    actual = model._event_embeddings(questions, q_features)

    # Reference: direct gather + masked mean over the KC dimension.
    counts = q_features.skill_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    weights = q_features.skill_mask.float().unsqueeze(-1)
    pooled_skill = (model.skill_embed(q_features.skill_ids) * weights).sum(
        dim=-2
    ) / counts
    skill_change = model.skill_change(q_features.skill_ids)
    pooled_change = (skill_change * weights).sum(dim=-2) / counts
    expected = (
        q_features.question_vector
        + pooled_skill
        + model.question_diff(questions) * pooled_change
    )
    torch.testing.assert_close(actual, expected)


def test_other_kc_response_does_not_change_private_state():
    model = _build_model(DEVICE)
    questions = torch.tensor([[0, 1, 0, 1]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 0]], device=DEVICE)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    baseline = model._local_pre_states(questions, responses, times, mask)
    changed_responses = responses.clone()
    changed_responses[:, 0] = 0
    changed = model._local_pre_states(
        questions,
        changed_responses,
        times,
        mask,
    )

    # Position 3 addresses KC 1, so changing KC 0 at position 0 cannot alter it.
    torch.testing.assert_close(changed[:, 3], baseline[:, 3])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_own_gap_does_not_decay_own_read_state():
    model = _build_model(DEVICE)
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[1, 0] = 2.0
        model.local_decay.weight.zero_()
        model.local_decay.bias.zero_()
        model.local_decay.weight[:, 0] = 1.0
    questions = torch.tensor([[0, 1, 0]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1]], device=DEVICE)
    mask = torch.ones_like(responses, dtype=torch.bool)
    # Identical question/response history; only the timestamp of the KC-0
    # occurrence being read differs (2s vs 16s -> gap buckets 1 vs 3).
    short_times = torch.tensor([[0.0, 1.0, 2.0]], device=DEVICE)
    long_times = torch.tensor([[0.0, 1.0, 16.0]], device=DEVICE)
    short_local = model._local_pre_states(questions, responses, short_times, mask)
    long_local = model._local_pre_states(questions, responses, long_times, mask)
    torch.testing.assert_close(short_local[:, 2], long_local[:, 2])


def test_past_gap_composes_into_later_kc_read_states():
    model = _build_model(DEVICE)
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[1, 0] = 2.0
        model.local_decay.weight.zero_()
        model.local_decay.bias.zero_()
        model.local_decay.weight[:, 0] = 1.0
    questions = torch.tensor([[0, 1, 0, 0]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 1]], device=DEVICE)
    mask = torch.ones_like(responses, dtype=torch.bool)
    short_times = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=DEVICE)
    long_times = torch.tensor([[0.0, 1.0, 16.0, 17.0]], device=DEVICE)
    short_local = model._local_pre_states(questions, responses, short_times, mask)
    long_local = model._local_pre_states(questions, responses, long_times, mask)

    # The perturbed occurrence still reads an identical state ...
    torch.testing.assert_close(short_local[:, 2], long_local[:, 2])
    # ... but the same KC's next read sees the decayed difference.
    assert not torch.allclose(short_local[:, 3], long_local[:, 3])


def test_packing_keeps_kc_segments_contiguous_with_large_real_times():
    model = _build_model(DEVICE)
    # A naive (skill * stride + real_seconds) sort key would interleave the two
    # skills here; segment contiguity must be preserved regardless of time scale.
    questions = torch.tensor([[0, 1, 0, 1]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 0]], device=DEVICE)
    times = torch.tensor([[0.0, 1000.0, 5000.0, 6000.0]], device=DEVICE)
    mask = torch.ones_like(questions, dtype=torch.bool)
    packed_skill, _, _, _, packed_valid, _, _ = model._pack_kc_occurrences(
        questions, responses, times, mask
    )
    valid_skill = packed_skill[0][packed_valid[0]]
    runs = (valid_skill[1:] != valid_skill[:-1]).sum().item() + 1
    assert runs == 2  # exactly one contiguous run per skill


def test_forward_handles_padding_when_valid_times_are_large():
    # Padding positions hold time 0 while valid times are large; without the
    # padding pin, relative padding times go negative and log2 yields NaN.
    model = _build_model(DEVICE)
    questions = torch.tensor([[0, 1, 2, 0]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 0]], device=DEVICE)
    times = torch.tensor([[100.0, 200.0, 300.0, 0.0]], device=DEVICE)
    mask = torch.tensor([[True, True, True, False]], device=DEVICE)
    with torch.no_grad():
        logits = model(questions, responses, times, mask)
    assert torch.isfinite(logits).all()


def test_fused_inference_matches_training_readout_path():
    model = _activate_head(_build_model(DEVICE)).eval()
    with torch.no_grad():
        model.local_readout.weight.normal_(0.0, 0.2)
        model.local_readout.bias.normal_(0.0, 0.2)
        model.question_diff.weight.normal_(0.0, 0.2)
    questions = torch.tensor([[0, 2, 1, 2, 0, 1], [2, 0, 2, 1, 0, 0]], device=DEVICE)
    responses = torch.tensor([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0]], device=DEVICE)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
        ],
        device=DEVICE,
    )
    times = _position_times(questions)

    reference = model(questions, responses, times, mask)
    with torch.no_grad():
        actual = model(questions, responses, times, mask)

    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_target_answer_does_not_leak_into_its_prediction():
    device = DEVICE
    model = _build_model(device)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    baseline = model(questions, responses, times, mask)
    changed_responses = responses.clone()
    changed_responses[:, 2] = 0
    changed = model(questions, changed_responses, times, mask)

    # Output position 1 predicts response at position 2.
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])


def test_target_timestamp_does_not_leak_into_its_prediction():
    device = DEVICE
    model = _activate_head(_build_model(device))
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[1, 0] = 2.0
        model.local_decay.weight.zero_()
        model.local_decay.bias.zero_()
        model.local_decay.weight[:, 0] = 1.0
    questions = torch.tensor([[0, 1, 0, 0]], device=device)
    responses = torch.tensor([[1, 0, 1, 1]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0]], device=device)

    baseline = model(questions, responses, times, mask)
    changed_times = times.clone()
    changed_times[:, 2] = 16.0
    changed = model(questions, responses, changed_times, mask)

    # Output position 1 predicts the response at position 2.
    assert baseline[:, :2].abs().max() > 1e-6
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_forward_backward_has_finite_gradients():
    device = DEVICE
    model = _build_model(device).train()
    questions = torch.tensor([[0, 1, 0, 2], [1, 2, 1, 0]], device=device)
    responses = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    loss = model(questions, responses, times, mask)[:, :-1].square().mean()
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_encoder_forward_shape_and_finite():
    device = DEVICE
    model = _build_model(device)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    logits = model(questions, responses, times, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()


def test_encoder_forward_handles_padding():
    # Trailing padding must not introduce NaN/Inf, nor move valid predictions out of range.
    device = DEVICE
    model = _build_model(device)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.tensor([[True, True, False, False]], device=device)
    times = _position_times(questions)

    logits = model(questions, responses, times, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()


def test_encoder_backward_has_finite_gradients():
    device = DEVICE
    model = _build_model(device).train()
    questions = torch.tensor([[0, 1, 0, 2], [1, 2, 1, 0]], device=device)
    responses = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    loss = model(questions, responses, times, mask)[:, :-1].square().mean()
    loss.backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_encoder_does_not_leak_future_response():
    # Output positions 0,1 predict responses at positions 1,2; changing the answer at position 2 must not affect earlier predictions.
    device = DEVICE
    model = _activate_head(_build_model(device))
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 1]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    baseline = model(questions, responses, times, mask)
    changed_responses = responses.clone()
    changed_responses[:, 2] = 0
    changed = model(questions, changed_responses, times, mask)

    # Guard against a vacuous comparison of identically-zero logits.
    assert baseline[:, :2].abs().max() > 1e-6
    torch.testing.assert_close(changed[:, :2], baseline[:, :2])


def test_global_encoder_state_is_truncation_invariant():
    # The blocks carry no key-padding mask: they rely on padding being trailing,
    # so a valid position's state must not depend on how much padding follows.
    # Breaking this (left padding, or a non-causal block) invalidates the design.
    device = DEVICE
    model = _build_model(device)
    questions = torch.tensor([[0, 1, 2]], device=device)
    responses = torch.tensor([[1, 0, 1]], device=device)

    event = model._event_embeddings(questions)
    short = model._global_history_states(
        event, responses, torch.ones_like(questions, dtype=torch.bool)
    )

    padded_questions = torch.tensor([[0, 1, 2, 1, 0]], device=device)
    padded_responses = torch.tensor([[1, 0, 1, 1, 0]], device=device)
    padded_mask = torch.tensor([[True, True, True, False, False]], device=device)
    padded_event = model._event_embeddings(padded_questions)
    padded = model._global_history_states(padded_event, padded_responses, padded_mask)

    assert short.abs().max() > 1e-6
    torch.testing.assert_close(padded[:, :3], short)
