import importlib.util

import pytest
import torch

ReKTP = pytest.importorskip("model.ReKTP.ReKTP_model").ReKTP

# Mamba exercises its real CUDA kernels; CPU-only logic tests use LSTM.
HAS_MAMBA = importlib.util.find_spec("mamba_ssm") is not None


def _device_for_encoder(encoder_type):
    if encoder_type != "mamba":
        return torch.device("cpu")
    if not HAS_MAMBA:
        pytest.skip("mamba_ssm required for the mamba encoder")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the mamba encoder tests")
    return torch.device("cuda")


def _build_model(device, *, activate_private_writes=True, encoder_type="lstm"):
    if encoder_type == "mamba" and device.type != "cuda":
        raise ValueError("Mamba tests must build the model on CUDA")
    # Skill id 2 is the padding sentinel.
    question_skill_ids = torch.tensor([[0, 2], [1, 2], [0, 1]])
    question_skill_mask = torch.tensor([[True, False], [True, False], [True, True]])
    model = ReKTP(
        data_metadata={"num_questions": 3, "num_skills": 2},
        question_skill_ids=question_skill_ids,
        question_skill_mask=question_skill_mask,
        hidden_dim=16,
        n_blocks=1,
        d_state=8,
        d_conv=2,
        expand=1,
        max_gap_bins=4,
        dropout=0.0,
        encoder_type=encoder_type,
        n_heads=4,
    )
    if activate_private_writes:
        with torch.no_grad():
            model.answer_embed.weight[0].zero_()
            model.answer_embed.weight[1].fill_(0.25)
            identity = torch.eye(model.hidden_dim)
            model.local_write.weight.copy_(0.05 * identity)
            model.local_residual.weight[:, :].fill_(0.01)
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
        "encoder_type": "lstm",
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


def _zero_global_context(model, questions: torch.Tensor) -> torch.Tensor:
    """Zero global context; the film layer then leaves the input unchanged."""
    return torch.zeros(*questions.shape, model.hidden_dim, device=questions.device)


def test_default_question_embed_dim_matches_hidden_dim():
    model = ReKTP(**_dim_kwargs())

    # At full width the shared projection is skipped entirely.
    assert model.question_embed_proj is None
    assert model.question_embed.weight.shape == (3, 16)


@pytest.mark.parametrize("dim", [4, 8])
def test_low_dim_question_embed_shrinks_per_question_parameters(dim):
    full = ReKTP(**_dim_kwargs())
    reduced = ReKTP(**_dim_kwargs(), question_embed_dim=dim)

    assert reduced.question_embed.weight.shape == (3, dim)
    assert reduced.question_embed_proj is not None
    # Per-question rows shrink; the shared projection is independent of them.
    assert reduced.question_embed.weight.numel() < full.question_embed.weight.numel()


@pytest.mark.parametrize("dim", [4, 16])
def test_question_vector_is_always_hidden_dim_wide(dim):
    model = ReKTP(**_dim_kwargs(), question_embed_dim=dim).eval()

    with torch.no_grad():
        vector = model._question_vector(torch.tensor([[0, 1, 2]]))

    assert vector.shape == (1, 3, 16)
    assert torch.isfinite(vector).all()


def test_zero_dim_removes_the_question_pathway():
    model = ReKTP(**_dim_kwargs(), question_embed_dim=0)

    assert model.question_embed is None
    assert model.question_embed_proj is None
    vector = model._question_vector(torch.tensor([[0, 1, 2]]))
    assert vector.shape == (1, 3, 16)
    assert torch.all(vector == 0.0)


def test_zero_dim_drops_exactly_the_question_rows():
    full = ReKTP(**_dim_kwargs())
    ablated = ReKTP(**_dim_kwargs(), question_embed_dim=0)

    dropped = sum(p.numel() for p in full.parameters()) - sum(
        p.numel() for p in ablated.parameters()
    )
    assert dropped == 3 * 16


def test_zero_dim_still_runs_and_keeps_the_difficulty_scalar():
    model = ReKTP(**_dim_kwargs(), question_embed_dim=0).eval()
    with torch.no_grad():
        # question_diff is zero-initialised, so activate it to expose the scalar.
        model.question_diff.weight.normal_(0.0, 0.5)
    questions = torch.tensor([[0, 1, 2, 0]])
    responses = torch.tensor([[1, 0, 1, 0]])
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
        ReKTP(**_dim_kwargs(), question_embed_dim=-2)


def test_model_requires_complete_2x2_state_blocks():
    question_skill_ids = torch.tensor([[0, 2], [1, 2], [0, 1]])
    question_skill_mask = torch.tensor([[True, False], [True, False], [True, True]])

    with pytest.raises(ValueError, match="divisible by 2"):
        ReKTP(
            data_metadata={"num_questions": 3, "num_skills": 2},
            question_skill_ids=question_skill_ids,
            question_skill_mask=question_skill_mask,
            hidden_dim=15,
            n_blocks=1,
        )


def test_residual_transitions_initialize_as_decay_only():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    event_input = torch.randn(2, 3, model.hidden_dim)
    decay = torch.rand_like(event_input)

    transition, bias = model._block_affine_transition(
        event_input,
        model.local_residual,
        model.local_write,
        decay,
    )

    expected = torch.diag_embed(
        decay.reshape(2, 3, model.num_state_blocks, model.state_block_size)
    )
    torch.testing.assert_close(transition, expected)
    torch.testing.assert_close(bias, torch.zeros_like(bias))


def test_event_conditioned_residual_blocks_respect_scale():
    model = _build_model(torch.device("cpu"))
    event_input = torch.randn(2, 3, model.hidden_dim)
    decay = torch.ones_like(event_input)

    transition, _ = model._block_affine_transition(
        event_input,
        model.local_residual,
        model.local_write,
        decay,
    )

    identity = torch.eye(model.state_block_size)
    residual = transition - identity
    block_norm = torch.linalg.vector_norm(residual, dim=(-2, -1))
    assert torch.all(block_norm < model.residual_scale)


def test_local_readout_initializes_as_masked_mean():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    local_state = torch.randn(1, 2, 2, model.hidden_dim)
    skill_ids = torch.tensor([[[0, 1], [0, 2]]])
    readout_mask = torch.tensor([[[True, True], [True, False]]])
    questions = torch.tensor([[2, 0]])

    actual = model._question_conditioned_local_readout(
        local_state,
        skill_ids,
        readout_mask,
        questions,
    )
    expected = model._masked_mean(local_state, readout_mask)

    torch.testing.assert_close(actual, expected)


def test_local_readout_can_weight_kcs_conditionally():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    local_state = torch.zeros(1, 1, 2, model.hidden_dim)
    local_state[0, 0, 0, 0] = -1.0
    local_state[0, 0, 1, 0] = 1.0
    skill_ids = torch.tensor([[[0, 1]]])
    readout_mask = torch.tensor([[[True, True]]])
    questions = torch.tensor([[2]])

    with torch.no_grad():
        model.local_readout.weight.zero_()
        model.local_readout.bias.zero_()
        model.local_readout.weight[0, 0] = 8.0

    actual = model._question_conditioned_local_readout(
        local_state,
        skill_ids,
        readout_mask,
        questions,
    )

    assert actual[0, 0, 0] > 0.9


def test_global_film_initializes_as_identity_conditioning():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    local_input = torch.randn(2, 3, model.hidden_dim)
    global_context = torch.randn_like(local_input)

    actual = model._condition_local_input(local_input, global_context)

    torch.testing.assert_close(actual, local_input)


def test_global_film_can_condition_local_write_input():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    local_input = torch.ones(1, 2, model.hidden_dim)
    global_context = torch.zeros_like(local_input)
    global_context[:, :, 0] = torch.tensor([[1.0, -1.0]])

    with torch.no_grad():
        model.local_global_film.weight.zero_()
        model.local_global_film.bias.zero_()
        model.local_global_film.weight[model.hidden_dim, 0] = 0.5

    actual = model._condition_local_input(local_input, global_context)

    assert actual[0, 0, 0] > local_input[0, 0, 0]
    assert actual[0, 1, 0] < local_input[0, 1, 0]


def test_other_kc_response_does_not_change_private_state():
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 0, 1]])
    responses = torch.tensor([[1, 0, 1, 0]])
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    baseline = model._local_pre_states(
        questions, responses, times, mask, _zero_global_context(model, questions)
    )
    changed_responses = responses.clone()
    changed_responses[:, 0] = 0
    changed = model._local_pre_states(
        questions,
        changed_responses,
        times,
        mask,
        _zero_global_context(model, questions),
    )

    # Position 3 addresses KC 1, so changing KC 0 at position 0 cannot alter it.
    torch.testing.assert_close(changed[:, 3], baseline[:, 3])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_question_static_state_is_a_pure_function_of_question_and_position():
    model = _build_model(torch.device("cpu"))
    with torch.no_grad():
        # question_decay is zero-initialised, which makes decay gap-independent;
        # activate it so the position modulation is observable.
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[1, 0] = 2.0
        model.question_decay.weight.zero_()
        model.question_decay.weight[:, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    times = torch.arange(4, dtype=torch.float64).unsqueeze(0)

    state = model._question_static_states(torch.tensor([[0, 1, 2, 0]]), times, mask)
    repeated = model._question_static_states(torch.tensor([[0, 0, 0, 0]]), times, mask)

    # Position 0 carries question 0 in both sequences, so the outputs agree.
    torch.testing.assert_close(state[:, 0], repeated[:, 0])
    # The same question at gap buckets 0 and 1 is modulated differently.
    assert not torch.allclose(repeated[:, 0], repeated[:, 1])


def test_question_static_state_matches_removed_scan_without_repeats():
    # The pathway is the closed form the original per-question scan collapsed to
    # at positions with no predecessor: decay(floor(log2(t+1))) * tanh(W E_q).
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 2, 1]])
    mask = torch.ones_like(questions, dtype=torch.bool)

    times = torch.arange(questions.size(1), dtype=torch.float64).unsqueeze(0)
    gap_bucket = torch.floor(torch.log2((times + 1).float())).long()
    gap_bucket = gap_bucket.clamp_max(model.max_gap_bins - 1).expand_as(questions)
    decay = torch.exp(
        -torch.nn.functional.softplus(model.question_decay(model.gap_embed(gap_bucket)))
    )
    expected = decay * torch.tanh(model.question_init(model.question_embed(questions)))

    actual = model._question_static_states(questions, times, mask)

    torch.testing.assert_close(actual, expected)


def test_question_static_state_zeroes_padding():
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 2, 0]])
    mask = torch.tensor([[True, True, False, False]])
    times = torch.arange(4, dtype=torch.float64).unsqueeze(0)

    state = model._question_static_states(questions, times, mask)

    assert torch.all(state[:, 2:] == 0.0)
    assert not torch.all(state[:, :2] == 0.0)


def test_current_gap_affects_kc_read_states():
    model = _build_model(torch.device("cpu"))
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[2, 0] = 2.0
        model.local_decay.weight.zero_()
        model.local_decay.bias.zero_()
        model.local_decay.weight[:, 0] = 1.0

    short_questions = torch.tensor([[0, 1, 0, 1, 1]])
    long_questions = torch.tensor([[0, 1, 1, 1, 0]])
    short_responses = torch.tensor([[1, 0, 1, 0, 0]])
    long_responses = torch.tensor([[1, 0, 0, 0, 1]])
    mask = torch.ones_like(short_responses, dtype=torch.bool)
    times = _position_times(short_questions)

    short_local = model._local_pre_states(
        short_questions,
        short_responses,
        times,
        mask,
        _zero_global_context(model, short_questions),
    )
    long_local = model._local_pre_states(
        long_questions,
        long_responses,
        times,
        mask,
        _zero_global_context(model, long_questions),
    )

    # KC 0 has the same first event but a gap of 2 versus 4.
    assert not torch.allclose(short_local[:, 2], long_local[:, 4])


def test_real_time_gap_affects_kc_read_states():
    model = _build_model(torch.device("cpu"))
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[1, 0] = 2.0
        model.local_decay.weight.zero_()
        model.local_decay.bias.zero_()
        model.local_decay.weight[:, 0] = 1.0
    questions = torch.tensor([[0, 1, 0]])
    responses = torch.tensor([[1, 0, 1]])
    mask = torch.ones_like(responses, dtype=torch.bool)
    # Identical question/response patterns; only the elapsed seconds between
    # the two KC-0 occurrences differ (2s vs 16s -> gap buckets 1 vs 3).
    short_times = torch.tensor([[0.0, 1.0, 2.0]])
    long_times = torch.tensor([[0.0, 1.0, 16.0]])
    short_local = model._local_pre_states(
        questions, responses, short_times, mask, _zero_global_context(model, questions)
    )
    long_local = model._local_pre_states(
        questions, responses, long_times, mask, _zero_global_context(model, questions)
    )
    assert not torch.allclose(short_local[:, 2], long_local[:, 2])


def test_packing_keeps_kc_segments_contiguous_with_large_real_times():
    model = _build_model(torch.device("cpu"))
    # A naive (skill * stride + real_seconds) sort key would interleave the two
    # skills here; segment contiguity must be preserved regardless of time scale.
    questions = torch.tensor([[0, 1, 0, 1]])
    responses = torch.tensor([[1, 0, 1, 0]])
    times = torch.tensor([[0.0, 1000.0, 5000.0, 6000.0]])
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
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 2, 0]])
    responses = torch.tensor([[1, 0, 1, 0]])
    times = torch.tensor([[100.0, 200.0, 300.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        logits = model(questions, responses, times, mask)
    assert torch.isfinite(logits).all()


def test_target_answer_does_not_leak_into_its_prediction():
    device = torch.device("cpu")
    model = _build_model(device)
    with torch.no_grad():
        model.local_global_film.weight.zero_()
        model.local_global_film.bias.zero_()
        model.local_global_film.weight[model.hidden_dim :, :] = 0.05
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


def test_forward_backward_has_finite_gradients():
    device = torch.device("cpu")
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


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_forward_shape_and_finite(encoder_type):
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)
    times = _position_times(questions)

    logits = model(questions, responses, times, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_forward_handles_padding(encoder_type):
    # Trailing padding must not introduce NaN/Inf, nor move valid predictions out of range.
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.tensor([[True, True, False, False]], device=device)
    times = _position_times(questions)

    logits = model(questions, responses, times, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_backward_has_finite_gradients(encoder_type):
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type).train()
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


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_does_not_leak_future_response(encoder_type):
    # Output positions 0,1 predict responses at positions 1,2; changing the answer at position 2 must not affect earlier predictions.
    device = _device_for_encoder(encoder_type)
    model = _activate_head(_build_model(device, encoder_type=encoder_type))
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


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_global_encoder_state_is_truncation_invariant(encoder_type):
    # The blocks carry no key-padding mask: they rely on padding being trailing,
    # so a valid position's state must not depend on how much padding follows.
    # Breaking this (left padding, or a non-causal block) invalidates the design.
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type)
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
