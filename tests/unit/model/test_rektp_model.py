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


def test_local_credit_gate_initializes_as_identity():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    model.local_credit_scale = 1.0
    bias = torch.randn(2, 3, model.num_state_blocks, model.state_block_size)
    local_input = torch.randn(2, 3, model.hidden_dim)

    actual = model._apply_local_credit(bias, local_input)

    torch.testing.assert_close(actual, bias)


def test_local_credit_gate_can_modulate_write_bias():
    model = _build_model(torch.device("cpu"), activate_private_writes=False)
    model.local_credit_scale = 1.0
    bias = torch.ones(1, 2, model.num_state_blocks, model.state_block_size)
    local_input = torch.zeros(1, 2, model.hidden_dim)
    local_input[:, :, 0] = torch.tensor([[1.0, -1.0]])

    with torch.no_grad():
        model.local_credit.weight.zero_()
        model.local_credit.bias.zero_()
        model.local_credit.weight[0, 0] = 1.0

    actual = model._apply_local_credit(bias, local_input)

    assert actual[0, 0, 0, 0] > bias[0, 0, 0, 0]
    assert actual[0, 1, 0, 0] < bias[0, 1, 0, 0]


def test_other_kc_response_does_not_change_private_state():
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 0, 1]])
    responses = torch.tensor([[1, 0, 1, 0]])
    mask = torch.ones_like(questions, dtype=torch.bool)

    baseline, _ = model._local_pre_states(questions, responses, mask)
    changed_responses = responses.clone()
    changed_responses[:, 0] = 0
    changed, _ = model._local_pre_states(questions, changed_responses, mask)

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

    state = model._question_static_states(torch.tensor([[0, 1, 2, 0]]), mask)
    repeated = model._question_static_states(torch.tensor([[0, 0, 0, 0]]), mask)

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

    times = torch.arange(questions.size(1)).unsqueeze(0)
    gap_bucket = torch.floor(torch.log2((times + 1).float())).long()
    gap_bucket = gap_bucket.clamp_max(model.max_gap_bins - 1).expand_as(questions)
    decay = torch.exp(
        -torch.nn.functional.softplus(model.question_decay(model.gap_embed(gap_bucket)))
    )
    expected = decay * torch.tanh(model.question_init(model.question_embed(questions)))

    actual = model._question_static_states(questions, mask)

    torch.testing.assert_close(actual, expected)


def test_question_static_state_zeroes_padding():
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 2, 0]])
    mask = torch.tensor([[True, True, False, False]])

    state = model._question_static_states(questions, mask)

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

    short_local, _ = model._local_pre_states(short_questions, short_responses, mask)
    long_local, _ = model._local_pre_states(long_questions, long_responses, mask)

    # KC 0 has the same first event but a gap of 2 versus 4.
    assert not torch.allclose(short_local[:, 2], long_local[:, 4])


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

    baseline = model(questions, responses, mask)
    changed_responses = responses.clone()
    changed_responses[:, 2] = 0
    changed = model(questions, changed_responses, mask)

    # Output position 1 predicts response at position 2.
    torch.testing.assert_close(changed[:, 1], baseline[:, 1])


def test_forward_with_auxiliary_logits_matches_main_shape():
    device = torch.device("cpu")
    model = _build_model(device)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)

    logits, aux_logits = model(questions, responses, mask, return_aux=True)

    assert logits.shape == aux_logits.shape == questions.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_logits).all()


def test_forward_backward_has_finite_gradients():
    device = torch.device("cpu")
    model = _build_model(device).train()
    questions = torch.tensor([[0, 1, 0, 2], [1, 2, 1, 0]], device=device)
    responses = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)

    loss = model(questions, responses, mask)[:, :-1].square().mean()
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

    logits = model(questions, responses, mask)

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

    logits = model(questions, responses, mask)

    assert logits.shape == questions.shape
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_aux_logits_match_shape(encoder_type):
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)

    logits, aux_logits = model(questions, responses, mask, return_aux=True)

    assert logits.shape == aux_logits.shape == questions.shape
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux_logits).all()


@pytest.mark.parametrize("encoder_type", ["mamba", "lstm", "transformer"])
def test_encoder_variant_backward_has_finite_gradients(encoder_type):
    device = _device_for_encoder(encoder_type)
    model = _build_model(device, encoder_type=encoder_type).train()
    questions = torch.tensor([[0, 1, 0, 2], [1, 2, 1, 0]], device=device)
    responses = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)

    loss = model(questions, responses, mask)[:, :-1].square().mean()
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
    model = _build_model(device, encoder_type=encoder_type)
    questions = torch.tensor([[0, 1, 0, 2]], device=device)
    responses = torch.tensor([[1, 0, 1, 0]], device=device)
    mask = torch.ones_like(questions, dtype=torch.bool)

    baseline = model(questions, responses, mask)
    changed_responses = responses.clone()
    changed_responses[:, 2] = 0
    changed = model(questions, changed_responses, mask)

    torch.testing.assert_close(changed[:, :2], baseline[:, :2])
