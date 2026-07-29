import pytest
import torch

ReKTP = pytest.importorskip("model.ReKTP.ReKTP_model").ReKTP


def _build_model(device, *, activate_private_writes=True):
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
    )
    # Logic tests run without a GPU; the real Mamba constructor above still
    # verifies the locked package API, while Identity avoids its CUDA kernel.
    for block in model.global_blocks:
        block.mamba = torch.nn.Identity()
    if activate_private_writes:
        with torch.no_grad():
            model.answer_embed.weight[0].zero_()
            model.answer_embed.weight[1].fill_(0.25)
            identity = torch.eye(model.hidden_dim)
            for write_layer in (model.local_write, model.question_write):
                write_layer.weight.copy_(0.05 * identity)
            for residual_layer in (
                model.local_residual,
                model.question_residual,
            ):
                residual_layer.weight[:, :].fill_(0.01)
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


def test_other_question_response_does_not_change_private_state():
    model = _build_model(torch.device("cpu"))
    questions = torch.tensor([[0, 1, 0, 1]])
    responses = torch.tensor([[1, 0, 1, 0]])
    mask = torch.ones_like(questions, dtype=torch.bool)
    event_embeddings, _ = model._event_embeddings(questions)

    baseline = model._question_pre_states(questions, responses, mask, event_embeddings)
    changed_responses = responses.clone()
    changed_responses[:, 0] = 0
    changed = model._question_pre_states(
        questions, changed_responses, mask, event_embeddings
    )

    # Question 1 has an independent segment, while question 0 retrieves its update.
    torch.testing.assert_close(changed[:, 3], baseline[:, 3])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_current_gap_affects_question_and_kc_read_states():
    model = _build_model(torch.device("cpu"))
    with torch.no_grad():
        model.gap_embed.weight.zero_()
        model.gap_embed.weight[2, 0] = 2.0
        for decay_layer in (model.local_decay, model.question_decay):
            decay_layer.weight.zero_()
            decay_layer.bias.zero_()
            decay_layer.weight[:, 0] = 1.0

    short_questions = torch.tensor([[0, 1, 0, 1, 1]])
    long_questions = torch.tensor([[0, 1, 1, 1, 0]])
    short_responses = torch.tensor([[1, 0, 1, 0, 0]])
    long_responses = torch.tensor([[1, 0, 0, 0, 1]])
    mask = torch.ones_like(short_responses, dtype=torch.bool)

    short_local, _ = model._local_pre_states(short_questions, short_responses, mask)
    long_local, _ = model._local_pre_states(long_questions, long_responses, mask)
    short_event, _ = model._event_embeddings(short_questions)
    long_event, _ = model._event_embeddings(long_questions)
    short_question = model._question_pre_states(
        short_questions, short_responses, mask, short_event
    )
    long_question = model._question_pre_states(
        long_questions, long_responses, mask, long_event
    )

    # KC/question 0 has the same first event but a gap of 2 versus 4.
    assert not torch.allclose(short_local[:, 2], long_local[:, 4])
    assert not torch.allclose(short_question[:, 2], long_question[:, 4])


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
