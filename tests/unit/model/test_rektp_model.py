import pytest
import torch

ReKTP = pytest.importorskip("model.ReKTP.ReKTP_model").ReKTP


def _build_model(device):
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
    return model.to(device).eval()


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

    baseline = model._question_pre_states(
        questions, responses, mask, event_embeddings
    )
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
