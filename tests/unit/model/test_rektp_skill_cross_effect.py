import pytest
import torch

from model.ReKTP.ReKTP_model import ReKTP
from model.ReKTP.skill_cross_effect import (
    SkillCrossEffect,
    diagonal_affine_exclusive_scan,
)


def _sequential_exclusive_scan(decay, write):
    state = torch.zeros_like(write[:, 0])
    states = []
    for position in range(write.size(1)):
        states.append(state)
        state = decay[:, position] * state + write[:, position]
    return torch.stack(states, dim=1)


def _cross_module():
    module = SkillCrossEffect(
        num_skills=2,
        rank=1,
        num_scales=2,
        max_gap_bins=4,
    ).eval()
    with torch.no_grad():
        module.source_embed.weight.zero_()
        module.target_embed.weight.zero_()
        module.logit_scale.fill_(10.0)
    return module


def _single_skill_inputs(responses, times=None):
    skill_ids = torch.tensor([[[0], [1], [0]]])
    skill_mask = torch.ones_like(skill_ids, dtype=torch.bool)
    mask = torch.ones(1, 3, dtype=torch.bool)
    if times is None:
        times = torch.tensor([[0.0, 1.0, 2.0]])
    return skill_ids, skill_mask, responses, times, mask


def _model_kwargs(**overrides):
    kwargs = {
        "data_metadata": {"num_questions": 3, "num_skills": 2},
        "question_skill_ids": torch.tensor([[0, 2], [1, 2], [0, 1]]),
        "question_skill_mask": torch.tensor(
            [[True, False], [True, False], [True, True]]
        ),
        "hidden_dim": 16,
        "n_blocks": 1,
        "encoder_type": "lstm",
        "max_gap_bins": 4,
        "dropout": 0.0,
    }
    kwargs.update(overrides)
    return kwargs


def test_work_efficient_scan_matches_sequential_recurrence():
    torch.manual_seed(0)
    decay = torch.rand(2, 7, 3, 2) * 0.4 + 0.5
    write = torch.randn_like(decay)

    actual = diagonal_affine_exclusive_scan(decay, write)
    expected = _sequential_exclusive_scan(decay, write)

    torch.testing.assert_close(actual, expected)


def test_cross_effect_is_directional_and_response_conditioned():
    module = _cross_module()
    with torch.no_grad():
        # Persistent scale: (skill 0, correct) excites target skill 1 only.
        module.source_embed.weight[2, 0] = 2.0
        module.target_embed.weight[1, 0] = 3.0

    correct_inputs = _single_skill_inputs(torch.tensor([[1, 0, 0]]))
    wrong_inputs = _single_skill_inputs(torch.tensor([[0, 0, 0]]))
    correct = module(*correct_inputs)
    wrong = module(*wrong_inputs)

    assert correct[0, 1] > 5.9
    assert wrong[0, 1] == 0.0

    reverse_skill_ids = torch.tensor([[[1], [0], [1]]])
    reverse = module(
        reverse_skill_ids,
        torch.ones_like(reverse_skill_ids, dtype=torch.bool),
        torch.tensor([[1, 0, 0]]),
        torch.tensor([[0.0, 1.0, 2.0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert reverse[0, 1] == 0.0


def test_cross_effect_decays_with_elapsed_time():
    module = _cross_module()
    with torch.no_grad():
        # Index 1 is the positive-rate scale; index 0 is persistent.
        module.source_embed.weight[2, 1] = 1.0
        module.target_embed.weight[1, 1] = 1.0

    responses = torch.tensor([[1, 0, 0]])
    short = module(*_single_skill_inputs(responses, torch.tensor([[0.0, 1.0, 2.0]])))
    long = module(*_single_skill_inputs(responses, torch.tensor([[0.0, 4.0, 5.0]])))

    assert short[0, 1] > long[0, 1] > 0.0
    torch.testing.assert_close(short[0, 1], 8.0 * long[0, 1])


def test_current_response_does_not_enter_its_own_cross_effect():
    module = _cross_module()
    with torch.no_grad():
        module.source_embed.weight[:, 0] = torch.arange(4, dtype=torch.float32)
        module.target_embed.weight[:2, 0] = 1.0

    baseline_inputs = _single_skill_inputs(torch.tensor([[1, 0, 1]]))
    changed_inputs = _single_skill_inputs(torch.tensor([[1, 1, 1]]))
    baseline = module(*baseline_inputs)
    changed = module(*changed_inputs)

    torch.testing.assert_close(changed[:, 1], baseline[:, 1])
    assert not torch.allclose(changed[:, 2], baseline[:, 2])


def test_cross_effect_is_invariant_to_trailing_padding():
    module = _cross_module()
    with torch.no_grad():
        module.source_embed.weight[2, 0] = 2.0
        module.target_embed.weight[1, 0] = 3.0
    short_inputs = _single_skill_inputs(torch.tensor([[1, 0, 0]]))
    short = module(*short_inputs)

    padded_skill_ids = torch.tensor([[[0], [1], [0], [1], [0]]])
    padded_mask = torch.tensor([[True, True, True, False, False]])
    padded = module(
        padded_skill_ids,
        torch.ones_like(padded_skill_ids, dtype=torch.bool),
        torch.tensor([[1, 0, 0, 1, 1]]),
        torch.tensor([[0.0, 1.0, 2.0, 0.0, 0.0]]),
        padded_mask,
    )

    torch.testing.assert_close(padded[:, :3], short)
    assert torch.all(padded[:, 3:] == 0.0)


def test_cross_effect_scan_has_finite_embedding_gradients():
    module = SkillCrossEffect(2, rank=3, num_scales=4, max_gap_bins=5)
    with torch.no_grad():
        module.logit_scale.fill_(0.5)
    inputs = _single_skill_inputs(torch.tensor([[1, 0, 1]]))

    loss = module(*inputs).square().sum()
    loss.backward()

    for parameter in (
        module.source_embed.weight,
        module.target_embed.weight,
        module.logit_scale,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_zero_residual_initialization_still_trains_target_factor():
    module = SkillCrossEffect(2, rank=3, num_scales=4, max_gap_bins=5)
    inputs = _single_skill_inputs(torch.tensor([[1, 0, 1]]))

    scores = module(*inputs)
    assert torch.all(scores == 0.0)
    scores.sum().backward()

    assert module.target_embed.weight.grad is not None
    assert module.target_embed.weight.grad.norm() > 0.0
    assert torch.isfinite(module.target_embed.weight.grad).all()


def test_disabled_and_zero_initialized_enabled_models_match():
    baseline = ReKTP(**_model_kwargs()).eval()
    enabled = ReKTP(
        **_model_kwargs(
            use_skill_cross_effect=True,
            cross_effect_rank=2,
            cross_effect_num_scales=2,
        )
    ).eval()
    enabled.load_state_dict(baseline.state_dict(), strict=False)
    questions = torch.tensor([[0, 1, 2, 0]])
    responses = torch.tensor([[1, 0, 1, 0]])
    times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.ones_like(questions, dtype=torch.bool)

    with torch.no_grad():
        baseline_logits = baseline(questions, responses, times, mask)
        enabled_logits = enabled(questions, responses, times, mask)

    assert torch.all(enabled.skill_cross_effect.target_embed.weight == 0.0)
    torch.testing.assert_close(enabled_logits, baseline_logits)


def test_active_cross_effect_changes_next_item_logit_without_target_leakage():
    model = ReKTP(
        **_model_kwargs(
            use_skill_cross_effect=True,
            cross_effect_rank=1,
            cross_effect_num_scales=2,
        )
    ).eval()
    cross = model.skill_cross_effect
    with torch.no_grad():
        cross.source_embed.weight.zero_()
        cross.target_embed.weight.zero_()
        cross.source_embed.weight[2, 0] = 2.0
        cross.target_embed.weight[1, 0] = 3.0
        cross.logit_scale.fill_(10.0)

    questions = torch.tensor([[0, 1, 0]])
    responses = torch.tensor([[1, 0, 0]])
    times = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.ones_like(questions, dtype=torch.bool)
    baseline = model(questions, responses, times, mask)
    changed_responses = responses.clone()
    changed_responses[:, 1] = 1
    changed = model(questions, changed_responses, times, mask)

    assert baseline[0, 0] > 5.9
    torch.testing.assert_close(changed[:, 0], baseline[:, 0])


def test_cross_effect_validates_state_dimensions():
    with pytest.raises(ValueError, match="rank"):
        SkillCrossEffect(2, rank=0, num_scales=2, max_gap_bins=4)

    with pytest.raises(ValueError, match="num_scales"):
        SkillCrossEffect(2, rank=2, num_scales=1, max_gap_bins=4)
