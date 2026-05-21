from types import SimpleNamespace

import torch

import model  # noqa: F401
from model.RobustKT.RobustKT_data import RobustKTModelData
from model.RobustKT.RobustKT_model import RobustKT, Smooth
from model.RobustKT.RobustKT_trainer import RobustKTTrainer
from utils.core import PARAM_CONFIGS, TRAINERS
from utils.model_data import SkillModelData


def _args(**overrides):
    defaults = {
        "d_model": 16,
        "n_blocks": 1,
        "num_attn_heads": 4,
        "d_ff": 16,
        "final_fc_dim": 32,
        "kernel_size": 5,
        "dropout": 0.0,
        "kq_same": 1,
        "separate_qa": 0,
        "l2": 1e-5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_robustkt_is_registered():
    assert "RobustKT" in TRAINERS
    assert "RobustKT" in PARAM_CONFIGS


def test_robustkt_uses_skill_model_data():
    assert issubclass(RobustKTModelData, SkillModelData)


def test_robustkt_forward_shape():
    robustkt = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})
    sequence = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    response = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    mask = torch.ones_like(sequence, dtype=torch.bool)
    question = torch.tensor([[10, 11, 12, 13], [1, 2, 3, 4]])

    preds, c_reg_loss = robustkt(sequence, response, mask, question=question)

    assert preds.shape == sequence.shape
    assert c_reg_loss.shape == torch.Size([])


def test_robustkt_forward_pass_skips_first_position():
    trainer = object.__new__(RobustKTTrainer)
    trainer.device_ = torch.device("cpu")
    trainer.model = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})

    def fake_forward(sequence, response, mask=None, question=None):
        return torch.tensor([[0.1, 0.2, 0.3, 0.4]]), torch.tensor(0.0)

    trainer.model.forward = fake_forward

    outputs = trainer.forward_pass(
        (
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[1, 0, 1, 0]]),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[10, 11, 12, 13]]),
        )
    )

    assert torch.allclose(outputs["y_hat"], torch.tensor([0.2, 0.3]))
    assert torch.equal(outputs["y_label"], torch.tensor([0.0, 1.0]))


def test_smooth_is_causal():
    torch.manual_seed(7)
    smooth = Smooth(dropout=0.0, hidden_size=8, kernel_size=5)
    smooth.eval()
    inputs = torch.randn(2, 8, 8)
    changed = inputs.clone()
    changed[:, 4:, :] = torch.randn_like(changed[:, 4:, :]) * 100.0

    out = smooth(inputs)
    changed_out = smooth(changed)

    assert torch.allclose(out[:, :4, :], changed_out[:, :4, :], atol=1e-6)


def test_layernorm_state_dict_backward_compatibility():
    robustkt = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})
    smooth = robustkt.model.smooth

    legacy_state = {}
    for name, param in smooth.state_dict().items():
        legacy_state[name.replace("gamma", "weight").replace("beta", "bias")] = (
            param.clone()
        )

    restored = Smooth(dropout=0.0, hidden_size=8, kernel_size=5)
    restored.load_state_dict(legacy_state)

    assert torch.allclose(restored.layer_norm.gamma, smooth.layer_norm.gamma)
    assert torch.allclose(restored.layer_norm.beta, smooth.layer_norm.beta)


def test_robustkt_pid_padding():
    robustkt = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})
    question = torch.tensor([[10, 11, 12]])
    mask = torch.tensor([[True, False, True]])

    pid = robustkt.build_pid_data(question, mask)

    assert torch.equal(pid, torch.tensor([[11, 0, 13]]))


def test_robustkt_test_forward_pass_returns_group_id():
    trainer = object.__new__(RobustKTTrainer)
    trainer.device_ = torch.device("cpu")
    trainer.model = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})

    sequence = torch.tensor([[0, 1, 2], [2, 3, 4]])
    response = torch.tensor([[1, 0, 1], [0, 1, 0]])
    mask = torch.tensor([[False, True, False], [True, False, True]])
    late_group_id = torch.tensor([[-1, 101, -1], [201, -1, 202]])
    true_labels = torch.tensor([[0, 1, 0], [0, 0, 1]])
    question = torch.tensor([[10, 11, 12], [3, 4, 5]])

    outputs = trainer.test_forward_pass(
        (sequence, response, mask, late_group_id, true_labels, question)
    )

    assert outputs["y_hat"].shape == (3,)
    assert outputs["group_id"].tolist() == [101, 201, 202]
    assert outputs["y_label"].tolist() == [1.0, 0.0, 1.0]


def test_robustkt_test_forward_pass_keeps_history_pid(monkeypatch):
    trainer = object.__new__(RobustKTTrainer)
    trainer.device_ = torch.device("cpu")
    trainer.model = RobustKT(_args(), {"num_skills": 5, "num_questions": 20})

    captured = {}

    def fake_forward(sequence, response, mask=None, question=None):
        captured["mask"] = mask.clone()
        return torch.full(sequence.shape, 0.5), torch.tensor(0.0)

    monkeypatch.setattr(trainer.model, "forward", fake_forward)

    sequence = torch.tensor([[0, 1, 2]])
    response = torch.tensor([[1, 0, 0]])
    target_mask = torch.tensor([[False, False, True]])
    late_group_id = torch.tensor([[101, 102, 103]])
    true_labels = torch.tensor([[1, 0, 1]])
    question = torch.tensor([[10, 11, 12]])

    trainer.test_forward_pass(
        (sequence, response, target_mask, late_group_id, true_labels, question)
    )

    assert torch.equal(captured["mask"], torch.tensor([[True, True, True]]))
