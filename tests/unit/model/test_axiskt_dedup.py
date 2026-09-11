"""Golden-snapshot regression for AxisKT."""

from pathlib import Path

import pytest
import torch

AxisKT = pytest.importorskip("model.AxisKT.AxisKT_model").AxisKT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AxisKT requires CUDA"
)

DEVICE = torch.device("cuda")

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "axiskt_dedup_golden.pt"
)


@pytest.fixture(scope="module")
def golden():
    return torch.load(_FIXTURE_PATH, weights_only=False, map_location="cpu")


def _fold_fixture_state_dict(golden):
    state = {name: tensor.clone() for name, tensor in golden["state_dict"].items()}
    if "local_decay_logits" in state:
        return state
    gap_embedding = state.pop("gap_embed.weight")
    decay_weight = state.pop("local_decay.weight")
    decay_bias = state.pop("local_decay.bias")
    state["local_decay_logits"] = torch.nn.functional.linear(
        gap_embedding, decay_weight, decay_bias
    )
    hidden_dim = golden["kwargs"]["hidden_dim"]
    state["local_readout.weight"] = state["local_readout.weight"][:, : 2 * hidden_dim]
    state.pop("local_readout.bias")
    return state


def _build_model(golden):
    model = AxisKT(**golden["kwargs"]).train()
    model.load_state_dict(_fold_fixture_state_dict(golden))
    return model.to(DEVICE)


def test_forward_matches_golden(golden):
    model = _build_model(golden)
    logits = model(
        golden["questions"].to(DEVICE),
        golden["responses"].to(DEVICE),
        golden["times"].to(DEVICE),
        golden["mask"].to(DEVICE),
    )
    torch.testing.assert_close(
        logits, golden["logits"].to(DEVICE), rtol=1e-5, atol=1e-6
    )


def test_backward_gradients_match_golden(golden):
    model = _build_model(golden)
    logits = model(
        golden["questions"].to(DEVICE),
        golden["responses"].to(DEVICE),
        golden["times"].to(DEVICE),
        golden["mask"].to(DEVICE),
    )
    logits[:, :-1].square().mean().backward()

    for name, param in model.named_parameters():
        if name == "local_decay_logits":
            continue
        golden_grad = golden["grads"].get(name)
        if golden_grad is None:
            continue
        if name == "local_readout.weight":
            golden_grad = golden_grad[:, : param.shape[1]]
        assert param.grad is not None, f"missing gradient for {name}"
        torch.testing.assert_close(
            param.grad, golden_grad.to(DEVICE), rtol=1e-5, atol=1e-6
        )


def test_forward_dedups_question_derived_lookups(golden):
    """Count shared question-derived lookups."""
    model = _build_model(golden)
    counts = {"skill_embed": 0, "question_embed": 0}

    def wrap(key, fn):
        def _wrapped(*args, **kwargs):
            counts[key] += 1
            return fn(*args, **kwargs)

        return _wrapped

    model.skill_embed.forward = wrap("skill_embed", model.skill_embed.forward)
    if model.question_embed is not None:
        model.question_embed.forward = wrap(
            "question_embed", model.question_embed.forward
        )

    model(
        golden["questions"].to(DEVICE),
        golden["responses"].to(DEVICE),
        golden["times"].to(DEVICE),
        golden["mask"].to(DEVICE),
    )

    # One packed-stream gather shared by the event pooling, the scan input,
    # and the readout.
    assert counts["skill_embed"] == 1
    if model.question_embed is not None:
        # Shared question vector (event + readout + static) plus the packed
        # question stream.
        assert counts["question_embed"] == 2
