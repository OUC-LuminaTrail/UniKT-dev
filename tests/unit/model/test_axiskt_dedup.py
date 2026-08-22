"""Equivalence regression for the AxisKT dedup refactor.

The refactor threads precomputed question-derived tensors (the
``question_skill_ids`` gather, ``skill_embed`` / ``skill_change`` lookups, and
the ``_question_vector`` projection) through the forward pass instead of
re-gathering and re-embedding them inside each sub-method. It must not change
the forward output or any backward gradient.

The oracle is the golden snapshot produced by
``tests/fixtures/generate_axiskt_dedup_golden.py``, captured before the refactor.
"""

from pathlib import Path

import pytest
import torch

AxisKT = pytest.importorskip("model.AxisKT.AxisKT_model").AxisKT

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="AxisKT requires CUDA"
)

DEVICE = torch.device("cuda")

# ``tests/unit/model/test_axiskt_dedup.py`` -> ``tests/fixtures/...``.
_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures" / "axiskt_dedup_golden.pt"
)


@pytest.fixture(scope="module")
def golden():
    # Trusted local fixture authored by the generator script in this repo.
    # The snapshot is stored on CPU (see the generator) and moved to CUDA here.
    return torch.load(_FIXTURE_PATH, weights_only=False, map_location="cpu")


def _build_model(golden):
    model = AxisKT(**golden["kwargs"]).train()
    model.load_state_dict(golden["state_dict"])
    return model.to(DEVICE)


def test_forward_matches_golden(golden):
    model = _build_model(golden)
    logits = model(
        golden["questions"].to(DEVICE),
        golden["responses"].to(DEVICE),
        golden["times"].to(DEVICE),
        golden["mask"].to(DEVICE),
    )
    # The refactor is a pure equivalence (shared tensors), so the output must
    # match bit-for-bit, not just within tolerance.
    torch.testing.assert_close(logits, golden["logits"].to(DEVICE), rtol=0, atol=0)


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
        golden_grad = golden["grads"].get(name)
        if golden_grad is None:
            continue
        assert param.grad is not None, f"missing gradient for {name}"
        # Forward is bit-exact; gradients match only to floating-point noise
        # because sharing intermediate nodes changes autograd's accumulation
        # order, which is mathematically equivalent.
        torch.testing.assert_close(
            param.grad, golden_grad.to(DEVICE), rtol=1e-5, atol=1e-6
        )


def test_forward_dedups_question_derived_lookups(golden):
    """Each shared lookup runs once for the question view plus once for the
    packed (sorted) KC stream — not once per consumer."""
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

    # skill_embed: one packed-stream gather shared by the event pooling, the
    # scan input, and the readout. Pre-refactor this was 3, then 2 with
    # per-branch packing, 1 since forward packs once for all consumers.
    assert counts["skill_embed"] == 1
    if model.question_embed is not None:
        # question_embed: shared question vector (event + readout + static) +
        # the packed question stream. Pre-refactor this was 4.
        assert counts["question_embed"] == 2
