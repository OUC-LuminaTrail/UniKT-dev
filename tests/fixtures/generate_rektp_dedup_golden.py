"""Generate the ReKTP dedup-equivalence golden snapshot.

Captures the forward output and backward gradients of a fixed model state on a
fixed input. The dedup refactor must reproduce these bit-for-bit, so this
snapshot is the equivalence oracle for ``tests/unit/model/test_rektp_dedup.py``.

ReKTP requires CUDA, so the snapshot is captured on the default GPU device;
all tensors are saved on CPU and moved back by the test. Regenerate only after
an intentional numerical change::

    pixi run python tests/fixtures/generate_rektp_dedup_golden.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model.ReKTP.ReKTP_model import ReKTP


def build_kwargs():
    """Five questions over three skills; skill id 3 is the padding sentinel."""
    question_skill_ids = torch.tensor([[0, 3], [1, 3], [0, 1], [2, 0], [1, 2]])
    question_skill_mask = torch.tensor(
        [[True, False], [True, False], [True, True], [True, True], [True, True]]
    )
    return {
        "data_metadata": {"num_questions": 5, "num_skills": 3},
        "question_skill_ids": question_skill_ids,
        "question_skill_mask": question_skill_mask,
        "hidden_dim": 16,
        "n_blocks": 1,
        "max_gap_bins": 4,
        "dropout": 0.0,
        # question_embed_dim left default (== hidden_dim) so the question-vector
        # sharing path is exercised.
    }


def build_model():
    if not torch.cuda.is_available():
        raise RuntimeError("ReKTP requires CUDA to generate the golden snapshot")
    torch.manual_seed(1234)
    model = ReKTP(**build_kwargs()).to("cuda")
    with torch.no_grad():
        # The IRT head and several local layers are zero-initialised; activate
        # them so the logits and gradients are non-trivial and a regression
        # cannot hide behind identically-zero output.
        model.ability_head.weight.normal_(0.0, 0.2)
        model.question_diff.weight.normal_(0.0, 0.2)
        model.local_write.weight.normal_(0.0, 0.05)
        model.local_readout.weight.normal_(0.0, 0.05)
        model.global_ffn[0].weight.normal_(0.0, 0.05)
        model.global_ffn[3].weight.normal_(0.0, 0.05)
    return model


def main():
    device = torch.device("cuda")
    model = build_model().train()
    questions = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 0, 4, 3, 1]], device=device)
    responses = torch.tensor([[1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1]], device=device)
    times = torch.tensor(
        [[0.0, 1.0, 2.0, 4.0, 8.0, 16.0], [0.0, 3.0, 9.0, 12.0, 15.0, 20.0]],
        dtype=torch.float64,
        device=device,
    )
    mask = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, True, True, True]],
        device=device,
    )

    logits = model(questions, responses, times, mask)
    loss = logits[:, :-1].square().mean()
    loss.backward()

    grads = {
        name: param.grad.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    snapshot = {
        "kwargs": build_kwargs(),
        "state_dict": {
            name: tensor.cpu() for name, tensor in model.state_dict().items()
        },
        "questions": questions.cpu(),
        "responses": responses.cpu(),
        "times": times.cpu(),
        "mask": mask.cpu(),
        "logits": logits.detach().cpu(),
        "grads": grads,
    }
    out_path = Path(__file__).parent / "rektp_dedup_golden.pt"
    torch.save(snapshot, out_path)
    print(f"saved {out_path}")
    print(f"  logits shape={tuple(logits.shape)}, max|logits|={logits.abs().max():.4f}")
    print(f"  {len(grads)} parameter gradients captured")


if __name__ == "__main__":
    main()
