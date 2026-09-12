from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from model.HDHKT.HDHKT_model import HDHKT, InformationBottleneckMoE
from model.HDHKT.HDHKT_trainer import HDHKTConfig, HDHKTTrainer
from model.layers import Hypergraph


def _relations() -> tuple[torch.Tensor, torch.Tensor]:
    question_skill = torch.tensor(
        [[0, 0, 1, 2, 2, 3, 4, 5], [0, 1, 1, 0, 2, 2, 1, 0]],
        dtype=torch.long,
    )
    question_hyperedge = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5], [0, 0, 1, 1, 2, 1, 2, 2]],
        dtype=torch.long,
    )
    return question_skill, question_hyperedge


def test_three_code_ib_component_smoke() -> None:
    torch.manual_seed(7)
    dim = 8
    module = InformationBottleneckMoE(
        dim=dim,
        num_skills=3,
        num_hyperedges=3,
        dropout=0.0,
        negative_samples=2,
    )

    # Exactly three semantic encoder modules feed a three-code router.
    assert hasattr(module, "private1")
    assert hasattr(module, "private2")
    assert hasattr(module, "common")
    assert not hasattr(module, "common1")
    assert not hasattr(module, "common2")
    assert module.router[0].in_features == 3 * dim

    view1 = torch.randn(6, dim, requires_grad=True)
    view2 = torch.randn(6, dim, requires_grad=True)
    question_skill, question_hyperedge = _relations()
    source_ids = torch.arange(6)

    common_outputs: list[torch.Tensor] = []
    shared_inputs: list[torch.Tensor] = []
    common_hook = module.common.register_forward_hook(
        lambda _module, _inputs, output: common_outputs.append(output.detach())
    )
    shared_hook = module.shared_expert[0].register_forward_pre_hook(
        lambda _module, inputs: shared_inputs.append(inputs[0].detach())
    )
    module.train()
    fused, auxiliary = module(
        view1,
        view2,
        question_skill_edges=question_skill,
        question_hyperedge_edges=question_hyperedge,
        source_ids=source_ids,
    )
    common_hook.remove()
    shared_hook.remove()

    assert fused.shape == (6, dim)
    assert set(auxiliary) == {
        "private_loss",
        "club_loss",
        "common_loss",
        "club_fit_loss",
    }
    assert all(torch.isfinite(value) for value in auxiliary.values())

    # Every row consumed by the shared expert is selected from one of the two
    # conditional paths, hence is a sample from their equal mixture.
    assert len(common_outputs) == 2
    assert len(shared_inputs) == 1
    mixture_sample = shared_inputs[0]
    from_view1 = torch.isclose(mixture_sample, common_outputs[0]).all(dim=-1)
    from_view2 = torch.isclose(mixture_sample, common_outputs[1]).all(dim=-1)
    assert torch.all(from_view1 | from_view2)
    assert from_view1.any() and from_view2.any()

    common_grad1, common_grad2 = torch.autograd.grad(
        auxiliary["common_loss"], (view1, view2), retain_graph=True
    )
    assert torch.isfinite(common_grad1).all() and common_grad1.abs().sum() > 0
    assert torch.isfinite(common_grad2).all() and common_grad2.abs().sum() > 0

    total_auxiliary = sum(auxiliary.values())
    total_auxiliary.backward()
    assert view1.grad is not None and torch.isfinite(view1.grad).all()
    assert view2.grad is not None and torch.isfinite(view2.grad).all()
    assert module.common.projection.weight.grad is not None

    # Eval consumes the exact mean of the two deterministic common paths.
    module.eval()
    eval_shared_inputs: list[torch.Tensor] = []
    eval_hook = module.shared_expert[0].register_forward_pre_hook(
        lambda _module, inputs: eval_shared_inputs.append(inputs[0].detach())
    )
    with torch.no_grad():
        c1 = module.common(view1)
        c2 = module.common(view2)
        eval_fused1, eval_auxiliary = module(view1, view2)
        eval_fused2, _ = module(view1, view2)
    eval_hook.remove()
    assert eval_auxiliary == {}
    assert torch.allclose(eval_shared_inputs[0], 0.5 * (c1 + c2))
    assert torch.allclose(eval_fused1, eval_fused2)


def _tiny_hetero_graph() -> HeteroData:
    graph = HeteroData()
    sizes = {"question": 6, "skill": 3, "assignment": 2, "template": 2}
    for node_type, size in sizes.items():
        graph[node_type].num_nodes = size
        ids = torch.arange(size)
        graph[node_type, "self", node_type].edge_index = torch.stack([ids, ids])
    return graph


def test_full_hdhkt_three_code_forward_backward_smoke() -> None:
    torch.manual_seed(11)
    graph = _tiny_hetero_graph()
    hypergraph = Hypergraph(
        num_v=6,
        e_list=[[0, 1], [1, 2, 4], [3, 4, 5]],
    )
    metadata = {
        "num_questions": 6,
        "num_skills": 3,
        "num_assignments": 2,
        "num_templates": 2,
    }
    model = HDHKT(
        data_metadata=metadata,
        hetero_metadata=graph.metadata(),
        hidden_dim=8,
        n_hop=1,
        dropout=0.0,
        history_neighbour=2,
        num_hyperedges=hypergraph.num_e,
        ib_negative_samples=2,
    )
    sequence = torch.tensor([[0, 1, 2, 3], [2, 4, 5, 1]])
    response = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 1]])
    mask = torch.ones_like(sequence, dtype=torch.bool)
    padding = metadata["num_skills"]
    skill_ids = torch.tensor(
        [[0, 1], [1, padding], [0, 2], [2, padding], [1, 2], [0, padding]]
    )

    model.train()
    logits, auxiliary = model(
        sequence,
        response,
        mask,
        graph,
        hypergraph,
        skill_ids,
        return_auxiliary=True,
    )
    assert logits.shape == sequence.shape
    assert set(auxiliary) == {
        "private_loss",
        "club_loss",
        "common_loss",
        "club_fit_loss",
    }
    loss = logits.square().mean() + sum(auxiliary.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fuse.common.projection.weight.grad is not None

    model.eval()
    with torch.no_grad():
        eval_logits1 = model(sequence, response, mask, graph, hypergraph, skill_ids)
        eval_logits2 = model(sequence, response, mask, graph, hypergraph, skill_ids)
    assert torch.allclose(eval_logits1, eval_logits2)


def test_three_code_trainer_loss_matches_derived_objective() -> None:
    config = HDHKTConfig()
    assert not hasattr(config, "ib_align_weight")
    assert not hasattr(config, "ib_route_weight")
    assert not hasattr(config, "ib_rate_capacity")
    assert not hasattr(config, "ib_rate_weight")

    trainer = HDHKTTrainer.__new__(HDHKTTrainer)
    trainer.run_config = SimpleNamespace(model=config)
    trainer.loss = torch.nn.BCEWithLogitsLoss()
    outputs = {
        "y_hat": torch.tensor([0.2, -0.4]),
        "y_label": torch.tensor([1.0, 0.0]),
        "_ib_private_loss": torch.tensor(1.0),
        "_ib_club_loss": torch.tensor(2.0),
        "_ib_common_loss": torch.tensor(3.0),
        "_ib_club_fit_loss": torch.tensor(4.0),
    }
    actual = trainer._compute_loss(outputs)
    expected = trainer.loss(outputs["y_hat"], outputs["y_label"])
    expected = expected + config.ib_private_weight * outputs["_ib_private_loss"]
    expected = expected + config.ib_club_weight * outputs["_ib_club_loss"]
    expected = expected + config.ib_common_weight * outputs["_ib_common_loss"]
    expected = expected + config.ib_club_fit_weight * outputs["_ib_club_fit_loss"]
    assert torch.allclose(actual, expected)
