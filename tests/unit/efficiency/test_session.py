"""Tests for ``_resolve_stages``: routing, validation, dedup, priority order.

Plus ``EfficiencySession.run`` stage-failure isolation: one stage raising
(e.g. CUDA OOM) is recorded in ``report.errors`` while later stages still run.
"""

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from utils.core import EFFICIENCY_STAGES, register_efficiency_stage
from utils.efficiency.session import EfficiencySession, _resolve_stages
from utils.efficiency.stages.base import EfficiencyStage


class _FakeStage(EfficiencyStage):
    """Concrete stage double; priority is set per registered subclass."""

    def run(self, ctx):
        return None

    @classmethod
    def format_table(cls, result):
        return None


@pytest.fixture
def two_fake_stages(registry_snapshot):
    """Register two same-priority stages under known registry keys."""
    for name in ("utest_stage_b", "utest_stage_a"):

        @register_efficiency_stage(name)
        class _Stage(_FakeStage):
            pass

        _Stage.name = name
        _Stage.priority = 50


class TestResolveStages:
    def test_empty_modes_selects_all_stages_by_priority(self):
        stages = _resolve_stages([])
        names = [name for name, _ in stages]
        from utils.core import get_supported_stages

        assert set(names) == set(get_supported_stages())
        priorities = [stage.priority for _, stage in stages]
        assert priorities == sorted(priorities)
        assert names == ["profile", "inference", "windowlate", "train", "trace"]

    def test_unknown_mode_exits_with_available_list(self):
        with pytest.raises(SystemExit, match="Unknown efficiency stage"):
            _resolve_stages(["profile", "utest_no_such_stage"])

    def test_selection_returns_fresh_stage_instances(self):
        stages = _resolve_stages(["train"])
        assert [name for name, _ in stages] == ["train"]
        again = _resolve_stages(["train"])
        assert again[0][1] is not stages[0][1]

    def test_dedup_preserves_first_seen_order(self, two_fake_stages):
        stages = _resolve_stages(["utest_stage_b", "utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_b", "utest_stage_a"]

    def test_priority_sort_is_stable_within_equal_priority(self, two_fake_stages):
        # Both fake stages share priority 50: first-seen order survives the sort.
        stages = _resolve_stages(["utest_stage_a", "utest_stage_b"])
        assert [name for name, _ in stages] == ["utest_stage_a", "utest_stage_b"]

    def test_registry_key_used_as_result_key(self, two_fake_stages):
        # Keys come from the registry name, not the stage's own ClassVar.
        stages = _resolve_stages(["utest_stage_a"])
        assert [name for name, _ in stages] == ["utest_stage_a"]
        assert EFFICIENCY_STAGES.get("utest_stage_a") is not None


class _FakeTarget:
    """Duck-typed BenchmarkTarget: CPU device, one synthetic batch."""

    device = torch.device("cpu")
    model = torch.nn.Linear(2, 2)

    def prepare(self, device):
        pass

    @property
    def train_data(self):
        return [{"questions": torch.zeros(2, 3, dtype=torch.long)}]

    def forward(self, batch):
        return {"y_label": torch.zeros(6)}


def _make_session(tmp_path, modes: str) -> EfficiencySession:
    # eff_cfg must be a real dataclass: session serialization runs it through
    # ``config_to_dict`` → ``asdict``, which rejects SimpleNamespace.
    @dataclass
    class _GeneralCfg:
        # default_factory closure: a plain ``= modes`` default would shadow the
        # enclosing parameter inside the class body.
        modes: str = field(default_factory=lambda: modes)
        resource_sample_interval: float = 0.05

    @dataclass
    class _EffCfg:
        general: _GeneralCfg = field(default_factory=_GeneralCfg)

    rc = SimpleNamespace(
        experiment=SimpleNamespace(model_name="UTestModel"),
        data=SimpleNamespace(dataset="tinyds", max_seq_len=3),
        general=SimpleNamespace(seed=42),
    )
    return EfficiencySession(
        target=_FakeTarget(), rc=rc, eff_cfg=_EffCfg(), output_dir=tmp_path
    )


@pytest.fixture
def fail_then_ok_stages(registry_snapshot):
    """A stage that raises OOM (runs first) and a healthy stage (runs second)."""

    @register_efficiency_stage("utest_fail_stage")
    class _FailStage(_FakeStage):
        priority = 10

        def run(self, ctx):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. (fake)")

    @register_efficiency_stage("utest_ok_stage")
    class _OkStage(_FakeStage):
        priority = 20

        def run(self, ctx):
            return {"ok": True}


class TestStageFailureIsolation:
    def test_failed_stage_recorded_and_later_stages_run(
        self, fail_then_ok_stages, tmp_path
    ):
        report = _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        assert report.results == {"utest_ok_stage": {"ok": True}}
        assert "utest_fail_stage" in report.errors
        assert "OutOfMemoryError" in report.errors["utest_fail_stage"]
        assert report.modes == ["utest_fail_stage", "utest_ok_stage"]

    def test_errors_persist_to_report_json(self, fail_then_ok_stages, tmp_path):
        _make_session(tmp_path, "utest_fail_stage,utest_ok_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert "OutOfMemoryError" in payload["errors"]["utest_fail_stage"]
        assert payload["results"]["utest_ok_stage"] == {"ok": True}

    def test_all_stages_failed_raises_after_writing_report(
        self, fail_then_ok_stages, tmp_path
    ):
        with pytest.raises(RuntimeError, match="All efficiency stages failed"):
            _make_session(tmp_path, "utest_fail_stage").run()
        payload = json.loads((tmp_path / "efficiency_report.json").read_text())
        assert set(payload["errors"]) == {"utest_fail_stage"}
        assert payload["results"] == {}

    def test_no_failure_leaves_errors_empty(self, fail_then_ok_stages, tmp_path):
        report = _make_session(tmp_path, "utest_ok_stage").run()
        assert report.errors == {}
        assert report.results == {"utest_ok_stage": {"ok": True}}
