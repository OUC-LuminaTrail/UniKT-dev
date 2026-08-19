"""Tests for ``_resolve_stages``: routing, validation, dedup, priority order."""

import pytest

from utils.core import EFFICIENCY_STAGES, register_efficiency_stage
from utils.efficiency.session import _resolve_stages
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
