"""Tests for EfficiencyReport serialization and the resource-summary formatter."""

import json

from utils.efficiency.environment import EnvironmentInfo, ResourceSummary
from utils.efficiency.report import EfficiencyReport, _rs
from utils.efficiency.stages.training import TrainingMetrics


def _make_report() -> EfficiencyReport:
    return EfficiencyReport(
        model_name="GIKT",
        dataset_name="assist09",
        timestamp="2026-08-19 00:00:00",
        batch_size=32,
        seq_len=100,
        modes=["train"],
        config={"general": {"warmup_iters": 50}},
        determinism={"seed": 42},
        environment=EnvironmentInfo(device_type="cpu"),
        resource=None,
        results={"train": TrainingMetrics(iters=2, batch_size=32)},
    )


class TestEfficiencyReport:
    def test_to_dict_includes_nested_stage_results(self):
        payload = _make_report().to_dict()
        assert payload["model_name"] == "GIKT"
        assert payload["results"]["train"]["iters"] == 2
        assert payload["environment"]["device_type"] == "cpu"

    def test_write_json_round_trip(self, tmp_path):
        report = _make_report()
        out = tmp_path / "sub" / "efficiency_report.json"
        report.write_json(out)
        payload = json.loads(out.read_text())
        assert payload["model_name"] == "GIKT"
        assert payload["results"]["train"]["iters"] == 2
        assert payload["determinism"] == {"seed": 42}


class TestRsHelper:
    def test_zero_samples_renders_dashes(self):
        assert _rs(ResourceSummary(n=0), "MiB") == ("—", "—")

    def test_values_render_with_unit(self):
        summary = ResourceSummary(mean=10.44, peak=20.46, n=3)
        assert _rs(summary, "MiB") == ("10.4 MiB", "20.5 MiB")

    def test_empty_unit_stripped(self):
        summary = ResourceSummary(mean=10.44, peak=20.46, n=3)
        assert _rs(summary, "") == ("10.4", "20.5")
