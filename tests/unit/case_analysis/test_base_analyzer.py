"""Tests for BaseCaseAnalyzer: template lifecycle, hooks, inference loop."""

from utils.case_analysis import BaseCaseAnalyzer, DataFrameSink
from utils.training import BaseTrainer


def test_not_subclass_of_base_trainer():
    assert not issubclass(BaseCaseAnalyzer, BaseTrainer)


def test_construction_order_and_hooks(
    rc, dummy_analyzer_cls, checkpoint_path, monkeypatch
):
    calls = []
    orig_build = dummy_analyzer_cls.build_components
    orig_device = dummy_analyzer_cls.on_device

    def build(self, rc_, ds):
        calls.append("build_components")
        return orig_build(self, rc_, ds)

    def on_device(self, device):
        calls.append("on_device")
        return orig_device(self, device)

    monkeypatch.setattr(dummy_analyzer_cls, "build_components", build)
    monkeypatch.setattr(dummy_analyzer_cls, "on_device", on_device)

    analyzer = dummy_analyzer_cls(rc, None, checkpoint_path, device="cpu")

    assert calls == ["build_components", "on_device"]
    assert analyzer.model.training is False
    assert analyzer.extra_tensor.device.type == "cpu"


def test_run_inference_feeds_sink_per_batch(rc, dummy_analyzer_cls, checkpoint_path):
    analyzer = dummy_analyzer_cls(rc, None, checkpoint_path, device="cpu")
    assert analyzer.run_inference().__len__() == 12
    df = analyzer.sink.result()
    assert list(df["user_id"]) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_run_inference_runs_under_no_grad(rc, dummy_analyzer_cls, checkpoint_path):
    analyzer = dummy_analyzer_cls(rc, None, checkpoint_path, device="cpu")
    observed = {}
    orig = analyzer.forward_pass

    def spy(batch):
        out = orig(batch)
        observed["requires_grad"] = out["y_hat"].requires_grad
        return out

    analyzer.forward_pass = spy
    analyzer.run_inference()
    assert observed["requires_grad"] is False


def test_default_sink_is_dataframe(rc, dummy_analyzer_cls, checkpoint_path):
    analyzer = dummy_analyzer_cls(rc, None, checkpoint_path, device="cpu")
    assert isinstance(analyzer.sink, DataFrameSink)
