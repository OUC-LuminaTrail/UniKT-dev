"""Function-level tests for the case_analysis.py CLI commands."""

import json

import numpy as np
import pandas as pd
import pytest

import case_analysis as cli
from utils.case_analysis import DataFrameSink


class _FakeRegistry:
    def __init__(self, mapping):
        self._mapping = mapping

    def __contains__(self, name):
        return name in self._mapping

    def get(self, name):
        return self._mapping.get(name)

    def keys(self):
        return self._mapping.keys()


@pytest.fixture
def run_dir(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for uid in range(40):
        n = 30
        label = rng.integers(0, 2, n)
        p = float(rng.uniform(0.2, 0.9))
        pred = (rng.random(n) < p).astype(int)
        for t in range(n):
            rows.append(
                {
                    "user_id": uid,
                    "question_id": int(rng.integers(0, 50)),
                    "skill": [int(rng.integers(0, 5))],
                    "label": int(label[t]),
                    "prediction": float(pred[t]),
                    "logit": float(pred[t]) - 0.5,
                    "mask": 1,
                    "knowledge_state": [
                        round(float(rng.random()), 3) for _ in range(5)
                    ],
                }
            )
    df = pd.DataFrame(rows)
    df["position"] = df.groupby("user_id").cumcount()
    out = tmp_path / "run"
    (out / "case_analysis").mkdir(parents=True)
    DataFrameSink.save(df, str(out / "case_analysis" / "predictions.parquet"))
    return out


def test_cmd_inference_end_to_end(
    tmp_path, monkeypatch, rc, dummy_analyzer_cls, checkpoint_path
):
    import torch

    run = tmp_path / "fakemodel_run"
    run.mkdir()
    torch.save(
        dummy_analyzer_cls(rc, None, checkpoint_path).model.state_dict(),
        run / "best_model.pth",
    )
    (run / "run_config.yaml").write_text("# stub; loader is patched")

    monkeypatch.setattr("utils.config.load_run_config_archive", lambda *_: rc)
    monkeypatch.setattr(cli, "get_data_source", lambda *_: None)
    monkeypatch.setattr(
        cli, "ANALYZERS", _FakeRegistry({"DummyModel": dummy_analyzer_cls})
    )

    args = type(
        "_Args",
        (),
        {"run_dir": str(run), "sink": "dataframe", "device": "cpu", "batch_size": 2},
    )()
    cli.cmd_inference(args)

    assert (run / "case_analysis" / "predictions.parquet").exists()
    assert (run / "case_analysis" / "user_summaries.parquet").exists()


def test_cmd_inference_unregistered_model_exits(tmp_path, monkeypatch, rc):
    (tmp_path / "best_model.pth").write_bytes(b"x")
    (tmp_path / "run_config.yaml").write_text("")
    monkeypatch.setattr("utils.config.load_run_config_archive", lambda *_: rc)
    monkeypatch.setattr(cli, "ANALYZERS", _FakeRegistry({}))

    args = type(
        "_Args",
        (),
        {
            "run_dir": str(tmp_path),
            "sink": "dataframe",
            "device": None,
            "batch_size": None,
        },
    )()
    with pytest.raises(SystemExit, match="no registered case analyzer"):
        cli.cmd_inference(args)


def _select_args(run_dir, selector):
    return type(
        "_Args",
        (),
        {
            "run_dir": str(run_dir),
            "selector": selector,
            "num_users": 3,
            "min_seq_len": 5,
            "min_error": 0.0,
            "max_error": 1.0,
        },
    )()


def test_cmd_select_writes_selected_users(run_dir):
    cli.cmd_select(_select_args(run_dir, "extreme"))
    path = run_dir / "case_analysis" / "extreme" / "selected_users.json"
    assert path.exists()
    records = json.loads(path.read_text())
    assert len(records) == 3
    assert "user_id" in records[0]


def test_cmd_select_unknown_selector_exits(run_dir):
    with pytest.raises(SystemExit, match="nope"):
        cli.cmd_select(_select_args(run_dir, "nope"))


def test_cmd_plot_renders_figures(run_dir):
    cli.cmd_select(
        type(
            "_Args",
            (),
            {
                "run_dir": str(run_dir),
                "selector": "extreme",
                "num_users": 2,
                "min_seq_len": 5,
                "min_error": 0.0,
                "max_error": 1.0,
            },
        )()
    )
    cli.cmd_plot(
        type(
            "_Args",
            (),
            {
                "run_dir": str(run_dir),
                "selected_users": "extreme",
                "visualizer": "heatmap",
                "max_seq_len": 20,
            },
        )()
    )
    figs = run_dir / "case_analysis" / "extreme" / "figures"
    pngs = [f for f in figs.iterdir() if f.suffix == ".png"]
    assert len(pngs) == 2


def test_filter_supported_options_drops_unsupported():
    class _Sel:
        def select(self, results, *, min_seq_len=20, max_users=20):
            return []

    opts = cli._filter_supported_options(
        _Sel, {"min_seq_len": 5, "error_rate_range": (0.1, 0.9), "max_users": 3}
    )
    assert opts == {"min_seq_len": 5, "max_users": 3}
