"""Shared fixtures for training-area tests: tiny trainers and stub manager.

Doubles follow the area style: duck-typed inline classes, no ``unittest.mock``.
``Loaders'' are plain lists of ``(x, y)`` tensor tuples — not
``torch.utils.data.Dataset`` instances — so ``BaseTrainer._setup_data_loaders``
passes them through verbatim and no DataLoader workers are spawned. All
filesystem effects stay under ``tmp_path`` via ``make_exp_manager``.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from utils.training.base_trainer import BaseTrainer
from utils.training.multi_trainer import MultiTrainer, StageComponents, StageConfig
from utils.training.runtime_components import RuntimeComponents

_UNSET = object()


class StubExpManager:
    """Duck-typed ExperimentManager: only ``get_log_dir`` + ``is_existing_run`` are read."""

    def __init__(self, log_dir, is_existing_run: bool = False):
        self.log_dir = str(log_dir)
        self.is_existing_run = is_existing_run

    def get_log_dir(self) -> str:
        return self.log_dir


@pytest.fixture
def make_exp_manager(tmp_path):
    """Factory: StubExpManager whose log dir is created under tmp_path."""

    def _make(log_dir=None, is_existing_run: bool = False):
        path = tmp_path / (log_dir or "run0")
        path.mkdir(parents=True, exist_ok=True)
        return StubExpManager(path, is_existing_run=is_existing_run)

    return _make


@pytest.fixture
def make_batches():
    """Factory: deterministic (x, y) tuple batches for tiny linear regression.

    Labels are binary (y == x[:, 0]) so val metrics (auc/acc) are defined.
    """

    def _make(batch_sizes=(4, 2)):
        batches = []
        offset = 0
        for size in batch_sizes:
            idx = torch.arange(offset, offset + size)
            x = torch.stack([idx % 2, (idx + 1) % 2], dim=1).float()
            y = (idx % 2).float()
            batches.append((x, y))
            offset += size
        return batches

    return _make


def _tiny_outputs(model, batch_data):
    """Shared forward for the tiny trainers: linear model over (x, y) tuples."""
    x, y = batch_data
    y_hat = model(x).reshape(-1)
    return {
        "y_hat": y_hat,
        "y_label": y.float(),
        "y_predict": (y_hat >= 0.5).to(torch.int),
        "y_score": y_hat,
        "y_prob": torch.sigmoid(y_hat),
    }


class TinyTrainer(BaseTrainer):
    """Minimal concrete single-stage trainer: Linear(2, 1) + MSELoss + SGD."""

    def __init__(
        self,
        rc,
        exp_manager,
        train,
        val=None,
        test=None,
        extra_callbacks=(),
        max_clip_grad_norm=None,
        lr_scheduler=None,
    ):
        # Stash before super().__init__ — build_components reads these.
        self._tiny = {
            "train": train,
            "val": val,
            "test": test,
            "extra": list(extra_callbacks),
            "clip": max_clip_grad_norm,
            "sched": lr_scheduler,
        }
        super().__init__(rc, None, exp_manager)

    def build_components(self, rc, data_src) -> RuntimeComponents:
        torch.manual_seed(7)
        model = torch.nn.Linear(2, 1)
        return RuntimeComponents(
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            loss_fn=torch.nn.MSELoss(),
            lr_scheduler=self._tiny["sched"],
            train_data=self._tiny["train"],
            val_data=self._tiny["val"],
            test_data=self._tiny["test"],
            max_clip_grad_norm=self._tiny["clip"],
        )

    def build_callbacks(self):
        return list(self._tiny["extra"])

    def forward_pass(self, batch_data):
        return _tiny_outputs(self.model, batch_data)


@pytest.fixture
def make_tiny_trainer(make_run_config, make_exp_manager, make_batches):
    """Factory: a built TinyTrainer on CPU with overridable rc/exp/data/callbacks."""

    def _make(
        *,
        rc=None,
        exp=None,
        train=None,
        val=_UNSET,
        test=None,
        extra_callbacks=(),
        max_clip_grad_norm=None,
        lr_scheduler=None,
        model_kwargs=None,
    ):
        rc = rc or make_run_config(model_kwargs={"epochs": 1, **(model_kwargs or {})})
        exp = exp or make_exp_manager()
        return TinyTrainer(
            rc,
            exp,
            make_batches() if train is None else train,
            val=make_batches() if val is _UNSET else val,
            test=test,
            extra_callbacks=extra_callbacks,
            max_clip_grad_norm=max_clip_grad_norm,
            lr_scheduler=lr_scheduler,
        )

    return _make


class TinyMultiTrainer(MultiTrainer):
    """Two tiny stages exercising build_stages / _apply_stage / on_stage_*.

    ``stage_specs`` maps stage name to builder options (``epochs``,
    ``early_stopping``, ``checkpoint_monitor`` / ``checkpoint_mode``); every
    lifecycle event is appended to ``events`` so tests can assert ordering.
    """

    def __init__(
        self, rc, exp_manager, train, val=None, stage_specs=None, extra_callbacks=()
    ):
        self.events = []
        self.stage_snapshots = {}
        self._extra = list(extra_callbacks)
        self._specs = stage_specs or {
            "km": {"epochs": 1, "early_stopping": _default_es()},
            "am": {"epochs": 1, "early_stopping": _default_es()},
        }
        self._train = train
        self._val = val
        super().__init__(rc, None, exp_manager)

    def build_stages(self):
        return [StageConfig(name, self._make_builder(name)) for name in self._specs]

    def build_callbacks(self):
        return list(self._extra)

    def _make_builder(self, name):
        def _build():
            self.events.append(("build", name))
            spec = self._specs[name]
            torch.manual_seed(7)
            model = torch.nn.Linear(2, 1)
            return StageComponents(
                model=model,
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                loss_fn=torch.nn.MSELoss(),
                train_data=self._train,
                val_data=self._val,
                epochs=spec.get("epochs", 1),
                early_stopping=spec.get("early_stopping"),
                checkpoint_monitor=spec.get("checkpoint_monitor"),
                checkpoint_mode=spec.get("checkpoint_mode"),
            )

        return _build

    def on_stage_begin(self, name):
        self.events.append(("begin", name))

    def on_stage_complete(self, name, result):
        from utils.training.callbacks import (
            CheckpointCallback,
            EarlyStoppingCallback,
        )

        self.events.append(("complete", name, result))
        checkpoint_cb = self.callback_manager.get_callback(CheckpointCallback)
        self.stage_snapshots[name] = {
            "has_es": self.early_stopping is not None,
            "has_es_cb": self.callback_manager.get_callback(EarlyStoppingCallback)
            is not None,
            "metric_step_offset": self._metric_step_offset,
            "callback_manager": self.callback_manager,
            "best_filename": checkpoint_cb.best_filename,
            "monitor_override": checkpoint_cb._monitor_override,
            "mode_override": checkpoint_cb._mode_override,
            "es_monitor": self.early_stopping.cfg.monitor
            if self.early_stopping
            else None,
        }

    def forward_pass(self, batch_data):
        return _tiny_outputs(self.model, batch_data)


def _default_es():
    from utils.config.run_config import EarlyStoppingConfig

    return EarlyStoppingConfig(patience=2)


class FakeCloudBackend:
    """Recording stand-in for a swanlab/wandb SDK module."""

    def __init__(self):
        self.calls = []

    def login(self, **kwargs):
        self.calls.append(("login", kwargs))

    def Settings(self):
        self.calls.append(("settings",))
        return object()

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))

    def log(self, data, **kwargs):
        self.calls.append(("log", dict(data), kwargs))

    def finish(self):
        self.calls.append(("finish",))


@pytest.fixture
def inject_fake_cloud_logger(monkeypatch):
    """Install a recording fake ``swanlab``/``wandb`` into sys.modules.

    Both cloud backends lazy-import their SDK inside every method, so a
    ``sys.modules`` entry is enough to intercept them. Returns the recording
    backend; ``monkeypatch.setitem`` restores the real modules on teardown.
    """

    def _install(name):
        backend = FakeCloudBackend()
        mod = types.ModuleType(name)
        if name == "swanlab":
            mod.login = backend.login
            mod.init = backend.init
            mod.Settings = backend.Settings
            mod.log = backend.log
            mod.finish = backend.finish

            exceptions = types.ModuleType(f"{name}.exceptions")

            class AuthenticationError(Exception):
                pass

            exceptions.AuthenticationError = AuthenticationError
            mod.exceptions = exceptions

            notification = types.ModuleType(f"{name}.plugin.notification")

            class LarkCallback:
                def __init__(self, **kwargs):
                    pass

            notification.LarkCallback = LarkCallback
            plugin = types.ModuleType(f"{name}.plugin")
            plugin.notification = notification
            mod.plugin = plugin

            monkeypatch.setitem(sys.modules, name, mod)
            monkeypatch.setitem(sys.modules, f"{name}.exceptions", exceptions)
            monkeypatch.setitem(sys.modules, f"{name}.plugin", plugin)
            monkeypatch.setitem(
                sys.modules, f"{name}.plugin.notification", notification
            )
        else:
            mod.init = backend.init
            mod.log = backend.log
            mod.finish = backend.finish
            monkeypatch.setitem(sys.modules, name, mod)
        return backend

    return _install
