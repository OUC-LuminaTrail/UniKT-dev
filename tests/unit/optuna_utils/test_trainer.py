"""Tests for TrainerObjectiveWrapper: rc isolation, metric extraction, error paths.

The trainer is a duck-typed double: the wrapper only touches ``add_callback``,
``run``, and the ``early_stopping`` attribute after construction.
"""

import optuna
import pytest

from utils.optuna_utils.trainer import TrainerObjectiveWrapper


class _FakeEarlyStopping:
    def __init__(self, best_metrics=None, best_score=None, best_epoch=None):
        self.best_metrics = best_metrics or {}
        self.best_score = best_score
        self.best_epoch = best_epoch


class _FakeTrainer:
    """Records callbacks; optional early_stopping attrs; optional failure."""

    def __init__(self, early_stopping=None, run_error: Exception | None = None):
        self.callbacks = []
        self.early_stopping = early_stopping
        self._run_error = run_error

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def run(self):
        if self._run_error is not None:
            raise self._run_error


class _TrainerFactory:
    """Trainer-class double: builds a fake trainer, counts constructions."""

    instances: list[_FakeTrainer]

    def __init__(self, **trainer_kwargs):
        self._kwargs = trainer_kwargs
        self.instances = []

    def __call__(self, *, rc, data_src, exp_manager):
        trainer = _FakeTrainer(**self._kwargs)
        trainer.rc = rc
        self.instances.append(trainer)
        return trainer


def _make_wrapper(base_rc, trainer_factory, metric="auc"):
    return TrainerObjectiveWrapper(
        trainer_class=trainer_factory,
        data_src_fn=lambda: None,
        base_rc=base_rc,
        metric_name=metric,
    )


class _FakePruningCallback:
    def __init__(self, best_value=None, pruned=False):
        self.best_value = best_value
        self.pruned = pruned


# --- _create_trial_rc ---


class TestCreateTrialRc:
    def test_params_applied_to_model_node(self, make_run_config, registry_snapshot):
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trial_rc = wrapper._create_trial_rc({"hidden_dim": 64})
        assert trial_rc.model.hidden_dim == 64

    def test_base_rc_untouched_by_trial_mutation(self, make_run_config):
        base = make_run_config()
        wrapper = _make_wrapper(base, _TrainerFactory())
        trial_rc = wrapper._create_trial_rc({"hidden_dim": 64})
        trial_rc.model.hidden_dim = 128
        trial_rc.general.seed = 0
        assert base.model.hidden_dim == 8
        assert base.general.seed == 42

    def test_trials_are_independent_copies(self, make_run_config):
        base = make_run_config()
        wrapper = _make_wrapper(base, _TrainerFactory())
        first = wrapper._create_trial_rc({"hidden_dim": 64})
        second = wrapper._create_trial_rc({"hidden_dim": 16})
        assert first.model.hidden_dim == 64
        assert second.model.hidden_dim == 16
        assert first is not second


# --- _extract_metric (single-objective) ---


class TestExtractMetric:
    def test_pruning_callback_best_value_wins(self, make_run_config):
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.5})
        )
        value = wrapper._extract_metric(trainer, _FakePruningCallback(best_value=0.9))
        assert value == 0.9

    def test_early_stopping_metrics_used_as_fallback(self, make_run_config):
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.8})
        )
        value = wrapper._extract_metric(trainer, _FakePruningCallback(best_value=None))
        assert value == 0.8

    def test_refuses_early_stopping_best_score_surrogate(self, make_run_config):
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(
            early_stopping=_FakeEarlyStopping(best_metrics={}, best_score=0.9)
        )
        with pytest.raises(RuntimeError, match="Could not extract metric 'auc'"):
            wrapper._extract_metric(trainer, _FakePruningCallback(best_value=None))

    def test_no_trainer_state_at_all_raises(self, make_run_config):
        wrapper = _make_wrapper(make_run_config(), _TrainerFactory())
        trainer = _FakeTrainer(early_stopping=None)
        with pytest.raises(RuntimeError, match="Could not extract metric 'auc'"):
            wrapper._extract_metric(trainer, None)


# --- _extract_multi (multi-objective) ---


class _StubTracker:
    """Duck-typed MultiMetricTracker: a plain best_values mapping."""

    def __init__(self, best_values):
        self.best_values = best_values


class TestExtractMulti:
    def test_missing_tracker_raises(self, make_run_config):
        wrapper = TrainerObjectiveWrapper(
            trainer_class=_TrainerFactory(),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        with pytest.raises(RuntimeError, match="tracker was not registered"):
            wrapper._extract_multi(None)

    def test_missing_metric_raises_with_tracked_summary(self, make_run_config):
        wrapper = TrainerObjectiveWrapper(
            trainer_class=_TrainerFactory(),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        tracker = _StubTracker({"auc": 0.7, "rmse": None})
        with pytest.raises(RuntimeError, match="Could not extract metric 'rmse'"):
            wrapper._extract_multi(tracker)

    def test_returns_per_objective_values(self, make_run_config):
        wrapper = TrainerObjectiveWrapper(
            trainer_class=_TrainerFactory(),
            data_src_fn=lambda: None,
            base_rc=make_run_config(),
            metric_name=["auc", "rmse"],
        )
        assert wrapper._extract_multi(_StubTracker({"auc": 0.7, "rmse": 0.2})) == [
            0.7,
            0.2,
        ]


# --- __call__ integration on fake trainers ---


class TestCall:
    def test_happy_path_returns_metric_and_registers_callback(
        self, make_run_config, fake_trial
    ):
        factory = _TrainerFactory(
            early_stopping=_FakeEarlyStopping(best_metrics={"auc": 0.42})
        )
        wrapper = _make_wrapper(make_run_config(), factory)
        value = wrapper(fake_trial, params={"hidden_dim": 64})
        assert value == 0.42
        # one pruning callback registered; params landed on the trial rc
        assert len(factory.instances[0].callbacks) == 1
        assert isinstance(factory.instances[0].callbacks[0].trial, type(fake_trial))
        assert factory.instances[0].rc.model.hidden_dim == 64
        assert fake_trial.user_attrs["best_epoch"] is None
        assert "duration_sec" in fake_trial.user_attrs

    def test_error_recorded_in_user_attrs_and_reraised(
        self, make_run_config, fake_trial
    ):
        factory = _TrainerFactory(run_error=ValueError("bad dataset"))
        wrapper = _make_wrapper(make_run_config(), factory)
        with pytest.raises(ValueError, match="bad dataset"):
            wrapper(fake_trial)
        assert "ValueError('bad dataset')" in fake_trial.user_attrs["error"]
        assert "ValueError" in fake_trial.user_attrs["traceback"]
        assert "duration_sec" in fake_trial.user_attrs

    def test_trial_pruned_reraised_without_error_attr(
        self, make_run_config, fake_trial
    ):
        factory = _TrainerFactory(run_error=optuna.TrialPruned("mid-run"))
        wrapper = _make_wrapper(make_run_config(), factory)
        with pytest.raises(optuna.TrialPruned):
            wrapper(fake_trial)
        assert "error" not in fake_trial.user_attrs
