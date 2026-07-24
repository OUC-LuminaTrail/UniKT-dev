"""Tests for ``utils.training.metrics``: base-class gating + per-metric behaviour.

Covers the template-method refactor: every metric inherits identical
data-sufficiency gating from the base (empty / single-class / non-finite
-> key omitted), and ``score`` is a pure function.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.training.metrics.accumulator import MetricsAccumulator
from utils.training.metrics.accuracy import AccuracyMetric
from utils.training.metrics.auc import AUCMetric
from utils.training.metrics.auprc import AUPRCMetric
from utils.training.metrics.base import Metric, MetricContext
from utils.training.metrics.kappa import KappaMetric
from utils.training.metrics.mae import MAEMetric
from utils.training.metrics.r2 import R2Metric
from utils.training.metrics.rmse import RMSEMetric


def _ctx(y_label, y_pred=None, y_score=None, y_prob=None, groups=None):
    def a(x):
        return None if x is None else np.asarray(x)

    return MetricContext(
        phase="val",
        y_label=np.asarray(y_label),
        y_pred=a(y_pred),
        y_score=a(y_score),
        y_prob=a(y_prob),
        groups=groups,
    )


# ---------------------------------------------------------------------------
# base-class gating (_eval)
# ---------------------------------------------------------------------------


class _ConstMetric(Metric):
    """Test double returning a fixed value, to probe ``_eval`` gating."""

    name = "const"
    source = "y_pred"
    value = 1.0

    def score(self, y_true, y_value):
        return self.value


class TestBaseGating:
    def test_empty_input_omits_key(self):
        assert _ConstMetric().compute(_ctx(np.array([]), y_pred=np.array([]))) == {}

    def test_single_class_omitted_when_required(self):
        class M(_ConstMetric):
            requires_two_classes = True

        assert M().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1])) == {}

    def test_two_classes_kept_when_required(self):
        class M(_ConstMetric):
            requires_two_classes = True

        assert M().compute(_ctx([1, 0, 1], y_pred=[1, 0, 1])) == {"const": 1.0}

    def test_nan_result_omitted(self):
        class M(_ConstMetric):
            value = float("nan")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_inf_result_omitted(self):
        class M(_ConstMetric):
            value = float("inf")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_score_value_error_omitted(self):
        class M(Metric):
            name = "boom"
            source = "y_pred"

            def score(self, y_true, y_value):
                raise ValueError("boom")

        assert M().compute(_ctx([1, 0], y_pred=[1, 0])) == {}

    def test_threshold_binarises_prediction(self):
        seen = {}

        class M(Metric):
            name = "t"
            source = "y_pred"
            threshold = 0.5

            def score(self, y_true, y_value):
                seen["v"] = y_value
                return 1.0

        M().compute(_ctx([1, 0], y_pred=[0.2, 0.7]))
        assert np.array_equal(seen["v"], [0.0, 1.0])


# ---------------------------------------------------------------------------
# per-metric values (known input -> known output)
# ---------------------------------------------------------------------------


class TestMetricValues:
    def test_auc_perfect_ranking(self):
        assert AUCMetric().compute(_ctx([1, 0], y_score=[0.9, 0.1])) == {"auc": 1.0}

    def test_auc_reversed_ranking(self):
        assert AUCMetric().compute(_ctx([1, 0], y_score=[0.1, 0.9])) == {"auc": 0.0}

    def test_auprc_perfect(self):
        out = AUPRCMetric().compute(
            _ctx([1, 0, 1, 0], y_score=[0.9, 0.1, 0.8, 0.2])
        )
        assert out["auprc"] == 1.0

    def test_acc_partial(self):
        out = AccuracyMetric().compute(_ctx([1, 0, 1], y_pred=[1, 0, 0]))
        assert out["acc"] == pytest.approx(2 / 3)

    def test_acc_binary_pred_unchanged_by_threshold(self):
        # y_pred already 0/1 -> binarisation at 0.5 is a no-op
        assert AccuracyMetric().compute(_ctx([1, 0, 1], y_pred=[1, 0, 1])) == {
            "acc": 1.0
        }

    def test_mae(self):
        out = MAEMetric().compute(_ctx([0, 1], y_prob=[0.1, 0.9]))
        assert out["mae"] == pytest.approx(0.1)

    def test_rmse_perfect(self):
        assert RMSEMetric().compute(_ctx([0, 1], y_prob=[0.0, 1.0])) == {"rmse": 0.0}

    def test_kappa_perfect_agreement(self):
        assert KappaMetric().compute(_ctx([1, 0, 1, 0], y_pred=[1, 0, 1, 0])) == {
            "kappa": 1.0
        }

    def test_r2_perfect_correlation(self):
        out = R2Metric().compute(_ctx([1, 2, 3], y_prob=[0.1, 0.2, 0.3]))
        assert out["r2"] == pytest.approx(1.0)

    def test_r2_zero_variance_returns_zero(self):
        # constant y_label -> zero variance -> r2 = 0.0 (kept, not omitted)
        out = R2Metric().compute(_ctx([1, 1, 1], y_prob=[0.2, 0.5, 0.8]))
        assert out == {"r2": 0.0}


# ---------------------------------------------------------------------------
# the bugs being fixed: single-class & empty input
# ---------------------------------------------------------------------------


class TestSingleClassAndEmpty:
    def test_auc_single_class_omitted(self):
        out = AUCMetric().compute(_ctx([1, 1, 1], y_score=[0.5, 0.6, 0.7]))
        assert out == {}

    def test_auprc_single_class_omitted(self):
        out = AUPRCMetric().compute(_ctx([1, 1, 1], y_score=[0.5, 0.6, 0.7]))
        assert out == {}

    def test_kappa_single_class_omitted(self):
        out = KappaMetric().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1]))
        assert out == {}

    def test_acc_single_class_defined(self):
        # acc is well-defined on a single class (all-correct => 1.0)
        out = AccuracyMetric().compute(_ctx([1, 1, 1], y_pred=[1, 1, 1]))
        assert out == {"acc": 1.0}

    def test_mae_single_class_defined(self):
        out = MAEMetric().compute(_ctx([1, 1, 1], y_prob=[0.9, 0.8, 0.7]))
        assert out["mae"] == pytest.approx(0.2)

    @pytest.mark.parametrize(
        "metric",
        [AUCMetric, AUPRCMetric, AccuracyMetric, MAEMetric, RMSEMetric, KappaMetric, R2Metric],
    )
    def test_empty_input_omits_key(self, metric):
        out = metric().compute(
            _ctx(
                np.array([]),
                y_pred=np.array([]),
                y_score=np.array([]),
                y_prob=np.array([]),
            )
        )
        assert out == {}


# ---------------------------------------------------------------------------
# test/group mode (ctx.groups)
# ---------------------------------------------------------------------------


def _group_ctx():
    groups = {
        "mean": (np.array([1.0, 0.0]), np.array([0.9, 0.1])),
        "vote": (np.array([1.0, 0.0]), np.array([0.8, 0.2])),
    }
    return MetricContext(phase="test", y_label=np.array([1.0, 0.0]), groups=groups)


class TestGroupMode:
    def test_auc_emits_per_fusion(self):
        out = AUCMetric().compute(_group_ctx())
        assert set(out) == {"mean_auc", "vote_auc"}
        assert out["mean_auc"] == 1.0

    def test_acc_binarises_group_score(self):
        out = AccuracyMetric().compute(_group_ctx())
        # scores 0.9 / 0.1 binarise to 1 / 0, matching labels 1 / 0
        assert out["mean_acc"] == 1.0

    def test_single_class_group_omits_auc(self):
        groups = {"mean": (np.array([1.0, 1.0]), np.array([0.9, 0.8]))}
        ctx = MetricContext(phase="test", y_label=np.array([1.0, 1.0]), groups=groups)
        assert AUCMetric().compute(ctx) == {}


# ---------------------------------------------------------------------------
# end-to-end: MetricsAccumulator over a single-class epoch
# ---------------------------------------------------------------------------


def _batch(y_label, y_pred, y_score, y_prob):
    def t(x):
        return torch.tensor(x, dtype=torch.float32)

    return {
        "y_label": t(y_label),
        "y_predict": t(y_pred),
        "y_score": t(y_score),
        "y_prob": t(y_prob),
    }


class TestAccumulator:
    def test_single_class_epoch_skips_undefined_metrics(self):
        # all labels 1 -> AUC / AUPRC / Kappa undefined, must be absent while
        # acc / mae / rmse remain present and valid
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update(
            "val",
            _batch([1, 1, 1, 1], [1, 1, 0, 1], [0.6, 0.7, 0.4, 0.8], [0.6, 0.7, 0.4, 0.8]),
        )
        m = accum.compute("val")
        assert "auc" not in m
        assert "auprc" not in m
        assert "kappa" not in m
        assert {"acc", "mae", "rmse"} <= set(m)

    def test_normal_epoch_emits_all_metrics(self):
        accum = MetricsAccumulator()
        accum.reset("val")
        accum.update(
            "val",
            _batch([1, 0, 1, 0], [1, 0, 0, 0], [0.9, 0.1, 0.8, 0.2], [0.9, 0.1, 0.8, 0.2]),
        )
        m = accum.compute("val")
        assert {"acc", "auc", "auprc", "mae", "rmse", "r2", "kappa"} <= set(m)
