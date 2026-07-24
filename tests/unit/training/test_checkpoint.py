"""Tests for ``utils.training.checkpoint``: queue coalescing + lifecycle edges.

Focuses on the async-save queue coalescing in ``CheckpointManager``: when the
single background worker cannot keep up (slow disk), successive saves to the
same path must collapse so ``close()`` does not drain a long tail of stale,
already-superseded writes. A ``slow_write`` fixture holds the worker on an
``Event`` so pending state is deterministic rather than racy.
"""

from __future__ import annotations

import os
import threading

import pytest
import torch

from utils.training.checkpoint import CheckpointManager


class _SlowGate:
    """Release gate + running flag for the ``slow_write`` fixture.

    ``set`` releases the blocked worker; ``wait_running`` blocks until the
    worker has entered the slow section (the first save is running), so a test
    can assert against a running future without a scheduling race.
    """

    def __init__(self):
        self._release = threading.Event()
        self._running = threading.Event()

    def set(self):
        self._release.set()

    def wait_running(self, timeout=2.0):
        return self._running.wait(timeout)


@pytest.fixture
def slow_write(monkeypatch):
    """Block the saver worker so submits pile up in the queue.

    Yields a gate: the test releases the worker (``gate.set()``) before
    calling ``close()``, and teardown sets it again as a safety net so a
    blocked worker can never hang the executor shutdown.
    """
    real = CheckpointManager._write_atomic
    gate = _SlowGate()

    def _slow(obj, filepath):
        gate._running.set()
        gate._release.wait()
        real(obj, filepath)

    monkeypatch.setattr(CheckpointManager, "_write_atomic", staticmethod(_slow))
    yield gate
    gate.set()


# ---------------------------------------------------------------------------
# coalescing: same-path pending saves are cancelled, running ones are not
# ---------------------------------------------------------------------------


class TestCoalesce:
    def test_same_path_pending_cancelled(self, tmp_path, slow_write):
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        for i in range(5):
            mgr._submit_save({"v": i}, path)

        # only the newest save per path is tracked; older pending ones cancelled
        assert len(mgr._latest) == 1

        slow_write.set()
        mgr.close()

    def test_running_save_not_cancelled(self, tmp_path, slow_write):
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        mgr._submit_save({"v": 0}, path)  # picked up by the worker, now blocked
        assert slow_write.wait_running()  # worker is inside the first save
        running = mgr._latest[path]
        assert not running.done()

        # a newer same-path submit cannot cancel the running future
        mgr._submit_save({"v": 1}, path)
        assert not running.cancelled()

        slow_write.set()
        mgr.close()

    def test_different_paths_kept(self, tmp_path, slow_write):
        mgr = CheckpointManager(str(tmp_path))
        best = str(tmp_path / "best_model.pth")
        last = str(tmp_path / "last_checkpoint.pth")
        mgr._submit_save({"v": 1}, best)
        mgr._submit_save({"v": 1}, last)

        assert best in mgr._latest and last in mgr._latest

        slow_write.set()
        mgr.close()

    def test_newest_value_persisted(self, tmp_path, slow_write):
        # invariant under any scheduling: the last submit wins on disk,
        # because every older pending save to the same path is cancelled
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "last_checkpoint.pth")
        for i in range(10):
            mgr._submit_save({"v": i}, path)

        slow_write.set()
        mgr.close()

        data = torch.load(path, weights_only=True)
        assert data["v"] == 9


# ---------------------------------------------------------------------------
# lifecycle edges
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_close_idempotent(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        mgr.close()
        mgr.close()  # second close must be a no-op

    def test_submit_after_close_writes_synchronously(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        mgr.close()

        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)

        # closed manager bypasses the executor: file exists now, nothing tracked
        assert os.path.exists(path)
        assert mgr._latest == {}

    def test_flush_drains_queue(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)

        mgr.flush()

        assert mgr._latest == {}
        assert os.path.exists(path)

    def test_flush_empty_is_noop(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        mgr.flush()  # draining an empty queue must not raise

    def test_no_tmp_residue(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        path = str(tmp_path / "x.pth")
        mgr._submit_save({"v": 1}, path)
        mgr.close()

        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")


# ---------------------------------------------------------------------------
# save_weights: snapshot semantics survive coalescing
# ---------------------------------------------------------------------------


class TestSaveWeights:
    def test_returns_cpu_snapshot_and_writes(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(3, 2)

        snapshot = mgr.save_weights(model, "best.pth")
        mgr.close()

        assert isinstance(snapshot, dict)
        assert all(t.device.type == "cpu" for t in snapshot.values())
        assert os.path.exists(tmp_path / "best.pth")

    def test_snapshot_returned_every_call_when_coalesced(self, tmp_path, slow_write):
        # disk writes coalesce, but each call still returns the snapshot taken
        # at that moment — it backs the in-memory best-state cache
        mgr = CheckpointManager(str(tmp_path))
        model = torch.nn.Linear(3, 2)

        snapshots = [mgr.save_weights(model, "best.pth") for _ in range(5)]

        assert all(s is not None for s in snapshots)
        slow_write.set()
        mgr.close()


# ---------------------------------------------------------------------------
# error handling: a failed save must not escape flush/close
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_failed_save_does_not_raise_on_close(self, tmp_path, monkeypatch):
        def boom(obj, filepath):
            raise RuntimeError("disk full")

        monkeypatch.setattr(CheckpointManager, "_write_atomic", staticmethod(boom))

        mgr = CheckpointManager(str(tmp_path))
        mgr._submit_save({"v": 1}, str(tmp_path / "x.pth"))

        # exception is logged inside _reap, never propagated to the caller
        mgr.close()
